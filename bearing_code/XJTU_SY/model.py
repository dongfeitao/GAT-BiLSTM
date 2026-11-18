"""Numerical time/frequency CNN -> hypergraph -> parallel attention -> BiLSTM.

Adapts the user's supplied implementation to bounded historical records.
Window features are averaged within one CSV; the graph nodes are acquisition times.
Batch dimensions never mix graphs, variances, or temporal sequences.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def hypergraph(x,tau):
    with torch.no_grad():
        z=F.normalize(x.detach().float(),dim=-1,eps=1e-8)
        adj=z@z.transpose(-1,-2)>=tau
        n=x.shape[1];eye=torch.eye(n,device=x.device,dtype=torch.bool)[None]
        neighbor=adj & ~eye
        valid=neighbor.any(-1)
        adj=neighbor|eye
        # Equal membership rows form one hyperedge. Keep first occurrence only.
        equal=(adj[:,:,None,:]==adj[:,None,:,:]).all(-1)
        earlier=torch.tril(torch.ones(n,n,device=x.device,dtype=torch.bool),diagonal=-1)
        keep=valid & ~(equal & earlier[None] & valid[:,None,:]).any(-1)
        incidence=(adj & keep[:,:,None]).transpose(-1,-2).to(x.dtype)
    return incidence,adj

class PrognosticGATBiLSTM(nn.Module):
    def __init__(self,cfg,task):
        super().__init__();self.cfg=cfg;self.task=task
        c=cfg['encoder_channels'];d=cfg['feature_dim'];h=cfg['heads'];hidden=cfg['hidden_dim']
        self.time_conv=nn.Conv1d(1,c,3,padding=1)
        self.freq_conv=nn.Conv1d(1,c,3,padding=1)
        self.self_linear=nn.Linear(2*c,d,bias=False)
        self.neighbor_linear=nn.Linear(2*c,d,bias=False)
        self.gat_linear=nn.Linear(d,h*d,bias=False)
        self.attn_source=nn.Parameter(torch.empty(h,d));self.attn_target=nn.Parameter(torch.empty(h,d))
        nn.init.xavier_uniform_(self.attn_source);nn.init.xavier_uniform_(self.attn_target)
        self.align=nn.Linear(h*d,d)
        self.query=nn.Linear(c,d,bias=False);self.key=nn.Linear(c,d,bias=False);self.value=nn.Linear(c,d,bias=False)
        self.dropout=nn.Dropout(cfg['dropout'])
        self.sequence=nn.LSTM(d,hidden,batch_first=True,bidirectional=True)
        self.output=nn.Linear(2*hidden,3 if task=='stage' else 1)

    def encode_records(self,t,f):
        # [unique acquisition records, 32 short windows, samples]
        records,windows=t.shape[:2]
        def branch(x,conv):
            x=x.reshape(-1,1,x.shape[-1]);parts=[]
            for chunk in x.split(self.cfg['encoder_chunk_size']):
                parts.append(F.relu(conv(chunk)).mean(-1))
            return torch.cat(parts).reshape(records,windows,-1).mean(1)
        return branch(t,self.time_conv),branch(f,self.freq_conv)

    def forward_features(self,t,f):
        # [history batch, history length, encoded domain dimension]
        x=torch.cat([t,f],-1);inc,adj=hypergraph(x,self.cfg['tau'])
        edges=(inc.transpose(-1,-2)@x)/inc.sum(1).clamp_min(1)[...,None]
        neighbors=(inc@edges)/inc.sum(2).clamp_min(1)[...,None]
        x=F.relu(self.self_linear(x)+self.neighbor_linear(neighbors))
        b,n,d=x.shape;heads=self.cfg['heads']
        v=self.gat_linear(x).reshape(b,n,heads,d).transpose(1,2)
        source=(v*self.attn_source[None,:,None,:]).sum(-1)
        target=(v*self.attn_target[None,:,None,:]).sum(-1)
        scores=F.leaky_relu(source[:,:,:,None]+target[:,:,None,:],.2)
        weights=scores.masked_fill(~adj[:,None],-torch.inf).softmax(-1)
        graph=F.relu(self.dropout(weights)@v).transpose(1,2).reshape(b,n,heads*d)
        graph=self.align(graph)
        q,k,v=self.query(t),self.key(f),self.value(f)
        cross=((q@k.transpose(-1,-2))/math.sqrt(d)).softmax(-1)@v
        vg=graph.float().var(dim=(1,2),unbiased=False);vc=cross.float().var(dim=(1,2),unbiased=False)
        weight=torch.where(vg+vc>0,vg/(vg+vc).clamp_min(1e-20),torch.full_like(vg,.5))[:,None,None]
        fused=(weight*graph+(1-weight)*cross).float()
        # Both final states summarize ONLY the provided past/current history.
        # Disable fp16 for the recurrent layer to keep training stable on Windows.
        with torch.autocast(device_type=t.device.type,enabled=False):
            _,(state,_)=self.sequence(fused)
            feature=torch.cat([state[-2],state[-1]],-1)
            out=self.output(feature)
        return out if self.task=='stage' else F.softplus(out[:,0])

    def forward_block(self,t,f,history):
        t,f=self.encode_records(t,f)
        ix=torch.arange(history,device=t.device)[None,:]+torch.arange(len(t)-history+1,device=t.device)[:,None]
        return self.forward_features(t[ix],f[ix])
