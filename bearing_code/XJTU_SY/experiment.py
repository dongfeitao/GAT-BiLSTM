"""Complete experiment: prepare -> benchmark -> train -> freeze -> test -> figures."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error
from data import config_load,prepare,load_manifest,BearingData,write_json
from model import PrognosticGATBiLSTM

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def inputs(data,start,end,cfg,norm,device):
    lo=start-cfg['history']+1
    assert lo>=0 and end<=data.n
    result=[]
    for name in ['time','freq']:
        a=np.array(getattr(data,name)[lo:end],copy=True)
        a-=norm[name]['mean'];a/=norm[name]['std']
        result.append(torch.from_numpy(a).to(device))
    return result

def amp_context(cfg,device):
    return torch.autocast(device_type=device.type,dtype=torch.float16,
                          enabled=bool(cfg['amp'] and device.type=='cuda'))

@torch.no_grad()
def predict(model,dataset,cfg,norm,device):
    model.eval();rows={}
    for bid,data in dataset.items():
        parts=[]
        for _,start,end in data.blocks(cfg['block_size']):
            t,f=inputs(data,start,end,cfg,norm,device)
            with amp_context(cfg,device):p=model.forward_block(t,f,cfg['history'])
            parts.append(p.float().cpu().numpy())
        pred=np.concatenate(parts)
        if not np.isfinite(pred).all():raise FloatingPointError(f'Nonfinite prediction for {bid}')
        frame=data.obs.iloc[data.first:].copy()
        if model.task=='stage':
            probs=torch.from_numpy(pred).softmax(-1).numpy()
            frame['predicted_stage']=pred.argmax(-1)
            for k in range(3):frame[f'probability_stage_{k}']=probs[:,k]
            frame['predicted_degradation_index']=(probs*np.arange(3)[None,:]).sum(-1)/2
        else:frame['predicted_rul_minutes']=pred*norm['rul_scale_minutes']
        rows[bid]=frame
    return rows

def scores(predictions,task):
    values=[]
    for bid,f in predictions.items():
        if task=='stage':
            values.append(float(f1_score(f.proxy_stage,f.predicted_stage,labels=[0,1,2],average='macro',zero_division=0)))
        else:values.append(float(mean_absolute_error(f.rul_minutes,f.predicted_rul_minutes)))
    return float(np.mean(values))

def benchmark(cfg,norm,dataset,device,out):
    seed_all(42);model=PrognosticGATBiLSTM(cfg,'rul').to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    data=next(d for d in dataset.values() if d.n>cfg['history']+cfg['block_size'])
    start=data.first;end=min(start+cfg['block_size'],data.n)
    t,f=inputs(data,start,end,cfg,norm,device)
    times=[]
    for i in range(5):
        optimizer.zero_grad(set_to_none=True);sync(device);tic=time.perf_counter()
        with amp_context(cfg,device):loss=model.forward_block(t,f,cfg['history']).mean()
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();sync(device)
        if i>0:times.append(time.perf_counter()-tic)
    blocks=sum(len(d.blocks(cfg['block_size'])) for d in dataset.values())
    sec=float(np.median(times));report={'train_block_seconds':sec,'train_blocks_per_epoch':blocks,
        'estimated_train_seconds_per_epoch':sec*blocks,
        'estimated_max_training_hours_without_validation':sec*blocks*cfg['epochs']*2*len(cfg['seeds'])/3600,
        'note':'Measured warm GPU/CPU blocks. Validation, I/O, checkpointing add overhead; early stopping reduces epochs.',
        'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    write_json(out/'benchmark.json',report);print('BENCHMARK '+json.dumps(report),flush=True)
    del model,optimizer,t,f
    if device.type=='cuda':torch.cuda.empty_cache()

def train_task(task,seed,cfg,norm,train,val,device,out,status):
    folder=out/f'seed_{seed}'/task;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        print(f'RESUME completed {task} seed={seed}',flush=True);return
    seed_all(seed);model=PrognosticGATBiLSTM(cfg,task).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    counts=np.bincount(np.concatenate([d.obs.proxy_stage.to_numpy()[d.first:] for d in train.values()]),minlength=3)
    if task=='stage' and np.any(counts==0):raise ValueError(f'Training data missing a proxy class: {counts.tolist()}')
    weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights/=weights.mean()
    loss_fn=(torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
             if task=='stage' else torch.nn.L1Loss())
    blocks=sum([d.blocks(cfg['block_size']) for d in train.values()],[])
    best=-1. if task=='stage' else float('inf');bad=0;history=[];first_epoch=1
    if (folder/'last.pt').exists():
        ck=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        best,bad,history=ck['best'],ck['bad'],ck['history'];first_epoch=ck['epoch']+1
        torch.set_rng_state(ck['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    elapsed_prior=sum(x['seconds'] for x in history)
    for epoch in range(first_epoch,cfg['epochs']+1):
        tic=time.perf_counter();model.train();loss_sum=0.;samples=0
        order=np.random.default_rng(seed+epoch).permutation(len(blocks))
        for step,idx in enumerate(order):
            bid,start,end=blocks[idx];data=train[bid]
            t,f=inputs(data,start,end,cfg,norm,device)
            if task=='stage':target=torch.tensor(data.obs.proxy_stage.to_numpy()[start:end],dtype=torch.long,device=device)
            else:target=torch.tensor(data.obs.rul_minutes.to_numpy()[start:end]/norm['rul_scale_minutes'],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(cfg,device):
                pred=model.forward_block(t,f,cfg['history']);loss=loss_fn(pred.float(),target)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {bid} {start}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*(end-start);samples+=end-start
            if (step+1)%50==0:
                print(f'TRAIN seed={seed} {task} epoch={epoch} block={step+1}/{len(blocks)}',flush=True)
                write_json(out/'status.json',{**status,'phase':'training','task':task,'seed':seed,'epoch':epoch,
                           'block':step+1,'blocks':len(blocks),'best_validation_metric':None if not np.isfinite(best) else best})
        vp=predict(model,val,cfg,norm,device);metric=scores(vp,task)
        seconds=time.perf_counter()-tic
        improved=metric>best+1e-6 if task=='stage' else metric<best-1e-6
        if improved:
            best=metric;bad=0
            torch.save({'model':model.state_dict(),'task':task,'seed':seed,'epoch':epoch,'validation_metric':metric,
                        'config':cfg,'normalization':norm},folder/'best.pt')
        else:bad+=1
        history.append({'epoch':epoch,'train_loss':loss_sum/samples,'validation_metric':metric,
                        'seconds':seconds,'selected':bool(improved)})
        pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch,'best':best,'bad':bad,'history':history,'rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},folder/'last.pt')
        write_json(out/'status.json',{**status,'phase':'training','task':task,'seed':seed,'epoch':epoch,
                   'epoch_seconds':seconds,'best_validation_metric':best,'patience_used':bad,
                   'remaining_task_upper_minutes':(cfg['epochs']-epoch)*seconds/60})
        print(f'EPOCH seed={seed} {task} {epoch}: loss={loss_sum/samples:.5f} val={metric:.5f} {seconds:.1f}s best={best:.5f}',flush=True)
        if bad>=cfg['patience']:break
    bestck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    write_json(folder/'complete.json',{'task':task,'seed':seed,'epochs':len(history),'best_epoch':bestck['epoch'],
               'validation_metric':best,'seconds':sum(x['seconds'] for x in history),
               'train_class_counts':counts.tolist(),'class_weights':weights.tolist() if task=='stage' else None,
               'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()})
    del model,optimizer
    if device.type=='cuda':torch.cuda.empty_cache()

def evaluate(cfg,norm,device,out):
    test={b:BearingData(cfg,b) for b in cfg['split']['test']};metrics=[]
    # All models have been fixed using validation before any test evaluation.
    frozen={}
    for seed in cfg['seeds']:
        frozen[str(seed)]={t:json.loads((out/f'seed_{seed}'/t/'complete.json').read_text()) for t in ['stage','rul']}
    write_json(out/'FREEZE.json',frozen)
    for seed in cfg['seeds']:
        preds={}
        for task in ['stage','rul']:
            folder=out/f'seed_{seed}'/task
            model=PrognosticGATBiLSTM(cfg,task).to(device)
            ck=torch.load(folder/'best.pt',map_location=device,weights_only=False)
            model.load_state_dict(ck['model']);preds[task]=predict(model,test,cfg,norm,device)
            del model
        for bid in cfg['split']['test']:
            f=preds['stage'][bid]
            f['predicted_rul_minutes']=preds['rul'][bid].predicted_rul_minutes.to_numpy()
            f.to_csv(out/f'seed_{seed}'/(bid+'_predictions.csv'),index=False)
            metrics.append({'seed':seed,'bearing':bid,'n_predictions':len(f),
                 'accuracy':float(accuracy_score(f.proxy_stage,f.predicted_stage)),
                 'macro_f1':float(f1_score(f.proxy_stage,f.predicted_stage,labels=[0,1,2],average='macro',zero_division=0)),
                 'rul_mae_minutes':float(mean_absolute_error(f.rul_minutes,f.predicted_rul_minutes)),
                 'stage0_support':int((f.proxy_stage==0).sum()),'stage1_support':int((f.proxy_stage==1).sum()),
                 'stage2_support':int((f.proxy_stage==2).sum())})
    pd.DataFrame(metrics).to_csv(out/'test_metrics_by_seed.csv',index=False)
    from figures import make_figures
    make_figures(cfg,out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','benchmark','train','evaluate','all','figures'])
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=config_load(args.config)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare(cfg);return
    if args.command=='figures':
        from figures import make_figures
        make_figures(cfg,out);return
    if args.command=='all':prepare(cfg)
    manifest=load_manifest(cfg);norm=manifest['normalization']
    device=torch.device('cuda' if cfg['device']=='auto' and torch.cuda.is_available() else ('cpu' if cfg['device']=='auto' else cfg['device']))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use a CUDA PyTorch environment or explicitly select cpu.')
    torch.set_num_threads(cfg['cpu_threads'])
    status={'started_local':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),
            'device':str(device),'torch':torch.__version__,'python':platform.python_version(),
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    if args.command in ['train','all']:
        conf=out/'config_resolved.json'
        if conf.exists():
            old=json.loads(conf.read_text(encoding='utf-8'))
            if old!=cfg:raise ValueError('Existing run configuration differs; select a new output_dir.')
            if not args.resume:raise ValueError('Existing run; pass --resume or select a new output_dir.')
        else:write_json(conf,cfg)
        write_json(out/'environment.json',status)
        write_json(out/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    train={b:BearingData(cfg,b) for b in cfg['split']['train']}
    val={b:BearingData(cfg,b) for b in cfg['split']['val']}
    if args.command in ['benchmark','train','all']:
        from figures import settings_table
        settings_table(cfg,out,manifest)
        benchmark(cfg,norm,train,device,out)
    if args.command in ['train','all']:
        for seed in cfg['seeds']:
            for task in ['stage','rul']:train_task(task,seed,cfg,norm,train,val,device,out,status)
    if args.command in ['evaluate','all']:
        evaluate(cfg,norm,device,out)
        write_json(out/'status.json',{**status,'phase':'complete','finished_local':time.strftime('%Y-%m-%d %H:%M:%S')})
        print('COMPLETE: '+str(out),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        import sys
        error=traceback.format_exc()
        try:
            config_path=(sys.argv[sys.argv.index('--config')+1] if '--config' in sys.argv
                         else str(Path(__file__).with_name('config.json')))
            fail_cfg=config_load(config_path)
            write_json(Path(fail_cfg['output_dir'])/'status.json',{'phase':'failed','error':error,
                       'time_local':time.strftime('%Y-%m-%d %H:%M:%S')})
        except Exception:pass
        raise
