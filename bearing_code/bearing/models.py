"""Record-wise models for the manuscript implementation.

Inputs: time [N,1,1024], frequency [N,1,512], chronologically ordered
windows from ONE record/retained segment. Outputs: logits [N,5].
No labels or cross-record memory are used to build graphs. Logits, not
Softmax probabilities, are passed to cross_entropy during training.

The baselines below are explicitly specified reference-style models. They
are not claimed to reproduce unnamed/unavailable original implementations.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function


class RecordModel(nn.Module):
    """Override stage_context with a context-manager factory for timing."""
    def __init__(self):
        super().__init__()
        self.stage_context = record_function

    @staticmethod
    def check_inputs(time: Tensor, freq: Tensor):
        if time.ndim != 3 or freq.ndim != 3:
            raise ValueError('Expected [number_of_windows, 1, window_length] arrays')
        if time.shape[0] != freq.shape[0] or time.shape[0] == 0:
            raise ValueError('Time/frequency inputs must have equal nonzero window counts')
        if time.shape[1] != 1 or freq.shape[1] != 1:
            raise ValueError('This implementation uses one selected vibration channel')

    def predict_proba(self, time: Tensor, freq: Tensor) -> Tensor:
        return self(time, freq).softmax(dim=-1)


class WindowEncoder(nn.Module):
    """Independent single-layer CNN/GAP encoders; chunking preserves gradients.

    encoder_chunk_size controls ONLY convolution work batches. It never
    truncates the record graph or BiLSTM sequence. For full record training
    all chunk results remain connected to autograd until the loss backward.
    """
    def __init__(self, channels=100, chunk_size=32, feature_mode='both'):
        super().__init__()
        if feature_mode not in ('both', 'time', 'frequency'):
            raise ValueError(feature_mode)
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError('encoder_chunk_size must be positive')
        self.feature_mode = feature_mode
        self.time_conv = nn.Conv1d(1, self.channels, 3, padding=1)
        self.freq_conv = nn.Conv1d(1, self.channels, 3, padding=1)
        # Disabled domains retain their zero feature slots, but their unused
        # encoder parameters are excluded from trainable-parameter counts.
        if feature_mode == 'time':
            self.freq_conv.requires_grad_(False)
        elif feature_mode == 'frequency':
            self.time_conv.requires_grad_(False)

    def encode(self, x: Tensor, conv: nn.Conv1d) -> Tensor:
        return torch.cat([F.relu(conv(c)).mean(dim=-1)
                          for c in x.split(self.chunk_size, dim=0)], dim=0)

    def forward(self, time: Tensor, freq: Tensor):
        zero = time.new_zeros((len(time), self.channels))
        t = self.encode(time, self.time_conv) if self.feature_mode != 'frequency' else zero
        f = self.encode(freq, self.freq_conv) if self.feature_mode != 'time' else zero
        return t, f, torch.cat((t, f), dim=-1)


@torch.no_grad()
def construct_hypergraph(features: Tensor, tau: float = .7):
    """Exact threshold hyperedges, deduplicated deterministically (Algorithm 1).

    For center i, form {i} U {j != i: cosine(i,j)>=tau} only if at least
    one neighbor exists. Isolated nodes therefore have zero incidence.
    H is [N,M], float, and adjacency is the direct threshold graph plus
    self loops used for the GAT neighbor mask. Structural comparisons are
    nondifferentiable; the subsequent feature aggregation is differentiable.
    """
    if not -1 <= tau <= 1:
        raise ValueError('tau must lie in [-1,1]')
    n = len(features)
    z = F.normalize(features.detach(), p=2, dim=-1, eps=1e-8)
    sim = z @ z.transpose(0, 1)
    neighbors = sim >= tau
    neighbors.fill_diagonal_(False)
    valid = neighbors.any(dim=1)
    adjacency = neighbors.clone()
    adjacency.fill_diagonal_(True)
    if valid.any():
        # Each row is a membership indicator; unique rows retain one copy
        # of equal node sets. Ordering is stable and independent of labels.
        memberships = torch.unique(adjacency[valid], sorted=True, dim=0)
        incidence = memberships.transpose(0, 1).to(dtype=features.dtype)
    else:
        incidence = features.new_zeros((n, 0))
    return incidence, adjacency


class HypergraphAggregation(nn.Module):
    """ReLU(W0 h + W1 mean_incident_edges(mean_member_nodes(h)))."""
    def __init__(self, in_dim=200, out_dim=128):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neighbor_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor, incidence: Tensor):
        if incidence.shape[1] == 0:
            return F.relu(self.self_linear(x))
        edge_size = incidence.sum(dim=0).clamp_min(1)
        node_degree = incidence.sum(dim=1).clamp_min(1)
        edge_means = (incidence.transpose(0, 1) @ x) / edge_size[:, None]
        node_means = (incidence @ edge_means) / node_degree[:, None]
        return F.relu(self.self_linear(x) + self.neighbor_linear(node_means))


class MultiHeadGraphAttention(nn.Module):
    """Additive GAT heads, each with its own linear/attention parameters."""
    def __init__(self, in_dim=128, head_dim=128, heads=4, dropout=0.):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        if self.heads < 1:
            raise ValueError('heads must be positive')
        self.projection = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, head_dim))
        self.attn_target = nn.Parameter(torch.empty(heads, head_dim))
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor):
        h = self.projection(x).reshape(len(x), self.heads, self.head_dim)
        outputs = []
        # Explicit head loop avoids retaining an extra [H,N,N,D] tensor.
        for k in range(self.heads):
            v = h[:, k]
            source = (v * self.attn_source[k]).sum(dim=-1)
            target = (v * self.attn_target[k]).sum(dim=-1)
            scores = F.leaky_relu(source[:, None] + target[None, :], .2)
            scores = scores.masked_fill(~adjacency, -torch.inf)
            weights = self.dropout(scores.softmax(dim=-1))
            outputs.append(F.relu(weights @ v))
        return torch.cat(outputs, dim=-1)


class ParallelAttention(nn.Module):
    def __init__(self, dim=128, heads=4, branch_dim=100, fusion='variance', dropout=0.):
        super().__init__()
        if fusion not in ('variance', 'equal', 'learnable'):
            raise ValueError('fusion must be variance, equal or learnable')
        self.dim = dim
        self.fusion = fusion
        self.graph_attention = MultiHeadGraphAttention(dim, dim, heads, dropout)
        self.graph_projection = nn.Linear(heads * dim, dim)
        self.query = nn.Linear(branch_dim, dim, bias=False)
        self.key = nn.Linear(branch_dim, dim, bias=False)
        self.value = nn.Linear(branch_dim, dim, bias=False)
        if fusion == 'learnable':
            self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, graph_features, adjacency, time_features, freq_features):
        graph = self.graph_projection(self.graph_attention(graph_features, adjacency))
        q, k, v = self.query(time_features), self.key(freq_features), self.value(freq_features)
        # Softmax is over frequency-domain key windows in this same record.
        cross = ((q @ k.transpose(0, 1)) / math.sqrt(self.dim)).softmax(dim=-1) @ v
        if self.fusion == 'variance':
            vg = graph.var(unbiased=False)
            vc = cross.var(unbiased=False)
            denominator = vg + vc
            # Population variance over all N*dim entries (Eq. S25).
            weight = torch.where(denominator > 0,
                                 vg / denominator.clamp_min(torch.finfo(vg.dtype).tiny),
                                 denominator.new_tensor(.5))
        elif self.fusion == 'equal':
            weight = .5
        else:
            weight = self.gate.sigmoid()
        return weight * graph + (1 - weight) * cross


class GATBiLSTM(RecordModel):
    """Manuscript full model; default tensor sizes 200->128->128->256->5."""
    def __init__(self, cfg: dict[str, Any], variant='full'):
        super().__init__()
        dim = int(cfg.get('feature_dim', 128))
        hidden = int(cfg.get('hidden_dim', 128))
        heads = int(cfg.get('heads', 4))
        self.tau = float(cfg.get('tau', .7))
        self.variant = variant
        mode = {'time_only': 'time', 'frequency_only': 'frequency'}.get(variant, 'both')
        self.encoder = WindowEncoder(100, cfg.get('encoder_chunk_size', 32), mode)
        self.graph = HypergraphAggregation(200, dim)
        self.attention = (None if variant == 'gnn_bilstm' else
                          ParallelAttention(dim, heads, 100, cfg.get('fusion', 'variance'),
                                            float(cfg.get('dropout', 0.))))
        self.sequence = nn.LSTM(dim, hidden, num_layers=1, batch_first=True,
                                bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, int(cfg.get('num_classes', 5)))

    def forward(self, time: Tensor, freq: Tensor):
        self.check_inputs(time, freq)
        with self.stage_context('encoding'):
            t, f, x = self.encoder(time, freq)
        with self.stage_context('graph_construction'):
            incidence, adjacency = construct_hypergraph(x, self.tau)
        with self.stage_context('graph_aggregation'):
            x = self.graph(x, incidence)
        if self.attention is not None:
            with self.stage_context('attention_fusion'):
                x = self.attention(x, adjacency, t, f)
        with self.stage_context('sequence'):
            x, _ = self.sequence(x.unsqueeze(0))  # B=1; T=all record windows
        with self.stage_context('classification'):
            return self.classifier(x.squeeze(0))


class TemporalBaseline(RecordModel):
    """CNN/GAP features followed by whole-record LSTM/GRU.

    bilstm/lstm ablations keep the full model's 100-channel encoders and
    remove graph/attention. cnn_bilstm/cnn_bigru are configurable reference
    baselines with convolution channels defaulting to 32 per branch.
    """
    def __init__(self, cfg, kind='bilstm', baseline=False):
        super().__init__()
        channels = int(cfg.get('conv_channels', 32)) if baseline else 100
        dim = int(cfg.get('feature_dim', 128))
        hidden = int(cfg.get('hidden_dim', 128))
        self.encoder = WindowEncoder(channels, cfg.get('encoder_chunk_size', 32))
        self.project = nn.Linear(2 * channels, dim)
        bidirectional = kind != 'lstm'
        cell = nn.GRU if kind == 'bigru' else nn.LSTM
        self.sequence = cell(dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden * (2 if bidirectional else 1),
                                    int(cfg.get('num_classes', 5)))

    def forward(self, time, freq):
        self.check_inputs(time, freq)
        with self.stage_context('encoding'):
            _, _, x = self.encoder(time, freq)
            x = F.relu(self.project(x))
        with self.stage_context('sequence'):
            x, _ = self.sequence(x.unsqueeze(0))
        with self.stage_context('classification'):
            return self.classifier(x.squeeze(0))


class CNNBaseline(RecordModel):
    def __init__(self, cfg):
        super().__init__()
        channels = int(cfg.get('conv_channels', 32))
        self.encoder = WindowEncoder(channels, cfg.get('encoder_chunk_size', 32))
        self.classifier = nn.Sequential(nn.Linear(2 * channels, 128), nn.ReLU(),
                                        nn.Linear(128, int(cfg.get('num_classes', 5))))

    def forward(self, time, freq):
        self.check_inputs(time, freq)
        with self.stage_context('encoding'):
            _, _, x = self.encoder(time, freq)
        with self.stage_context('classification'):
            return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        # No BatchNorm: whole-record/segment independence remains explicit.
        self.conv1 = nn.Conv1d(incoming, outgoing, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(outgoing, outgoing, 3, padding=1)
        self.skip = (nn.Identity() if incoming == outgoing and stride == 1
                     else nn.Conv1d(incoming, outgoing, 1, stride=stride))

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + self.skip(x))


class ResNetBaseline(RecordModel):
    """Explicit compact 1D residual reference baseline, two independent branches."""
    def __init__(self, cfg):
        super().__init__()
        c = int(cfg.get('conv_channels', 32))
        self.chunk_size = int(cfg.get('encoder_chunk_size', 32))
        def branch():
            return nn.Sequential(ResidualBlock(1, c, 2), ResidualBlock(c, c * 2, 2),
                                 ResidualBlock(c * 2, c * 2), nn.AdaptiveAvgPool1d(1))
        self.time_encoder = branch()
        self.freq_encoder = branch()
        self.classifier = nn.Linear(c * 4, int(cfg.get('num_classes', 5)))

    def forward(self, time, freq):
        self.check_inputs(time, freq)
        with self.stage_context('encoding'):
            t = torch.cat([self.time_encoder(x).squeeze(-1)
                           for x in time.split(self.chunk_size)], dim=0)
            f = torch.cat([self.freq_encoder(x).squeeze(-1)
                           for x in freq.split(self.chunk_size)], dim=0)
        with self.stage_context('classification'):
            return self.classifier(torch.cat((t, f), dim=-1))


class TransformerBaseline(RecordModel):
    def __init__(self, cfg):
        super().__init__()
        dim = int(cfg.get('feature_dim', 128))
        heads = int(cfg.get('heads', 4))
        if dim % heads:
            raise ValueError('Transformer feature_dim must be divisible by heads')
        self.encoder = WindowEncoder(100, cfg.get('encoder_chunk_size', 32))
        self.project = nn.Linear(200, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim,
                    dropout=float(cfg.get('dropout', 0.)), activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, int(cfg.get('transformer_layers', 2)))
        self.classifier = nn.Linear(dim, int(cfg.get('num_classes', 5)))

    @staticmethod
    def positions(n, d, device, dtype):
        position = torch.arange(n, device=device, dtype=dtype)[:, None]
        scale = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype)
                          * (-math.log(10000.) / d))
        pe = torch.zeros(n, d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale[:pe[:, 1::2].shape[1]])
        return pe

    def forward(self, time, freq):
        self.check_inputs(time, freq)
        with self.stage_context('encoding'):
            _, _, x = self.encoder(time, freq)
            x = self.project(x)
        with self.stage_context('sequence'):
            x = x + self.positions(len(x), x.shape[-1], x.device, x.dtype)
            x = self.transformer(x.unsqueeze(0)).squeeze(0)
        with self.stage_context('classification'):
            return self.classifier(x)


class GATBaseline(RecordModel):
    """Threshold-graph single-/multi-head GAT reference, no BiLSTM/hypergraph."""
    def __init__(self, cfg, multi=False):
        super().__init__()
        dim = int(cfg.get('hidden_dim', 128))
        heads = int(cfg.get('heads', 4)) if multi else 1
        self.tau = float(cfg.get('tau', .7))
        self.encoder = WindowEncoder(100, cfg.get('encoder_chunk_size', 32))
        self.attention = MultiHeadGraphAttention(200, dim, heads,
                                                float(cfg.get('dropout', 0.)))
        self.classifier = nn.Linear(heads * dim, int(cfg.get('num_classes', 5)))

    def forward(self, time, freq):
        self.check_inputs(time, freq)
        with self.stage_context('encoding'):
            _, _, x = self.encoder(time, freq)
        with self.stage_context('graph_construction'):
            with torch.no_grad():
                z = F.normalize(x.detach(), dim=-1, eps=1e-8)
                adjacency = z @ z.T >= self.tau
                adjacency.fill_diagonal_(True)
        with self.stage_context('attention_fusion'):
            x = self.attention(x, adjacency)
        with self.stage_context('classification'):
            return self.classifier(x)


MODEL_NAMES = ('full', 'gnn_bilstm', 'bilstm', 'lstm', 'time_only',
               'frequency_only', 'transformer', 'cnn_bilstm', 'cnn_bigru',
               'resnet', 'cnn1d', 'gat', 'mhgat')


def make_model(name: str, cfg: dict[str, Any] | None = None) -> RecordModel:
    """Build a fresh model. Seed the caller before invoking this function."""
    cfg = dict(cfg or {})
    if name in ('full', 'gnn_bilstm', 'time_only', 'frequency_only'):
        return GATBiLSTM(cfg, name)
    if name in ('bilstm', 'lstm'):
        return TemporalBaseline(cfg, name)
    if name in ('cnn_bilstm', 'cnn_bigru'):
        return TemporalBaseline(cfg, name[4:], baseline=True)
    if name == 'cnn1d':
        return CNNBaseline(cfg)
    if name == 'resnet':
        return ResNetBaseline(cfg)
    if name == 'transformer':
        return TransformerBaseline(cfg)
    if name in ('gat', 'mhgat'):
        return GATBaseline(cfg, multi=name == 'mhgat')
    raise ValueError(f'Unknown neural model {name!r}. Choices: {MODEL_NAMES}; SVM is handled by the experiment runner.')
