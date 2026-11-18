"""Measured whole-record latency and major-arithmetic FLOPs of frozen models.

Use validation inputs. Common raw-signal preprocessing is excluded. FLOPs
count convolution/matrix-multiplication arithmetic (2 per multiply-add), not
comparisons, sorting, memory movement or all scalar/nonlinear operations.
Those operations ARE included in measured latency. Never infer latency by
adding individually instrumented stages.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from .experiments import choose_device, dump_json, load_split, record_tensors, svm_arrays, digest
from .models import make_model
from .metrics import write_csv


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def profile_neural(model, x, f, device, warmup, repeats):
    model.eval()
    def call():
        return model(x, f).softmax(-1)
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            sync(device)
            started = time.perf_counter()
            call()
            sync(device)
            times.append(1000 * (time.perf_counter() - started))
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        # Diagnostic stage timing is a separate, synchronized pass.
        stage_times = defaultdict(list)
        @contextmanager
        def timing(name):
            sync(device)
            started = time.perf_counter()
            yield
            sync(device)
            stage_times[name].append(1000 * (time.perf_counter() - started))
        old_context = model.stage_context
        model.stage_context = timing
        try:
            for _ in range(repeats):
                logits = model(x, f)
                with timing('classification_softmax'):
                    logits.softmax(-1)
        finally:
            model.stage_context = old_context
    # Count on CPU with fused RNN/Transformer fast paths disabled, so major
    # arithmetic expands to registered matmul/convolution operations.
    import copy
    count_model = copy.deepcopy(model).to('cpu').eval()
    xc, fc = x.cpu(), f.cpu()
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    stage_flops = defaultdict(int)
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
            with FlopCounterMode(display=False) as counter:
                @contextmanager
                def counting(name):
                    before = counter.get_total_flops()
                    yield
                    stage_flops[name] += counter.get_total_flops() - before
                count_model.stage_context = counting
                count_model(xc, fc).softmax(-1)
            flops = counter.get_total_flops()
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
    return {
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'all_parameters': sum(p.numel() for p in model.parameters()),
        'major_arithmetic_flops_per_record': int(flops),
        'major_arithmetic_flops_per_window': float(flops / len(x)),
        'latency_record_ms_mean': float(np.mean(times)),
        'latency_record_ms_sd': float(np.std(times, ddof=1)),
        'latency_window_ms_mean': float(np.mean(times) / len(x)),
        'peak_cuda_allocated_bytes': peak,
        'stage_ms_mean': {k: float(np.mean(v)) for k, v in stage_times.items()},
        'stage_major_arithmetic_flops': dict(stage_flops),
        'latency_samples_ms': times,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='Experiment directory containing FREEZE.json')
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 2:
        p.error('warmup >= 0 and repeats >= 2 are required')
    run, out = Path(args.run), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new profiling output directory')
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((run / 'FREEZE.json').read_text(encoding='utf-8'))
    rows = load_split(args.cache, 'val')
    device = choose_device(args.device)
    report, details = [], []
    for name, info in frozen['models'].items():
        ck = next(c for c in info['checkpoints'] if c['seed'] == 6)
        path = run / ck['path']
        if digest(path) != ck['sha256']:
            raise ValueError(f'Checkpoint changed: {path}')
        model = joblib.load(path) if name == 'svm' else make_model(name, info['config']).to(device)
        if name != 'svm':
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True)['state_dict'])
        for row in rows:
            if name == 'svm':
                x, _ = svm_arrays([row]); times = []
                for _ in range(args.warmup):
                    model.predict(x)
                for _ in range(args.repeats):
                    start = time.perf_counter(); model.predict(x)
                    times.append((time.perf_counter()-start)*1000)
                result = dict(trainable_parameters=None, all_parameters=None,
                    major_arithmetic_flops_per_record=None, major_arithmetic_flops_per_window=None,
                    latency_record_ms_mean=float(np.mean(times)), latency_record_ms_sd=float(np.std(times, ddof=1)),
                    latency_window_ms_mean=float(np.mean(times)/len(x)), peak_cuda_allocated_bytes=None,
                    note='SVM runs on CPU. Neural trainable parameter count is not applicable. Kernel FLOPs are not estimated.',
                    support_vector_count=int(model[-1].support_vectors_.shape[0]))
            else:
                x, f, _ = record_tensors(row, device)
                result = profile_neural(model, x, f, device, args.warmup, args.repeats)
            details.append(dict(model=name, record_id=row['record_id'], n_windows=len(row['y']), **result))
            report.append({k: v for k,v in details[-1].items() if not isinstance(v,(dict,list))})
    fields = sorted(set().union(*(row.keys() for row in report)))
    write_csv(out/'cost_by_record.csv', [{k: row.get(k) for k in fields} for row in report])
    dump_json(out/'cost_details.json', details)
    dump_json(out/'profiling_protocol.json', dict(torch=torch.__version__,python=platform.python_version(),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        cpu=platform.processor(), threads=torch.get_num_threads(), seed=6, split='val',
        batch_sequences=1, warmup=args.warmup, repeats=args.repeats, dtype='float32',
        boundary='Device-resident preprocessed arrays -> class probabilities; common preprocessing and device transfer excluded.',
        flops_scope='Major matrix/convolution arithmetic only, 2 FLOPs/MAC. Recurrent fused paths disabled for FLOP counting only. Nonlinear/scalar/structural operations excluded from FLOPs but included in timing.',
        stage_note='Synchronized stage runs are diagnostic; do not sum them as whole-model latency. CPU FLOP counting is separate from measured device timing.',
        memory_note='CUDA peak allocated includes model/input resident memory; no CPU peak-memory measurement.',
        freeze_sha256=digest(run/'FREEZE.json')))
    print(out/'cost_by_record.csv')


if __name__ == '__main__':
    main()

