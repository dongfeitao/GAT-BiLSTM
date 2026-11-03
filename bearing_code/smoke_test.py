"""Synthetic integration check. NEVER use these outputs as paper results."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from bearing.models import make_model, MODEL_NAMES, construct_hypergraph
from bearing.experiments import train_phase, test_phase
from bearing.profiling import profile_neural
from bearing.data import data_read, window_features


def main():
    torch.set_num_threads(1)
    import uuid
    root = Path('smoke_artifacts') / ('SYNTHETIC_ONLY_' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    rng = np.random.default_rng(123)
    x, f = torch.randn(4,1,1024), torch.randn(4,1,512)
    cfg = {'feature_dim':16,'hidden_dim':8,'heads':2,'encoder_chunk_size':2,'transformer_layers':1}
    for name in MODEL_NAMES:
        model = make_model(name,cfg)
        out=model(x,f)
        assert out.shape==(4,5) and torch.isfinite(out).all(), name
        torch.nn.functional.cross_entropy(out,torch.arange(4)).backward()
    H, _ = construct_hypergraph(torch.eye(3), .9)
    assert H.shape==(3,0)
    H, _ = construct_hypergraph(torch.ones(3,4), .7)
    assert H.shape==(3,1) and H.sum()==3
    cache=root/'cache'; cache.mkdir()
    records=[]
    for split in ('train','val','test'):
        for label in range(5):
            filename=f'{split}_{label}.npz'
            np.savez(cache/filename,time=rng.normal(size=(2,1,1024)).astype('float32'),
                     freq=rng.normal(size=(2,1,512)).astype('float32'),y=np.full(2,label))
            records.append(dict(path=filename,split=split,record_id=f'{split}_{label}',label=label,n_windows=2))
    (cache/'manifest.json').write_text(json.dumps({'dataset':'laboratory','records':records}),encoding='utf-8')
    config=root/'config.json'; config.write_text(json.dumps({'base':cfg}),encoding='utf-8')
    run=root/'run'
    train_phase(SimpleNamespace(cache=str(cache),output=str(run),dataset='laboratory',suite='module',
                  models='full',config=str(config),epochs=1,patience=0,budget=1,no_tune=True,device='cpu'))
    assert not list(run.rglob('test_predictions.npz'))
    test_phase(SimpleNamespace(cache=str(cache),output=str(run),device='cpu',plot_all=False))
    assert len(list(run.rglob('test_predictions.npz')))==5
    try:
        test_phase(SimpleNamespace(cache=str(cache),output=str(run),device='cpu',plot_all=False))
    except FileExistsError:
        pass
    else:
        raise AssertionError('Repeated test was not blocked')
    result=profile_neural(make_model('full',cfg),x,f,torch.device('cpu'),1,2)
    assert result['major_arithmetic_flops_per_record']>0
    (root/'smoke_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS: 13 forward/backward models; isolated/deduplicated graphs; five-seed train/freeze/test; confusion plots; profiling.')
    print('SYNTHETIC ONLY artifacts:',root)


if __name__=='__main__':main()

