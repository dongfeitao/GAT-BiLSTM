"""Synthetic CSV/MAT tests; no original bearing data required."""
from pathlib import Path
import json
import uuid
import numpy as np
from scipy.io import savemat
from bearing.data import data_read, temporal_intervals, prepare, scan_ottawa


def main():
    root=Path('smoke_artifacts')/('DATA_SYNTHETIC_ONLY_'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    intervals=temporal_intervals(1024000,10240)
    assert [(b-a)//1024 for a,b in intervals.values()]==[595,190,195]
    for sep in [',','\t']:
        p=root/('tab.csv' if sep=='\t' else 'comma.csv')
        p.write_text('header\n'*5+sep.join(['1','2','3','4','5'])+'\n',encoding='utf-8')
        assert data_read(p).values.tolist()==[[1.,2.,3.,4.]]
    rng=np.random.default_rng(13)
    dirs={}
    for state in ['B','C','H','I','O']:
        directory=root/state;directory.mkdir();dirs[state]=str(directory.resolve())
        for trial in [1,2,3]:
            savemat(directory/f'{state}A{trial}.mat',{'Channel_1':rng.normal(size=(2048,1)),'Channel_2':np.ones((2048,1))})
    config={'window_length':1024,'ottawa':{'sampling_frequency':200000,'samples_per_record':2048,
                'directories':dirs,'condition':'A','grouping':'recording','channel':'Channel_1'}}
    c=root/'config.json';c.write_text(json.dumps(config),encoding='utf-8')
    scan_ottawa(config['ottawa'],root/'map.csv')
    manifest=prepare(c,root/'cache','ottawa',str(root/'map.csv'))
    assert len(manifest['records'])==15
    assert manifest['physical_bearing_verified'] is False
    for split in ['train','val','test']:
        assert sum(r['n_windows'] for r in manifest['records'] if r['split']==split)==10
    norm=json.loads((root/'cache'/'normalization.json').read_text())
    assert norm['raw_samples']==5*2048
    try:prepare(c,root/'strict_cache','ottawa',str(root/'map.csv'),'bearing')
    except ValueError:pass
    else:raise AssertionError('Missing bearing identity accepted')
    print('PASS: CSV/tab first five skipped/first four retained; lab595/190/195; MAT15record partition; train-only normalization; strict bearing rejection')


if __name__=='__main__':main()
