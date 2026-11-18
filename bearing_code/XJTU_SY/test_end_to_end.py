"""Small synthetic CLI integration test; its outputs are never paper results."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

base=Path(__file__).resolve().parent
root=base/'tests_generated_synthetic'
cfg=json.loads((base/'config.json').read_text(encoding='utf-8'))
cfg.update({'data_root':str(root/'raw'),'cache_dir':str(root/'cache'),'output_dir':str(root/'results'),
            'record_points':128,'window_points':32,'encoder_channels':4,'feature_dim':8,
            'hidden_dim':8,'heads':2,'block_size':6,'epochs':2,'patience':2,
            'seeds':[9],'device':'cpu','amp':False,'cpu_threads':2})
for bid in sum(cfg['split'].values(),[]):
    p=root/'raw'/bid;p.mkdir(parents=True,exist_ok=True)
    for i in range(20):
        amp=1+.01*i if i<10 else (1.2 if i<14 else 1.6)
        wave=amp*np.sin(np.arange(128)*2*np.pi/16)
        pd.DataFrame({'Horizontal_vibration_signals':wave,'Vertical_vibration_signals':wave*.9}).to_csv(p/f'{i+1}.csv',index=False)
path=root/'synthetic_config.json';path.write_text(json.dumps(cfg),encoding='utf-8')
subprocess.run([sys.executable,str(base/'experiment.py'),'all','--config',str(path),'--resume'],check=True)
out=root/'results'
metrics=pd.read_csv(out/'test_metrics_by_seed.csv')
assert len(metrics)==3 and set(metrics.bearing)==set(cfg['split']['test'])
for name in ['01_experiment_settings','02_test_summary','03_stage_recognition','04_rul_prediction']:
    assert (out/(name+'.png')).stat().st_size>1000
for bid in cfg['split']['test']:
    f=pd.read_csv(out/'seed_9'/(bid+'_predictions.csv'))
    assert len(f)==10 and f.record.iloc[0]==11 and f.rul_minutes.iloc[-1]==0
    assert np.isfinite(f.predicted_rul_minutes).all()
print('SYNTHETIC END-TO-END TEST PASSED. These are NOT real experimental results.')
