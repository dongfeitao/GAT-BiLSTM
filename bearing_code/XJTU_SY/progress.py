"""Read-only progress view: python progress.py."""
import json
from pathlib import Path
from data import config_load

cfg=config_load(Path(__file__).with_name('config.json'))
out=Path(cfg['output_dir'])
for name in ['status.json','benchmark.json']:
    p=out/name
    print('\n'+name)
    print(p.read_text(encoding='utf-8') if p.exists() else 'Not yet generated')
for seed in cfg['seeds']:
    for task in ['stage','rul']:
        p=out/f'seed_{seed}'/task/'complete.json'
        if p.exists():print(f'\nseed={seed} task={task}\n'+p.read_text(encoding='utf-8'))
