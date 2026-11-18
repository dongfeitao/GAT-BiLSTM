"""Train ALL suites before optionally entering final testing. Never retune on test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['laboratory','ottawa'],required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--budget',type=int,default=9)
    p.add_argument('--config')
    p.add_argument('--final-test',action='store_true',help='After all training finishes, test all frozen suites.')
    a=p.parse_args();root=Path(a.output)
    def run(args):subprocess.run([sys.executable,'-m','bearing.experiments',*args],check=True)
    common=['--cache',a.cache,'--device',a.device]
    command=['train',*common,'--output',str(root/'baseline'),'--dataset',a.dataset,'--suite','baseline',
             '--epochs',str(a.epochs),'--budget',str(a.budget)]
    if a.config:command.extend(['--config',a.config])
    run(command)
    # Controlled ablations inherit the full model's validation-selected settings.
    frozen=json.loads((root/'baseline'/'FREEZE.json').read_text())
    config=root/'ablation_locked_config.json'
    config.write_text(json.dumps({'base':frozen['models']['full']['config']},indent=2),encoding='utf-8')
    for suite in ['module','feature']:
        run(['train',*common,'--output',str(root/suite),'--dataset',a.dataset,'--suite',suite,
             '--epochs',str(a.epochs),'--config',str(config),'--no-tune'])
    if a.final_test:
        for suite in ['baseline','module','feature']:
            run(['test',*common,'--output',str(root/suite)])
    else:
        print('All suites frozen. Execute their test commands when ready; do not select by test outcomes.')


if __name__=='__main__':main()
