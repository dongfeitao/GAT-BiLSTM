"""Independently recompute exported metrics and check frozen model provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from data import config_load, load_manifest, write_json


def verify(cfg):
    out = Path(cfg['output_dir'])
    manifest = load_manifest(cfg)
    metrics = pd.read_csv(out / 'test_metrics_by_seed.csv')
    frozen = json.loads((out / 'FREEZE.json').read_text(encoding='utf-8'))
    source_hashes = json.loads((out / 'source_sha256.json').read_text(encoding='utf-8'))
    for name, digest in source_hashes.items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    checked = []
    for seed in cfg['seeds']:
        for task in ['stage', 'rul']:
            p = out / f'seed_{seed}' / task / 'best.pt'
            assert hashlib.sha256(p.read_bytes()).hexdigest() == frozen[str(seed)][task]['checkpoint_sha256']
        for bid in cfg['split']['test']:
            f = pd.read_csv(out / f'seed_{seed}' / (bid + '_predictions.csv'))
            record_count = manifest['bearings'][bid]['records']
            first = max(cfg['baseline_records'], cfg['history'] - 1)
            assert len(f) == record_count - first
            np.testing.assert_array_equal(f.record, np.arange(first + 1, record_count + 1))
            np.testing.assert_allclose(f.elapsed_minutes, (f.record - 1) * cfg['record_interval_minutes'])
            np.testing.assert_allclose(f.rul_minutes, (record_count - f.record) * cfg['record_interval_minutes'])
            prob = f[[f'probability_stage_{i}' for i in range(3)]].to_numpy()
            assert np.isfinite(prob).all() and (prob >= 0).all()
            np.testing.assert_allclose(prob.sum(1), 1, atol=2e-6)
            np.testing.assert_array_equal(f.predicted_stage, prob.argmax(1))
            assert f.proxy_stage.isin([0, 1, 2]).all()
            assert np.isfinite(f.predicted_rul_minutes).all() and (f.predicted_rul_minutes >= 0).all()
            a = float((f.proxy_stage == f.predicted_stage).mean())
            class_f1 = []
            for k in range(3):
                truth = f.proxy_stage == k
                pred = f.predicted_stage == k
                denominator = int(truth.sum() + pred.sum())
                class_f1.append(2 * int((truth & pred).sum()) / denominator if denominator else 0.)
            macro = float(np.mean(class_f1))
            mae = float(np.abs(f.rul_minutes - f.predicted_rul_minutes).mean())
            row = metrics[(metrics.seed == seed) & (metrics.bearing == bid)]
            assert len(row) == 1
            row = row.iloc[0]
            np.testing.assert_allclose([a, macro, mae], [row.accuracy, row.macro_f1, row.rul_mae_minutes], atol=1e-8)
            checked.append({'seed': seed, 'bearing': bid, 'n': len(f), 'accuracy': a,
                            'macro_f1': macro, 'rul_mae_minutes': mae,
                            'predicted_rul_min': float(f.predicted_rul_minutes.min()),
                            'predicted_rul_max': float(f.predicted_rul_minutes.max())})
    for stem in ['01_experiment_settings', '02_test_summary', '03_stage_recognition', '04_rul_prediction']:
        for ext in ['png', 'pdf', 'svg']:
            assert (out / (stem + '.' + ext)).stat().st_size > 1000
    report = {'status': 'passed', 'independently_recomputed_metrics': checked,
              'training_source_hashes_match': True,
              'frozen_checkpoint_hashes_match': True,
              'labels_timestamps_and_prediction_counts_match': True,
              'all_four_outputs_present_in_png_pdf_svg': True}
    write_json(out / 'output_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    verify(config_load(parser.parse_args().config))
