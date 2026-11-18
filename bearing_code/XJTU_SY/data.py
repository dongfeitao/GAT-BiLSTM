"""XJTU-SY: split by bearing, chronological CSVs, train-only normalization."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def config_load(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for key in ('cache_dir', 'output_dir', 'data_root'):
        p = Path(cfg[key])
        cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    ids = sum(cfg['split'].values(), [])
    if len(ids) != len(set(ids)):
        raise ValueError('A bearing cannot appear in multiple splits.')
    if cfg['history'] < 2 or cfg['baseline_records'] < 2:
        raise ValueError('history and baseline_records must be >=2')
    if cfg['record_points'] % cfg['window_points'] or cfg['window_points'] % 2:
        raise ValueError('record_points must divide into even-length windows')
    if not -1 <= cfg['tau'] <= 1 or len(cfg['stage_z_thresholds']) != 2:
        raise ValueError('Invalid thresholds')
    if cfg['stage_z_thresholds'][0] >= cfg['stage_z_thresholds'][1]:
        raise ValueError('Stage thresholds must increase.')
    return cfg

def signature(cfg):
    keys = ['data_root','sample_rate_hz','record_interval_minutes','record_points',
            'window_points','channel','baseline_records','rms_trailing_median',
            'stage_z_thresholds','split']
    return hashlib.sha256(json.dumps({k:cfg[k] for k in keys}, sort_keys=True).encode()).hexdigest()

def discover(cfg):
    folders = {}
    wanted = set(sum(cfg['split'].values(), []))
    for p in Path(cfg['data_root']).rglob('Bearing*'):
        if p.is_dir() and p.name in wanted:
            if p.name in folders:
                raise ValueError(f'Duplicate bearing directory: {p.name}')
            folders[p.name] = p
    if wanted != set(folders):
        raise ValueError(f'Missing bearings: {sorted(wanted - set(folders))}')
    return folders

def preprocess_record(x, window):
    """DC removal -> Hann -> one-sided amplitude spectrum, excluding Nyquist."""
    x = np.asarray(x, np.float64)
    rms = float(np.sqrt(np.mean((x-x.mean())**2)))
    w = x.reshape(-1, window)
    w = (w - w.mean(axis=1, keepdims=True)) * np.hanning(window)[None, :]
    f = 2.0 * np.abs(np.fft.rfft(w, axis=1)[:, :window//2]) / window
    return w.astype('float32'), f.astype('float32'), rms

def proxy_labels(rms, cfg):
    """Uses fixed initial calibration and trailing-only smoothing; no future maxima."""
    k = cfg['baseline_records']
    if len(rms) <= k:
        raise ValueError('Insufficient records after baseline calibration')
    mu, sd = float(np.mean(rms[:k])), max(float(np.std(rms[:k], ddof=1)), 1e-8)
    smooth = pd.Series(rms).rolling(cfg['rms_trailing_median'], min_periods=1).median().to_numpy()
    z = (smooth-mu)/sd
    stages = np.digitize(z, cfg['stage_z_thresholds']).astype('int64')
    return smooth, z, stages, mu, sd

def prepare(cfg):
    cache = Path(cfg['cache_dir']); cache.mkdir(parents=True, exist_ok=True)
    folders = discover(cfg)
    manifest = {'signature': signature(cfg), 'bearings': {}, 'preprocessing':
                'DC removal -> Hann; FFT magnitude 2/L, no Nyquist; horizontal channel by default'}
    start = time.perf_counter()
    for split, ids in cfg['split'].items():
        for bid in ids:
            fs = list(folders[bid].glob('*.csv'))
            if any(not p.stem.isdigit() for p in fs):
                raise ValueError(f'Non-numeric CSV filename in {bid}')
            fs.sort(key=lambda p:int(p.stem))
            numbers = [int(p.stem) for p in fs]
            if not numbers or numbers != list(range(1, len(fs)+1)):
                raise ValueError(f'{bid}: expected consecutive CSV numbers 1..N; missing files cannot be ignored')
            n = len(fs); out = cache/bid; out.mkdir(exist_ok=True)
            meta_path = out/'metadata.json'
            fingerprints = [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in fs]
            srcsig = hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if meta.get('signature') == signature(cfg) and meta.get('source_signature') == srcsig:
                    if all((out/f).exists() for f in ['time.npy','freq.npy','observations.csv']):
                        manifest['bearings'][bid] = meta
                        print(f'CACHE {bid}: {n} records', flush=True); continue
                raise ValueError(f'{bid}: stale cache; use a new cache_dir to preserve reproducibility')
            shape = (n, cfg['record_points']//cfg['window_points'], cfg['window_points'])
            xt = np.lib.format.open_memmap(out/'time.npy',mode='w+',dtype='float32',shape=shape)
            xf = np.lib.format.open_memmap(out/'freq.npy',mode='w+',dtype='float32',shape=shape[:-1]+(shape[-1]//2,))
            rms, hashes = [], []
            for i,p in enumerate(fs):
                # Official CSV header is retained; reject malformed/truncated data.
                frame = pd.read_csv(p)
                if frame.shape != (cfg['record_points'],2):
                    raise ValueError(f'{p}: expected {cfg["record_points"]} x 2, got {frame.shape}')
                vals = frame.to_numpy(dtype=np.float64)
                if not np.isfinite(vals).all():
                    raise ValueError(f'Nonfinite observation: {p}')
                a,b,r = preprocess_record(vals[:,cfg['channel']],cfg['window_points'])
                xt[i],xf[i] = a,b; rms.append(r)
                hashes.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
                if (i+1)%500==0: print(f'PREP {bid}: {i+1}/{n}',flush=True)
            xt.flush();xf.flush(); del xt,xf
            rms=np.asarray(rms);sm,z,stage,mu,sd=proxy_labels(rms,cfg)
            elapsed=(np.asarray(numbers)-1)*cfg['record_interval_minutes']
            pd.DataFrame({'record':numbers,'elapsed_minutes':elapsed,'rms':rms,
                          'rms_trailing_median':sm,'rms_z':z,'proxy_stage':stage,
                          'rul_minutes':elapsed[-1]-elapsed}).to_csv(out/'observations.csv',index=False)
            meta={'bearing':bid,'condition':int(bid[7]),'split':split,'records':n,
                  'endpoint_minutes':float(elapsed[-1]),'source_folder':str(folders[bid]),
                  'signature':signature(cfg),'source_signature':srcsig,'baseline_mean':mu,'baseline_sd':sd}
            write_json(out/'source_hashes.json',hashes);write_json(meta_path,meta)
            manifest['bearings'][bid]=meta
            print(f'PREPARED {bid}: {n} records; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    # Fit only on train arrays, global scalar mean/std per domain (no per-window rescaling).
    norm={}
    for domain in ['time','freq']:
        count=0; total=0.; sq=0.
        for bid in cfg['split']['train']:
            a=np.load(cache/bid/(domain+'.npy'),mmap_mode='r')
            for first in range(0,len(a),64):
                v=np.asarray(a[first:first+64],np.float64)
                count+=v.size;total+=float(v.sum());sq+=float(np.square(v).sum())
        mean=total/count;sd=max(float(np.sqrt(max(sq/count-mean*mean,0))),1e-8)
        norm[domain]={'mean':mean,'std':sd,'count':count}
    norm['rul_scale_minutes']=max(manifest['bearings'][bid]['endpoint_minutes'] for bid in cfg['split']['train'])
    manifest['normalization']=norm
    manifest['elapsed_seconds']=time.perf_counter()-start
    write_json(cache/'manifest.json',manifest)
    print('PREPROCESSING COMPLETE',flush=True)
    return manifest

class BearingData:
    def __init__(self,cfg,bid):
        p=Path(cfg['cache_dir'])/bid
        self.bid=bid;self.time=np.load(p/'time.npy',mmap_mode='r');self.freq=np.load(p/'freq.npy',mmap_mode='r')
        self.obs=pd.read_csv(p/'observations.csv')
        self.first=max(cfg['baseline_records'],cfg['history']-1)
        self.n=len(self.obs)
        if self.first>=self.n:raise ValueError(f'{bid}: no prediction samples')
    def blocks(self,size):
        return [(self.bid,s,min(s+size,self.n)) for s in range(self.first,self.n,size)]

def load_manifest(cfg):
    m=json.loads((Path(cfg['cache_dir'])/'manifest.json').read_text(encoding='utf-8'))
    if m['signature']!=signature(cfg):raise ValueError('Cache configuration mismatch')
    return m
