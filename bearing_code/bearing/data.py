"""Split first, then create numerical time/frequency windows.

No fabricated observations are returned on I/O failure. Test statistics never fit
normalization. A cached NPZ is one intact recording or one laboratory time segment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLASS_NAMES = ["ball", "compound", "healthy", "inner", "outer"]
STATE_TO_LABEL = {"B": 0, "C": 1, "H": 2, "I": 3, "O": 4}
TRIAL_SPLIT = {1: "train", 2: "val", 3: "test"}
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ["path", "record_id", "state", "condition", "trial", "bearing_id", "split"]


def _json_write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(path: str | Path) -> Path:
    """Allow one whitespace-only filename discrepancy, never ambiguous matches."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    normalized = lambda s: re.sub(r"\s+", "", s).casefold()
    candidates = []
    if path.parent.is_dir():
        candidates = [p for p in path.parent.iterdir()
                      if p.is_file() and normalized(p.name) == normalized(path.name)]
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise FileNotFoundError(f"File absent or whitespace match ambiguous: {path}; matches={candidates}")


def data_read(file_path: str | Path, skip_rows: int = 5) -> pd.DataFrame:
    """Read exactly the first four CSV/tab columns after the first five rows.

    This uses the user's latest five-row instruction, not the older 16-row code.
    Malformed or nonnumeric values raise an error rather than returning zeros.
    """
    file_path = resolve_file(file_path)
    errors = []
    for encoding in ("utf-8-sig", "gb18030"):
        for sep in (",", "\t", ";"):
            try:
                frame = pd.read_csv(file_path, header=None, skiprows=skip_rows,
                                    sep=sep, encoding=encoding, usecols=[0, 1, 2, 3])
                frame = frame.iloc[:, :4].astype(np.float64)
                if frame.empty or not np.isfinite(frame.to_numpy()).all():
                    raise ValueError("Empty/nonfinite values in the first four columns")
                frame.columns = ["col1", "col2", "col3", "col4"]
                return frame
            except (ValueError, pd.errors.ParserError, UnicodeError, IndexError) as exc:
                errors.append(f"{encoding}/{repr(sep)}: {exc}")
    raise ValueError(f"Cannot read four numerical columns from {file_path}. " + " | ".join(errors))


def read_mat_channel(path: str | Path, channel: str = "Channel_1") -> np.ndarray:
    """Read ordinary MATLAB MAT files and MATLAB v7.3 HDF5 files."""
    from scipy.io import loadmat
    path = resolve_file(path)
    try:
        obj = loadmat(path, variable_names=[channel])
        if channel not in obj:
            raise KeyError(f"{channel} absent in {path}")
        values = np.asarray(obj[channel]).squeeze()
    except (NotImplementedError, ValueError, OSError) as exc:
        import h5py
        if not h5py.is_hdf5(path):
            raise exc
        with h5py.File(path, "r") as handle:
            if channel not in handle:
                raise KeyError(f"{channel} absent in {path}")
            values = np.asarray(handle[channel]).squeeze()
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite one-dimensional {channel} array: {path}, shape={values.shape}")
    return values.astype(np.float64, copy=False)


def parse_ottawa_id(path: str | Path) -> tuple[str, str, int]:
    """Accept HA1, H-A-1, H_A_1, and whitespace-separated equivalents."""
    stem = re.sub(r"[\s_-]+", "", Path(path).stem.upper())
    match = re.fullmatch(r"([BC HIO])([ABCD])([123])".replace(" ", ""), stem)
    if not match:
        raise ValueError(f"Cannot parse Ottawa state/profile/trial from {Path(path).name}")
    return match.group(1), match.group(2), int(match.group(3))


def scan_ottawa(cfg: dict, output_csv: str | Path) -> list[dict]:
    """Write observed file paths and the authorized deterministic trial split.

    Bearing identity is deliberately blank: a filename is not proof of specimen ID.
    """
    rows = []
    seen = set()
    for state, folder in cfg["directories"].items():
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            raise FileNotFoundError(f"Ottawa directory not found: {folder}")
        for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".mat"):
            state_found, condition, trial = parse_ottawa_id(path)
            if state_found != state:
                raise ValueError(f"Directory state {state} disagrees with {path.name}")
            record_id = f"{state}-{condition}-{trial}"
            if record_id in seen:
                raise ValueError(f"Duplicate Ottawa recording ID: {record_id}")
            seen.add(record_id)
            rows.append(dict(path=str(path.resolve()), record_id=record_id, state=state,
                             condition=condition, trial=trial, bearing_id="",
                             split=TRIAL_SPLIT[trial]))
    if not rows:
        raise ValueError("No Ottawa MAT files found")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _load_mapping(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty mapping CSV: {path}")
    required = set(MANIFEST_FIELDS)
    if not required.issubset(rows[0]):
        raise ValueError(f"Mapping requires columns {sorted(required)}")
    for row in rows:
        for key in row:
            row[key] = row[key].strip() if isinstance(row[key], str) else row[key]
        row["trial"] = int(row["trial"])
        row["state"] = row["state"].upper()
        row["condition"] = row["condition"].upper()
        row["split"] = row["split"].lower()
    return rows


def _ottawa_rows(cfg: dict, outdir: Path, mapping: str | None, grouping: str) -> tuple[list[dict], bool]:
    if mapping:
        rows = _load_mapping(mapping)
    else:
        rows = scan_ottawa(cfg, outdir / "ottawa_record_manifest.csv")
    selected = [r for r in rows if r["condition"] == cfg.get("condition", "A")]
    expected = {(state, trial) for state in STATE_TO_LABEL for trial in (1, 2, 3)}
    actual = [(r["state"], r["trial"]) for r in selected]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("The selected Ottawa condition must contain exactly three records for each of five states")
    all_ids, all_paths, physical = set(), set(), {}
    verified = all(bool(r["bearing_id"]) for r in selected)
    for row in selected:
        path = resolve_file(row["path"])
        state, condition, trial = parse_ottawa_id(path)
        if (state, condition, trial) != (row["state"], row["condition"], row["trial"]):
            raise ValueError(f"File metadata mismatch for {path}")
        if row["record_id"] != f"{state}-{condition}-{trial}":
            raise ValueError(f"record_id mismatch for {path}")
        if row["record_id"] in all_ids or path in all_paths:
            raise ValueError(f"Duplicate file or recording ID: {row}")
        if row["split"] not in SPLITS:
            raise ValueError(f"Invalid split: {row}")
        all_ids.add(row["record_id"])
        all_paths.add(path)
        row["path"] = str(path)
    for row in rows:
        bearing_id = row["bearing_id"]
        if bearing_id and row["split"] in SPLITS:
            previous = physical.setdefault(bearing_id, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Physical bearing {bearing_id} crosses subsets")
    if grouping == "bearing" and not verified:
        raise ValueError("Strict bearing grouping requires verified bearing_id values in the CSV manifest")
    if verified and len({r["bearing_id"] for r in selected}) != 15:
        raise ValueError("The declared 15-bearing protocol requires 15 distinct IDs in the selected condition")
    for split in SPLITS:
        chosen = [r for r in selected if r["split"] == split]
        if len(chosen) != 5 or {r["state"] for r in chosen} != set(STATE_TO_LABEL):
            raise ValueError(f"Each subset requires exactly one recording per class: {split}")
    return selected, verified


def temporal_intervals(n_samples: int, fs: int, gap_seconds: float = 1.0) -> dict[str, tuple[int, int]]:
    boundaries = [round(n_samples * 0.6), round(n_samples * 0.8)]
    half_gap = round(fs * gap_seconds / 2)
    intervals = {"train": (0, boundaries[0] - half_gap),
                 "val": (boundaries[0] + half_gap, boundaries[1] - half_gap),
                 "test": (boundaries[1] + half_gap, n_samples)}
    if any(start < 0 or stop <= start or stop > n_samples for start, stop in intervals.values()):
        raise ValueError("Exclusion gaps leave invalid temporal intervals")
    return intervals


def window_features(raw: np.ndarray, mean: float, std: float, length: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Normalize using TRAIN raw statistics, demean each window, symmetric Hann,
    then FFT. The time branch receives the same demeaned/tapered window.

    Spectrum follows paper indices k=0,...,L/2-1 and amplitude 2*abs(FFT)/L;
    Nyquist is excluded. No spectral fit, image conversion or augmentation occurs.
    """
    n_windows = len(raw) // length
    if n_windows == 0:
        raise ValueError("Record/segment shorter than one complete window")
    normalized = (np.asarray(raw[:n_windows * length], dtype=np.float64) - mean) / std
    windows = normalized.reshape(n_windows, length)
    windows -= windows.mean(axis=1, keepdims=True)
    windows *= np.hanning(length)[None, :]
    spectrum = 2.0 * np.abs(np.fft.rfft(windows, axis=1)[:, :length // 2]) / length
    return windows[:, None, :].astype(np.float32), spectrum[:, None, :].astype(np.float32)


def prepare(config_path: str | Path, outdir: str | Path, dataset: str = "laboratory",
            mapping: str | None = None, grouping: str | None = None) -> dict:
    config_path, outdir = Path(config_path), Path(outdir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = config[dataset]
    if (outdir / "manifest.json").exists():
        raise FileExistsError(f"Cache already prepared; choose a new output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "records").mkdir(exist_ok=True)
    sources = []
    if dataset == "laboratory":
        fs, expected = int(cfg["sampling_frequency"]), int(cfg["samples_per_record"])
        for state, filepath in cfg["files"].items():
            filepath = resolve_file(filepath)
            frame = data_read(filepath, int(cfg.get("skip_rows", 5)))
            raw = frame.iloc[:, int(cfg.get("channel_index", 0))].to_numpy(dtype=np.float64)
            if len(raw) < expected:
                raise ValueError(f"{filepath}: {len(raw)} samples; expected at least {expected}")
            raw = raw[:expected]
            record_id = f"{state}-1200-0%"
            digest = file_sha256(filepath)
            for split, (start, stop) in temporal_intervals(expected, fs, cfg.get("gap_seconds", 1.0)).items():
                sources.append(dict(raw=raw[start:stop], path=str(filepath), sha256=digest,
                                    raw_start=start, raw_stop=stop, record_id=record_id,
                                    segment_id=f"{record_id}_{split}", bearing_id=f"laboratory_{state}",
                                    state=state, label=STATE_TO_LABEL[state], split=split))
        grouping_unit, verified = "temporal_segment_within_record", False
        grouping_note = "Same five physical bearings and records occur in all subsets. Temporal gaps reduce boundary proximity; they do not establish specimen or record independence."
    elif dataset == "ottawa":
        fs, expected = int(cfg["sampling_frequency"]), int(cfg["samples_per_record"])
        grouping = grouping or cfg.get("grouping", "recording")
        if grouping not in ("recording", "bearing"):
            raise ValueError("grouping must be recording or bearing")
        mapping = mapping or cfg.get("manifest") or None
        rows, verified = _ottawa_rows(cfg, outdir, mapping, grouping)
        for row in rows:
            raw = read_mat_channel(row["path"], cfg.get("channel", "Channel_1"))
            if len(raw) != expected:
                raise ValueError(f"{row['path']}: {len(raw)} samples; expected exactly {expected}")
            sources.append(dict(raw=raw, path=row["path"], sha256=file_sha256(Path(row["path"])),
                                raw_start=0, raw_stop=expected, record_id=row["record_id"],
                                segment_id=row["record_id"], bearing_id=row["bearing_id"] or None,
                                state=row["state"], label=STATE_TO_LABEL[row["state"]], split=row["split"]))
        grouping_unit = "bearing" if verified else "recording"
        grouping_note = ("Bearing IDs supplied in the manifest; no bearing crosses subsets."
                         if verified else "Recording IDs are disjoint. Physical-bearing identity has not been verified; filenames are not specimen IDs. Trial 1=train, 2=val, 3=test unless the supplied mapping specifies otherwise.")
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    # Streaming combination of batch moments avoids concatenating every training recording.
    count, mean, m2 = 0, 0.0, 0.0
    for source in sources:
        if source["split"] != "train":
            continue
        x = source["raw"]
        n = len(x)
        new_mean = float(np.mean(x))
        new_m2 = float(np.sum((x - new_mean) ** 2))
        delta = new_mean - mean
        combined = count + n
        m2 += new_m2 + delta * delta * count * n / combined
        mean += delta * n / combined
        count = combined
    if not count:
        raise ValueError("No training samples")
    std = float(np.sqrt(m2 / count))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("Degenerate training-signal standard deviation")
    normalization = dict(mean=mean, std=std, fitted_split="train", raw_samples=count,
                         convention="population standard deviation; before windowing and per-window DC removal")
    records = []
    for source in sources:
        raw = source.pop("raw")
        time, freq = window_features(raw, mean, std, int(config.get("window_length", 1024)))
        y = np.full(len(time), source["label"], dtype=np.int64)
        filename = re.sub(r"[^a-zA-Z0-9_-]", "_", source["segment_id"]) + ".npz"
        dest = outdir / "records" / filename
        np.savez_compressed(dest, time=time, freq=freq, y=y)
        records.append({**source, "source_path": source["path"], "path": f"records/{filename}",
                        "n_windows": len(time), "discarded_tail_samples": len(raw) % time.shape[-1]})
    manifest = dict(dataset=dataset, records=records, class_names=CLASS_NAMES,
                    sampling_frequency=fs, window_length=int(config.get("window_length", 1024)),
                    stride=int(config.get("window_length", 1024)), grouping_unit=grouping_unit,
                    physical_bearing_verified=verified, grouping_note=grouping_note,
                    preprocessing_order=["split_raw", "train_only_global_standardization", "nonoverlapping_windows",
                                         "per_window_DC_removal", "symmetric_Hann", "FFT_first_L_over_2_bins"],
                    input_representation="numerical_arrays", config=config,
                    config_sha256=file_sha256(config_path))
    _json_write(outdir / "normalization.json", normalization)
    _json_write(outdir / "manifest.json", manifest)
    return manifest


def load_records(cache: str | Path, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        with np.load(cache / record["path"], allow_pickle=False) as arrays:
            result.append({**record, "time": arrays["time"], "freq": arrays["freq"], "y": arrays["y"]})
    if not result:
        raise ValueError(f"No {split} records in {cache}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--dataset", choices=["laboratory", "ottawa"], default="laboratory")
    parser.add_argument("--output", default="cache/laboratory")
    parser.add_argument("--mapping", help="CSV mapping with paths, recording IDs and optional verified bearing IDs")
    parser.add_argument("--grouping", choices=["recording", "bearing"])
    parser.add_argument("--scan-ottawa", metavar="CSV", help="Scan directories and write mapping without preparing data")
    args = parser.parse_args()
    if args.scan_ottawa:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))["ottawa"]
        rows = scan_ottawa(cfg, args.scan_ottawa)
        print(f"Wrote {len(rows)} records to {args.scan_ottawa}; bearing_id is intentionally blank.")
    else:
        manifest = prepare(args.config, args.output, args.dataset, args.mapping, args.grouping)
        print(json.dumps({"dataset": args.dataset, "grouping_unit": manifest["grouping_unit"],
                          "physical_bearing_verified": manifest["physical_bearing_verified"],
                          "windows": {s: sum(r["n_windows"] for r in manifest["records"] if r["split"] == s) for s in SPLITS}}, indent=2))


if __name__ == "__main__":
    main()

