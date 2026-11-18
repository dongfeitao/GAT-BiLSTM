"""Validation-only selection, five fixed training seeds, then a sealed test phase.

Examples (from the project root)::
    python -m bearing.experiments train --cache cache/laboratory --output runs/lab_module --dataset laboratory --suite module
    python -m bearing.experiments test --cache cache/laboratory --output runs/lab_module

All reported accuracy/F1 values are fractions. Epoch selection uses validation
Macro-F1 (fixed labels 0..4); exact ties retain the earlier epoch. No seed is
selected or discarded. The test command runs only after every selected model
has all five checkpoints and the validation summary has been frozen.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import random
import time

import joblib
import numpy as np
import sklearn
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from .metrics import scores, seed_summary, write_csv, save_confusion, save_confusion_counts
from .models import make_model

SEEDS = [6, 42, 102, 1010, 2025]
SUITES = {
    "baseline": ["svm", "transformer", "cnn_bilstm", "cnn_bigru", "resnet", "cnn1d", "gat", "mhgat", "full"],
    "module": ["full", "gnn_bilstm", "bilstm", "lstm"],
    "feature": ["full", "time_only", "frequency_only"],
    "full": ["full"],
}


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Unsupported nondeterministic operations raise instead of silently changing protocol.
    torch.use_deterministic_algorithms(True)


def choose_device(value):
    device = "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(device)


def cache_signature(cache):
    """Only metadata files are inspected here; test signal arrays are never read."""
    cache = Path(cache)
    signature = {"manifest.json": digest(cache / "manifest.json")}
    for name in ("normalization.json", "preprocessing.json"):
        if (cache / name).exists():
            signature[name] = digest(cache / name)
    return signature


def load_split(cache, split):
    cache = Path(cache).resolve()
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["records"]
    rows = []
    for entry in entries:
        if entry["split"] != split:
            continue
        path = (cache / entry["path"]).resolve()
        if not path.is_relative_to(cache):
            raise ValueError("A cache entry points outside its cache directory.")
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: np.array(z[key], copy=True) for key in ("time", "freq", "y")}
        x, f, y = arrays["time"], arrays["freq"], arrays["y"]
        if x.ndim != 3 or x.shape[1:] != (1, 1024) or f.shape != (len(x), 1, 512):
            raise ValueError(f"Unexpected feature shapes in {path}: {x.shape}, {f.shape}")
        if y.shape != (len(x),) or len(x) == 0 or not np.isin(y, np.arange(5)).all():
            raise ValueError(f"Invalid targets in {path}")
        if not np.isfinite(x).all() or not np.isfinite(f).all():
            raise ValueError(f"Nonfinite input in {path}")
        arrays.update(record_id=entry["record_id"], bearing_id=entry.get("bearing_id"),
                      cache_path=str(path), sha256=digest(path))
        rows.append(arrays)
    if not rows:
        raise ValueError(f"No records in {split} split of {cache}")
    found = set(np.concatenate([r["y"] for r in rows]).tolist())
    if found != set(range(5)):
        raise ValueError(f"All five classes must be covered in {split}; found {sorted(found)}")
    return rows


def record_tensors(row, device):
    return (torch.as_tensor(row["time"], dtype=torch.float32, device=device),
            torch.as_tensor(row["freq"], dtype=torch.float32, device=device),
            torch.as_tensor(row["y"], dtype=torch.long, device=device))


def predict_neural(model, rows, device):
    model.eval()
    truths, predictions, ids = [], [], []
    with torch.inference_mode():
        for row in rows:
            x, f, y = record_tensors(row, device)
            logits = model(x, f)
            if logits.shape != (len(y), 5) or not torch.isfinite(logits).all():
                raise ValueError("Model produced invalid logits.")
            predictions.append(logits.argmax(-1).cpu().numpy())
            truths.append(row["y"])
            ids.extend([str(row["record_id"])] * len(y))
    return np.concatenate(truths), np.concatenate(predictions), np.asarray(ids)


def svm_arrays(rows):
    # The same preprocessed numerical signal inputs, without labels or fitted test transforms.
    x = np.concatenate([np.concatenate((r["time"].reshape(len(r["y"]), -1),
                                        r["freq"].reshape(len(r["y"]), -1)), axis=1) for r in rows])
    return x, np.concatenate([r["y"] for r in rows])


def predict_svm(model, rows):
    x, y = svm_arrays(rows)
    ids = np.concatenate([np.repeat(str(r["record_id"]), len(r["y"])) for r in rows])
    return y, model.predict(x), ids


def save_predictions(path, truth, pred, record_ids):
    np.savez_compressed(path, y_true=truth.astype(np.int64), y_pred=pred.astype(np.int64),
                        record_id=record_ids)


def train_one(name, cfg, seed, train, val, run_dir, device, epochs, patience):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    started = time.perf_counter()
    if name == "svm":
        model = make_pipeline(StandardScaler(), SVC(C=float(cfg.get("C", 100)),
                              gamma=cfg.get("gamma", .01), kernel="rbf", random_state=seed,
                              cache_size=1024, probability=False))
        x, y = svm_arrays(train)
        model.fit(x, y)
        truth, pred, ids = predict_svm(model, val)
        checkpoint = run_dir / "model.joblib"
        joblib.dump(model, checkpoint)
        result = scores(truth, pred)
        result.update(seed=seed, best_epoch=None, training_seconds=time.perf_counter() - started)
        result["seed_note"] = "RBF SVC is deterministic with probability=False; seed variation may be zero."
        write_csv(run_dir / "epochs.csv", [{"epoch": 1, "validation_accuracy": result["accuracy"],
                                            "validation_macro_f1": result["macro_f1"]}])
    else:
        model = make_model(name, cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", .001)),
                                     weight_decay=float(cfg.get("weight_decay", .0001)))
        criterion = nn.CrossEntropyLoss()
        best_score, best_epoch, best_state, stale = -1., 0, None, 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, count = 0., 0
            # Each optimizer update sees a COMPLETE record/segment; no mixed-record graph.
            for index in np.random.permutation(len(train)):
                x, f, y = record_tensors(train[int(index)], device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, f)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss for {name}, seed {seed}, epoch {epoch}")
                loss.backward()
                if cfg.get("gradient_clip") is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
                optimizer.step()
                loss_sum += float(loss.detach()) * len(y)
                count += len(y)
            truth, pred, ids = predict_neural(model, val, device)
            current = scores(truth, pred)
            history.append({"epoch": epoch, "train_loss": loss_sum/count,
                            "validation_accuracy": current["accuracy"],
                            "validation_macro_f1": current["macro_f1"]})
            write_csv(run_dir / "epochs.csv", history)
            if current["macro_f1"] > best_score:
                best_score, best_epoch, stale = current["macro_f1"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if patience > 0 and stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("No checkpoint was selected.")
        model.load_state_dict(best_state)
        truth, pred, ids = predict_neural(model, val, device)
        result = scores(truth, pred)
        result.update(seed=seed, best_epoch=best_epoch, epochs_trained=len(history),
                      training_seconds=time.perf_counter() - started)
        checkpoint = run_dir / "model.pt"
        torch.save({"model_name": name, "config": cfg, "seed": seed,
                    "best_epoch": best_epoch, "state_dict": best_state,
                    "selection_metric": "validation_macro_f1"}, checkpoint)
    save_predictions(run_dir / "validation_predictions.npz", truth, pred, ids)
    dump_json(run_dir / "validation_metrics.json", result)
    return result, checkpoint


def default_candidates(name, budget):
    if budget < 1:
        raise ValueError("Search budget must be positive.")
    if name == "svm":
        all_configs = [{"C": c, "gamma": g} for c, g in itertools.product([1, 10, 100], [.001, .01, .1])]
    else:
        # A common 3 x 3 learning-rate/weight-decay grid, with fixed model-appropriate architecture.
        # Replace with the actual paper's prespecified candidates via --config when available.
        all_configs = [{"lr": lr, "weight_decay": wd} for lr, wd in itertools.product([.0001, .0005, .001], [0., .00001, .0001])]
    if budget > len(all_configs):
        raise ValueError(f"Default space contains {len(all_configs)} unique configurations; supply a larger explicit space.")
    indexes = np.random.default_rng(42).permutation(len(all_configs))[:budget]
    return [all_configs[int(i)] for i in indexes]


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def train_phase(args):
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    manifest_dataset = json.loads((cache / 'manifest.json').read_text(encoding='utf-8')).get('dataset')
    if manifest_dataset != args.dataset:
        raise ValueError(f'Dataset argument {args.dataset} does not match cache {manifest_dataset}')
    output.mkdir(parents=True, exist_ok=True)
    if (output / "FREEZE.json").exists() or (output / "TRAIN_STARTED.json").exists():
        raise FileExistsError("Use a NEW output directory; existing training/frozen results must not be overwritten.")
    names = args.models.split(",") if args.models else SUITES[args.suite]
    if len(names) != len(set(names)) or any(n not in set(sum(SUITES.values(), [])) for n in names):
        raise ValueError("Unknown or duplicate model names.")
    if args.epochs < 1 or args.patience < 0:
        raise ValueError("epochs must be positive and patience nonnegative.")
    spec = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    signature = cache_signature(cache)
    device = choose_device(args.device)
    train, val = load_split(cache, "train"), load_split(cache, "val")
    dump_json(output / "TRAIN_STARTED.json", {"dataset": args.dataset, "suite": args.suite,
              "models": names, "seeds": SEEDS, "cache_signature": signature,
              "configuration": spec, "environment": environment(), "device": str(device)})
    per_seed, summaries, frozen_models = [], [], {}
    for name in names:
        model_dir = output / name
        model_spec = spec.get("models", {}).get(name, {})
        base = {"lr": .001, "weight_decay": .0001, "tau": .7,
                "hidden_dim": 128, "heads": 4, "encoder_chunk_size": 32}
        base.update(spec.get("base", {}))
        base.update(model_spec.get("base", {}))
        if args.no_tune:
            candidates = [{}]
        else:
            candidates = model_spec.get("candidates", default_candidates(name, args.budget))
            if len(candidates) != args.budget:
                raise ValueError(f"{name}: supply exactly {args.budget} candidates for the common budget.")
        if len({json.dumps(c, sort_keys=True) for c in candidates}) != len(candidates):
            raise ValueError("Candidate configurations must be distinct.")
        selected_cfg, best_score = None, -1.
        tuning_rows = []
        # Even a no-tune reference run records its validation score; it is NOT a completed search.
        for i, candidate in enumerate(candidates):
            cfg = {**base, **candidate}
            print(f'[{name}] validation candidate {i+1}/{len(candidates)}, seed 42', flush=True)
            result, _ = train_one(name, cfg, 42, train, val, model_dir / "tuning" / f"candidate_{i:02d}",
                                  device, args.epochs, args.patience)
            tuning_rows.append({"candidate": i, "seed": 42, "accuracy": result["accuracy"],
                                "macro_f1": result["macro_f1"], "config": cfg})
            if result["macro_f1"] > best_score:
                selected_cfg, best_score = cfg, result["macro_f1"]
        dump_json(model_dir / "tuning.json", {"mode": "fixed_reference_no_search" if args.no_tune else "validation_search",
                  "candidates": tuning_rows, "selected_config": selected_cfg,
                  "tie_break": "earlier candidate; earlier epoch within candidate",
                  "test_used": False})
        runs, checkpoints = [], []
        for seed in SEEDS:
            print(f'[{name}] formal training seed {seed}', flush=True)
            result, checkpoint = train_one(name, selected_cfg, seed, train, val,
                                           model_dir / f"seed_{seed}", device, args.epochs, args.patience)
            row = {"model": name, "seed": seed, "accuracy": result["accuracy"],
                   "macro_f1": result["macro_f1"], "n_windows": result["n_windows"]}
            runs.append(row)
            per_seed.append(row)
            checkpoints.append({"seed": seed, "path": str(checkpoint.relative_to(output)),
                                "sha256": digest(checkpoint), "best_epoch": result["best_epoch"]})
        summary = {"model": name, **seed_summary(runs)}
        summaries.append(summary)
        dump_json(model_dir / "validation_summary.json", summary)
        frozen_models[name] = {"config": selected_cfg, "checkpoints": checkpoints}
    write_csv(output / "validation_per_seed.csv", per_seed)
    write_csv(output / "validation_summary.csv", summaries)
    payload = {"version": 1, "dataset": args.dataset, "suite": args.suite, "seeds": SEEDS,
               "models": frozen_models, "cache_signature": signature,
               "train_val_files": [{"path": str(Path(r["cache_path"]).relative_to(cache)),
                                    "sha256": r["sha256"]} for r in train + val],
               "epochs": args.epochs, "patience": args.patience,
               "selection": "validation Macro-F1, earliest epoch/candidate ties; no seed selection",
               "test_payloads_read": False,
               "metric_aggregation": "window-level scores per seed, then mean and sample SD (ddof=1)"}
    payload["configuration_sha256"] = hashlib.sha256(json.dumps(payload["models"], sort_keys=True).encode()).hexdigest()
    dump_json(output / "FREEZE.json", payload)
    # The seal also detects accidental editing of the phase boundary file.
    (output / "FREEZE.sha256").write_text(digest(output / "FREEZE.json") + "\n", encoding="ascii")
    print(f"Training/validation frozen: {output / 'FREEZE.json'}")
    print("No test predictions have been generated. Run the test subcommand next.")


def test_phase(args):
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    freeze_path = output / "FREEZE.json"
    if not freeze_path.exists():
        raise FileNotFoundError("Complete the training/validation phase before testing.")
    expected = (output / "FREEZE.sha256").read_text(encoding="ascii").strip()
    if digest(freeze_path) != expected:
        raise ValueError("Frozen configuration was changed after selection.")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["seeds"] != SEEDS or cache_signature(cache) != frozen["cache_signature"]:
        raise ValueError("Seeds or data metadata differ from the frozen protocol.")
    for entry in frozen["train_val_files"]:
        if digest(cache / entry["path"]) != entry["sha256"]:
            raise ValueError("Training/validation data changed after selection.")
    for info in frozen["models"].values():
        if [r["seed"] for r in info["checkpoints"]] != SEEDS:
            raise ValueError("Each model must have all five seed checkpoints.")
        for run in info["checkpoints"]:
            if digest(output / run["path"]) != run["sha256"]:
                raise ValueError("A selected checkpoint was changed after selection.")
    device = choose_device(args.device)
    # Exclusive creation makes an accidental second invocation fail before any test inference.
    with (output / "TEST_STARTED.json").open("x", encoding="utf-8") as marker:
        json.dump({"status": "started", "freeze_sha256": expected}, marker)
    test = load_split(cache, "test")
    rows, summaries = [], []
    plot_required = ((frozen["dataset"] == "laboratory" and frozen["suite"] == "module") or
                     (frozen["dataset"] == "ottawa" and frozen["suite"] == "baseline"))
    for name, info in frozen["models"].items():
        model_rows = []
        pooled = np.zeros((5, 5), dtype=np.int64)
        for run in info["checkpoints"]:
            seed_everything(run["seed"])
            checkpoint = output / run["path"]
            if name == "svm":
                model = joblib.load(checkpoint)
                truth, pred, ids = predict_svm(model, test)
            else:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = make_model(name, info["config"]).to(device)
                model.load_state_dict(payload["state_dict"], strict=True)
                truth, pred, ids = predict_neural(model, test, device)
            run_dir = checkpoint.parent
            save_predictions(run_dir / "test_predictions.npz", truth, pred, ids)
            metric = {"model": name, "seed": run["seed"], **scores(truth, pred)}
            dump_json(run_dir / "test_metrics.json", metric)
            counts = save_confusion(truth, pred, run_dir / "test_confusion",
                                    f"{name}, seed {run['seed']}", plot=plot_required or args.plot_all)
            pooled += counts
            model_rows.append(metric)
            rows.append(metric)
        summary = {"model": name, **seed_summary(model_rows)}
        summaries.append(summary)
        dump_json(output / name / "test_summary.json", summary)
        save_confusion_counts(pooled, output / name / "pooled_test_confusion",
                              f"{name}: counts pooled across five seeds",
                              plot=plot_required or args.plot_all)
    write_csv(output / "test_per_seed.csv", rows)
    write_csv(output / "test_summary.csv", summaries)
    dump_json(output / "TEST_COMPLETE.json", {
        "status": "complete", "models": list(frozen["models"]), "seeds": SEEDS,
        "pooled_confusion_note": "Sum raw counts across five runs, then normalize columns. Not an ensemble; the same test windows appear once per seed.",
        "metric_note": "Mean and sample SD of five independently calculated test scores; no metric is computed from averaged confusion diagonals.",
        "test_files": [{"record_id": r["record_id"], "sha256": r["sha256"]} for r in test]})
    print(f"Final test results saved to {output / 'test_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Tune on validation only and train all five seeds.")
    train.add_argument("--cache", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--dataset", choices=["laboratory", "ottawa"], required=True)
    train.add_argument("--suite", choices=SUITES, required=True)
    train.add_argument("--models", help="Optional comma-separated subset, useful for smoke tests.")
    train.add_argument("--config", help="JSON with optional base and models.{name}.{base,candidates}.")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=0, help="0 trains all epochs; checkpoint still uses validation Macro-F1.")
    train.add_argument("--budget", type=int, default=9)
    train.add_argument("--no-tune", action="store_true", help="Use declared reference configuration; not a completed search.")
    train.add_argument("--device", default="auto")
    test = commands.add_parser("test", help="Evaluate all frozen checkpoints once, with no training.")
    test.add_argument("--cache", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--device", default="auto")
    test.add_argument("--plot-all", action="store_true")
    args = parser.parse_args()
    (train_phase if args.command == "train" else test_phase)(args)


if __name__ == "__main__":
    main()
