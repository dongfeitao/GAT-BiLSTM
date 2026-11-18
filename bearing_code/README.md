# GAT-BiLSTM Bearing Diagnosis Experiments

Start with **START_HERE.txt**. The complete execution sequence, commands,
dataset partitions, five-seed protocol, result locations, and profiling
definitions are in **Execution_Guide.txt**.

## Project structure

- `01_preprocess.py`: preprocessing entry point.
- `02_gat_bilstm.py`: full-model training/testing entry point.
- `03_compare_models.py`: baseline-comparison entry point.
- `04_module_ablation.py`: module-ablation entry point.
- `05_feature_ablation.py`: feature-ablation entry point.
- `06_compute_cost.py`: profiling entry point.
- `run_experiments.py`: train all suites before optional final testing.
- `bearing/data.py`: complete CSV/MAT reading and preprocessing implementation.
- `bearing/models.py`: full GAT-BiLSTM and neural reference baselines.
- `bearing/experiments.py`: tuning, training, freezing, and final evaluation.
- `bearing/metrics.py`: metrics, summaries, and confusion-matrix plots.
- `bearing/profiling.py`: parameter counts, major-arithmetic FLOPs, and timing.
- `configs/data.json`: actual data locations and preprocessing settings.
- `configs/ottawa_split_plan.csv`: predefined logical recording partition.

The short numbered scripts import the complete implementations from bearing/.
Keep the directory structure intact. Python 3.10 or later is recommended.

## Data and protocol

Laboratory CSV loading skips FIVE rows, reads the first FOUR columns, and
uses the FIRST vibration channel. Ottawa uses Channel_1 and increasing-speed
condition A, with trial 1 for training, 2 for validation, and 3 for testing.
All models use the same dataset partition. Formal seeds are 6, 42, 102,
1010, and 2025. Test scores are calculated separately per seed and then
aggregated as mean and sample standard deviation. F1 means five-class Macro-F1.

JSON paths use Unicode escapes for non-ASCII directory names. JSON decoding
preserves the exact original paths; no data-directory renaming is required.
Source filenames and documentation are English. Runtime manifests necessarily
record actual source paths, including non-ASCII characters when present.

## Implementation choices

This is a reconstruction from the supplied manuscript, not recovered original
training code. It does not guarantee reproduction of published accuracy.
Full-model dimensions are 100+100 encoded features, 200-dimensional nodes,
128-dimensional hypergraph output, four 128-dimensional GAT heads projected
from 512 to 128, 128-dimensional cross-attention, and a bidirectional LSTM
with 128 units per direction followed by a 256-to-5 classifier.

Cross-attention takes queries from time features and keys/values from frequency
features. Variance fusion uses population variance across each record's branch
outputs. Threshold hyperedges are deduplicated; isolated nodes use only their
self transform. GAT uses direct cosine-threshold neighbors plus self loops.
Graphs are rebuilt from current features; threshold membership is not
differentiated, but feature aggregation remains differentiable.

An encoder chunk size of 32 does not truncate the full record graph or temporal
sequence. B=1 means one complete record/retained segment. Full-record GPU memory
requirements must be checked on the target machine.

Baselines are explicitly defined reference architectures, not exact copies of
the cited authors' implementations. MHGAT uses a multi-head threshold graph,
not a DTW-based literature implementation; ResNet is a compact 1D residual
network. Replace those definitions if an exact published baseline is required.

Default full-model trainable parameters total 488,101: the encoder has 800,
the two hypergraph linear maps have 51,200, the PyTorch BiLSTM has 264,192,
and the classifier has 1,285; the remainder belongs to attention. Do not
alter measured counts to fit older manuscript tables.

The default nine-candidate neural search uses learning rate and weight decay;
SVM searches C and gamma. The driver freezes the full model's selected
configuration for controlled ablations. Custom search spaces can be supplied
using --config. Do not describe default candidates as a different search
protocol reported in an earlier manuscript.

## Profiling definitions

Latency includes model-specific processing and final Softmax, but excludes
common preprocessing and device transfers. FLOPs count major matrix and
convolution arithmetic at two operations per multiply-add, not every scalar,
nonlinear, comparison, sorting, or memory operation. All executed operations
are included in measured latency. Fused recurrent paths are expanded for
FLOP counting only. SVM kernel FLOPs are left unavailable, not fabricated.
Parameter counts are derived from the actual model. SVM reports support-vector
information rather than a misleading neural trainable-parameter count.

See VERIFICATION.md for local checks. Actual Linux data and CUDA training have
not been verified in the local CPU test environment.
