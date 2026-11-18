# Local verification

Previous functional checks used Windows, Python 3.10.15, and PyTorch 2.5.1+cpu.
All test data were synthetic and must not be reported as paper results.

Passed checks:

- Forward computation and cross-entropy backward computation for 13 neural variants.
- Isolated nodes and duplicate-hyperedge removal.
- CSV/tab input: skip five rows and retain four columns.
- Laboratory interval calculation: 595/190/195 windows from 1,024,000 samples.
- Fifteen synthetic MAT records and five records in each subset.
- Training-only normalization and rejection of unverified strict bearing grouping.
- Five-seed training, validation summary, freeze, final test, and repeated-test protection.
- One-epoch synthetic five-seed training/testing for all nine comparison methods.
- Required column-normalized confusion-matrix generation.
- Profiling CLI for all nine methods and stage timing for the full model.

This English package preserves the Python implementations. Documentation and
filenames are English; JSON data-path strings have been serialized with
Unicode escapes and checked to decode to exactly the original locations.
Package integrity and Python syntax were checked after repackaging.

Not performed: actual Linux data training, formal 50-epoch searches, full-size
1,953-window CUDA training, or CUDA timing. Run smoke_test.py and
test_preprocessing.py on the target machine before formal experiments.
