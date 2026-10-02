# F002 — OpenPOCUS architecture qualification and static reference

**Status:** T007 methodology accepted after independent review and PR #22
merge; no F002 model has been trained, evaluated or selected. T008 implements
F002a/F002b under the accepted D024 contract. The authoritative scientific
contract and ordered T008 handoff are
[`docs/f002_methodology.md`](../../docs/f002_methodology.md) and
[`configs/f002_protocol.json`](../../configs/f002_protocol.json). D024 records
the accepted decision. This folder pins the dependency API versions; it does
not contain a run script or result yet.

Use the existing project environment on Charalampos's machine; do not create
an experiment-specific environment. The precise local
versions and CUDA build are in the protocol. On another machine, install
equivalent pinned package versions from `requirements.txt`, then rerun the
strict model/FLOPs tests and record the actual wheel/build/hardware. Never
download weights merely to run the T007 tests.

The primary runtime comparison uses a matched **DAS-5
CUDA GPU model/node class**, with optimized FP32 benchmark flags and a fixed
200-row validation request list. The local T007 tests build untrained models
only; they do not substitute local laptop timing for that comparison.

From the repository root, check the methodology-support code with the active
project Python (the current checkout's existing `.venv` was used for T007):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_f002_*.py' -v
```

T008 will add the training/benchmark commands and tracked run configuration
before execution. Generated checkpoints, predictions and timings belong under
ignored `artifacts/F002/`; preserve per-row and duplicate-aware outputs as
specified by the contract. The T006 test partition stays untouched by model
inference during T008.
