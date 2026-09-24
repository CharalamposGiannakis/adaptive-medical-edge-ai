# F001b — Full Depth Characterization

F001b evaluates every truncation depth from 4 through 12 on the shared DeiT-Tiny backbone trained in F001. It reuses the F001 heads at depths 4 and 8 and the full-depth classifier at depth 12; only heads at depths 5, 6, 7, 9, 10, and 11 are trained. The backbone stays frozen while these heads are trained.

The runner imports F001's dataset, training, metric, latency, and fvcore FLOP utilities to preserve its preprocessing and measurement conventions. It checks that the F001 combined checkpoint contains exactly the same backbone and depth-4/8 head tensors as the phase checkpoints, and it verifies the reused validation metrics against F001's saved results. Only train and validation splits are loaded. The official test split remains untouched.

Run from the repository root after installing `experiments/feasibility_001/requirements.txt`:

```powershell
python experiments/F001b_depth_sweep/run_depth_sweep.py
```

F001b takes the seed, head-training settings, device, and latency settings from `artifacts/feasibility_001/run_config.json`. Outputs are written under the ignored `artifacts/F001b_depth_sweep/` directory, with new head checkpoints below its `checkpoints/` subdirectory. F001 checkpoints and outputs are read-only inputs.

The output includes one CSV and JSON row per depth, three plots, per-transition deltas and marginal gains, and a generated `summary.md`. FLOPs are explicitly reported as fvcore FLOPs. Unsupported operators and uncalled modules are recorded without custom handlers. Plot lines only connect measured points as visual guides; no curve is fitted or interpolated.

Latency follows F001's batch-1 timing procedure on a fixed all-zero `1x3x224x224` tensor. It measures model inference only and excludes data loading and preprocessing. Use `--reuse-trained-heads` to reload any already trained F001b head checkpoints and repeat evaluation/benchmarking without retraining them.

Latency remains specific to the device and software used for the run. The summary describes measured patterns but does not choose an optimal depth.
