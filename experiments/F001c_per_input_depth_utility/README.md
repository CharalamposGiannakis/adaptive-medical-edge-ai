# F001c — Per-input depth utility / rescue analysis

F001c characterizes which DermaMNIST-224 validation examples change correctness or prediction as inference proceeds through depths 4–12 of the shared DeiT-Tiny backbone.

This is analysis-only. It loads the completed F001 backbone and depth-4/depth-8 heads plus the frozen-backbone F001b heads at depths 5, 6, 7, 9, 10, and 11. It does not train or modify weights, does not instantiate train or test datasets, and uses the official validation split only. Depth 12 uses the classifier stored in the full F001 backbone checkpoint. F001/F001b source, checkpoints, and results are SHA-256 snapshotted before and after execution; a mismatch stops the run.

Run from the repository root using the existing project environment:

```powershell
.\.venv\Scripts\python.exe experiments/F001c_per_input_depth_utility/run_analysis.py
```

Optional arguments: `--device auto|cpu|cuda`, `--no-amp`, `--batch-size`, `--num-workers`, `--data-root`, and `--artifact-root`. The default artifact directory is `artifacts/F001c_per_input_depth_utility/`, ignored by Git. The runner uses F001 preprocessing and class mapping and ordered validation indices for stable sample IDs.

Outputs include all sample/depth probabilities and confidence/entropy, consecutive and selected pairwise transitions, uncertainty-quantile rescue rates, fixed-depth/oracle comparisons, unrestricted/restricted oracle upper bounds, later-depth first-correct counts, and six plots. All plots are generated from the saved CSV analyses. Oracle policies select the shallowest measured correct prediction with hindsight; these are unreachable upper bounds, not deployable controllers. If no allowed depth is correct, the reported all-sample compute charges the maximum allowed depth, and coverage separately records the failure.

Compute values are the existing measured **fvcore FLOPs** from F001b, not new estimates. See `summary.md` and `run_metadata.json` in the generated artifact directory for results, runtime configuration, checkpoint paths, and input hashes. Confidence/entropy comparisons are descriptive; they do not validate a routing signal. No clinical interpretation is made.
