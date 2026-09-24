# Experiment log

## F001 — Feasibility

Fine-tuned a pretrained DeiT-Tiny on DermaMNIST-224 and exposed shared-backbone operating points at depths 4, 8, and 12. Train/validation only; the official test split remains untouched. See `experiments/feasibility_001/` and ignored outputs in `artifacts/feasibility_001/`.

## F001b — Full depth characterization

Evaluated depths 4–12 using one F001 backbone, reusing the F001 depth-4/8 heads and depth-12 classifier and training only the missing frozen-backbone heads. Validation accuracy peaked at 85.14% at depth 9; macro-F1 peaked at depth 10 (0.7061). Compute is reported as fvcore FLOPs and was measured for every depth. No test split was loaded. See `experiments/F001b_depth_sweep/` and ignored outputs in `artifacts/F001b_depth_sweep/`.

## F001c — Per-input depth utility / rescue analysis

Analysis-only inference reused the existing F001 backbone and all depth-4–12 heads on the 1,003-example validation split. No training, threshold fitting, architecture changes, or test-set access occurred. Stable-index per-sample probabilities, confidence, and entropy were saved for every depth; all six figures derive from saved CSV analyses. SHA-256 snapshots confirm all F001/F001b source, checkpoint, and artifact inputs remained unchanged.

Key measurements:

- Consecutive transitions 4→9 all have positive net corrections (+45, +21, +22, +10, +9), while 9→10 is −2 and 10→11 is −8; 11→12 is +3.
- Comparing 9→12 directly: 28 rescues versus 35 regressions, net −7. There are 28 samples wrong at 9 but correct at 12; depth 12 is first correct across the complete 4–12 set for 5 samples.
- The unrestricted oracle covers 925/1,003 (92.2%) at mean 0.4781 measured fvcore GFLOPs under the documented max-depth fallback cost for oracle failures. It is an unreachable hindsight upper bound, not deployable controller performance. The restricted {6,7,8,9} and {6,7,8,9,12} oracles cover 88.7% and 90.9%, respectively.
- At shallow depths 6–9, the lowest-confidence quintile has shallow errors and next-depth rescues (conditional rates 26.7–32.4%); the highest-confidence quintile has no shallow errors, making conditional rescue undefined. High-entropy quintiles have 23.6–26.4% rescue rates, but lowest-entropy quintiles have no errors. This is descriptive evidence only, not validation of a deployable uncertainty signal.
- Aggregate validation accuracy remains highest at depth 9 (85.14%). Keep 6–9 and 12 available as candidates for scenario-specific analysis; do not declare an optimal depth from this feasibility-stage validation analysis.

Reproduce using the existing `.venv` with `python experiments/F001c_per_input_depth_utility/run_analysis.py`. Source and interpretation are in `experiments/F001c_per_input_depth_utility/`; generated tables/plots are ignored under `artifacts/F001c_per_input_depth_utility/`. The official test set remains untouched.
