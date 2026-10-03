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

## F002a — OpenPOCUS architecture qualification (T008)

Research question: which of MobileNetV2, ResNet-18 and DeiT-Tiny, under the D024 protocol, should become the single F002b static backbone? Ran on DAS-5 (GTX 1080 Ti, node201/node205) on 2026-10-03 with seed 42, train gradients only on T006 train and selection only on validation. No test-partition inference. Commands, environment deviations (Python 3.12.14 instead of the pinned 3.14.3; all other packages exact) and full tables are in `experiments/F002/README.md`. Outputs are in ignored `artifacts/F002/`.

Measured, duplicate-aware validation (254 units):

- References: all-normal macro-F1 0.412; train-fitted device + zone metadata model 0.482.
- K16, all valid on the first attempt with no rescue: MobileNetV2 0.867 (best epoch 5 of 8), ResNet-18 0.782 (epoch 1 of 4), DeiT-Tiny 0.710 (epoch 2 of 5).
- Strict fvcore K16 sequence FLOPs: 5.11 G, 29.17 G and 20.29 G. Model-only median latency on one 1080 Ti: 11.35, 7.53 and 12.74 ms (DeiT fused; fused vs unfused max logit difference 9.5e-7).
- The K16 gate retained MobileNetV2 and ResNet-18 and eliminated DeiT-Tiny.
- The same checkpoints at K4/8/16: MobileNetV2 0.813/0.858/0.867, ResNet-18 0.750/0.775/0.782. MobileNetV2 has the higher F1 and lower FLOPs at every K, and ResNet-18 the >10% faster model-only runtime at every K.
- The final gate gave no automatic backbone. The K16 gap is 0.086, so seed 43 was not triggered. At the human evidence gate, Charalampos and Salo selected MobileNetV2 (D025).

Interpretation: MobileNetV2 is ahead on predictive quality and arithmetic cost. ResNet-18's advantage is GPU latency only, and the JPEG-decode pipeline (21 to 87 ms per request) shrinks it to 3 to 5 ms. F1 rises from K4 to K16 for both, which gives T009 a first hint that variable K is usable. These are single-seed validation numbers that also drove selection, so they are development evidence, not generalization or clinical performance.

## F002b — Frozen MobileNetV2 static reference (T008)

Research question: does a fresh MobileNetV2, trained under the unchanged D024 recipe with seed 2026, pass the validity gate so it can serve as the frozen static baseline for T009 onwards? Ran on DAS-5 node201 (GTX 1080 Ti) on 2026-10-03, same code (source hashes match commit `ee4aaa4`), train gradients on T006 train, checkpoint selection on validation, no test-partition inference.

Measured, duplicate-aware validation (254 units):

- Valid on the first attempt (lr 1e-4, no rescue). Best epoch 8; early stopping after epoch 11; 38.2 min of training.
- Macro-F1 0.859, accuracy 0.886, normal recall 0.944, abnormal recall 0.750, loss 0.354. All-normal reference 0.412.
- K16 cost: 5.11 GFLOPs; model-only 11.40 ms median, 87.7 seq/s, 1403 frames/s; JPEG pipeline 89.17 ms, 11.2 seq/s.
- Frozen checkpoint `artifacts/F002/f002b/mobilenet_v2_seed2026/lr1e-4/checkpoint.pt`, SHA-256 `5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823`, read-only locally and on DAS-5.

Interpretation: seed 2026 lands close to F002a's seed-42 MobileNetV2 (0.859 vs 0.867; both find 57 of 76 abnormal units, 53 of them the same), which suggests the backbone choice was not a lucky seed. It is still one validation split that also chose the checkpoint, so it is development evidence, not a generalization estimate.
