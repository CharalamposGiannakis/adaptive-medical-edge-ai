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

## F002c — Exploratory TinyViT transfer (2026-10-04)

User-authorized one-evening learning experiment, separate from frozen F002a/F002b.
Question: can one pretrained transformer roughly match the static MobileNetV2
reference while offering useful later adaptive dimensions? Choose only TinyViT-5M
(`tiny_vit_5m_224.dist_in22k_ft_in1k`), mean K16 frame embeddings and a binary head.
Use a small-data AdamW recipe with separate encoder/head learning rates, warm-up,
cosine decay and frozen BatchNorm statistics. T006 cohort/labels/splits and
duplicate-aware evaluation units remain unchanged; threshold 0.5 is primary.
Exact recipe, rationale and commands: `experiments/F002c/README.md`.

Measured engineering checks: 36 existing F002 tests and 12 F002c tests passed;
pretrained K16 forward/backward/Adam smoke passed on the local RTX 4060 Laptop GPU
with 3,122 MiB peak allocated memory. The 1,232 train/265 validation inventories
and 254 validation evaluation units verified. No test-image inference occurred;
only frozen test-manifest metadata was verified by the shared helper.

Full DAS-5 run completed on 2026-10-04 as Slurm job `110822` on
`node205` (NVIDIA GeForce GTX 1080 Ti). The local batch copy removed only the
GTX1080Ti scheduler constraint while retaining the generic one-GPU request; Slurm
assigned a 1080 Ti anyway. All 48 F002/F002c tests and the same-device CUDA
preflight passed. Training stopped after epoch 13; the selected epoch-8 checkpoint
has SHA-256
`30b70734426e84f429b83011abca04511540149d1fc3e416274c6fbcb14853b2`.
The run took 1,480.97 s (24 min 41 s) and recorded 3,142 MiB peak allocated /
3,556 MiB peak reserved GPU memory.

Measured duplicate-aware validation for the selected checkpoint:

- K4: macro-F1 0.8602, accuracy 0.8819, normal recall 0.9101, abnormal recall 0.8158.
- K8: macro-F1 0.8676, accuracy 0.8898, normal recall 0.9270, abnormal recall 0.8026.
- K16: macro-F1 0.8737, accuracy 0.8937, normal recall 0.9213, abnormal recall 0.8289.
- Frozen F002b MobileNetV2 K16 remains macro-F1 0.859, accuracy 0.886,
  normal recall 0.944 and abnormal recall 0.750.

Interpretation: this one exploratory TinyViT run is predictively competitive with
the frozen MobileNetV2 reference and gives strong same-checkpoint K4/K8/K16
behavior. It does **not** overturn D025 or establish a formal architecture winner:
the recipes differ, only one TinyViT seed was run, validation selected the
checkpoint, and TinyViT F002-style FLOPs/latency/throughput have not yet been
measured. F002b therefore remains the frozen static reference; F002c justifies a
bounded cost/adaptivity characterization before any architecture decision.
Raw evidence remains ignored under `artifacts/F002c/`; a complete external
artifact archive has SHA-256
`e098f0a671d98cafb70817b494afb09794730771ed7c75e4a4b73478a65ca03c`.

**Subsequent decision (2026-10-05):** After reviewing this evidence, Charalampos
and Salo selected the F002c epoch-8 TinyViT checkpoint as the adaptive-system
backbone for T009 onward (D026). This later project decision does not alter the
original exploratory interpretation, reopen F002 or replace the frozen F002b
MobileNetV2 static reference. The canonical shared frozen copy is
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](../checkpoints/f002c_tinyvit5m_epoch8.pt).
The ignored `artifacts/F002c/tinyvit5m_seed2026/checkpoint.pt` remains raw
experiment history, and the external archive remains archival evidence outside
Git.

## F003 — Adaptive dimensions of the D026 TinyViT (T009)

Research question: which ways of changing the compute of the frozen D026 TinyViT checkpoint give useful accuracy–cost trade-offs for T010? Ran on DAS-5 on 2026-10-06, validation only, no test-partition inference. Axes: frame count K ∈ {1, 2, 4, 8, 12, 16} (nested frame sets), linear early exits fitted on train-split features of the frozen trunk, and input resolution 160/192/224 px, plus a K4 → K16 confidence cascade. Strict fvcore FLOPs with the F002 handlers and reviewed TinyViT bookkeeping; F002 §9 timing on node201 (GTX 1080 Ti, shared node, with a same-job re-timing of MobileNetV2 F002b). Commands, environment and full tables are in `experiments/F003/README.md`; outputs in ignored `artifacts/F003/`.

Measured, duplicate-aware validation (254 units):

- Frame count at 224 px, macro-F1 / GFLOPs / JPEG-pipeline latency: K1 0.816 / 1.29 / 16.7 ms, K2 0.830 / 2.57 / 22.6 ms, K4 0.860 / 5.15 / 34.2 ms, K8 0.868 / 10.30 / 58.8 ms, K12 0.870 / 15.45 / 86.8 ms, K16 0.874 / 20.60 / 115.1 ms. Model-only latency is about 11 ms for K1–K4, 12.2 ms at K8 and 22.9 ms at K16.
- Paired differences to K16: K1 −0.057 and K2 −0.044 (CIs exclude 0); K4, K8 and K12 are not distinguishable from K16.
- Same-units comparison with MobileNetV2 F002b (5.11 GFLOPs, 0.859; re-timed 11.33 ms model-only, 81.8 ms pipeline): TinyViT K4 +0.001 (95% CI −0.057 to +0.064) at 5.15 GFLOPs and 34.2 ms pipeline.
- Early exits score below the frame-count curve except at K1, where the mid-stage-2 exit matches full K1 at 72% of its FLOPs. 160 px collapses abnormal recall to 0.49; 192 px keeps macro-F1 but lowers abnormal recall to 0.75.
- K4 → K16 cascade, τ tuned on validation: 11% of units escalated, about 6.9 GFLOPs, macro-F1 0.874 (equal to K16).

Interpretation: frame count is the primary adaptive axis, with K1/K4/K8/K16 as candidate shared-weight operating points and the K4 → K16 cascade as the input-adaptive candidate for T012. Exits and resolution are rejected without retraining. At matched compute TinyViT ties the static baseline, so the adaptive benefit must come from the cheaper points and the cascade. Single checkpoint, single seed, and a cascade threshold chosen on the scored data, so these are development results. Proposed as D027.
