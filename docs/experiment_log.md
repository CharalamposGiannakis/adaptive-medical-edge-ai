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
- Frozen checkpoint originally at `artifacts/F002/f002b/mobilenet_v2_seed2026/lr1e-4/checkpoint.pt`, now canonically tracked at `checkpoints/f002b_mobilenetv2_seed2026.pt`, SHA-256 `5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823`; the raw DAS-5 copy remains read-only.

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

Which ways of changing the compute of the frozen D026 TinyViT give useful accuracy–cost trade-offs? Ran on DAS-5 on 2026-10-06, validation only (254 units, 76 abnormal), no test-partition inference. Tested frame count K ∈ {1, 2, 4, 8, 12, 16}, frame selection (scan share, consecutive frames), frozen-trunk exits, 160/192 px input and K → K16 escalation, with strict fvcore FLOPs, F002 §9 timing on a GTX 1080 Ti, AUROC and paired tests. Design, commands, tables and figures: `experiments/F003/README.md`; outputs in ignored `artifacts/F003/`.

Measured: macro-F1 0.816 / 0.860 / 0.868 / 0.874 and AUROC 0.888 / 0.937 / 0.950 / 0.952 at K1 / K4 / K8 / K16, for 1.29 / 5.15 / 10.30 / 20.60 GFLOPs and 16.7 / 34.2 / 58.8 / 115.1 ms end to end. MobileNetV2 F002b: 0.859 / 0.913 at 5.11 GFLOPs. Exits and lower resolution have lower AUROC than the K curve at similar cost. Consecutive frames are worse than spread frames at K4 (p = 0.04). Cross-validated escalation scatters around plain K4. MobileNetV2 F002b at the same nested K: macro-F1 0.708 / 0.761 / 0.831 / 0.859 at K1 / K4 / K8 / K16 (0.32 / 1.28 / 2.55 / 5.11 GFLOPs), level with or below TinyViT at every matched cost.

Interpretation: frame count is the primary axis (accepted as D027); escalation and the short-scan idea are unresolved; at the static baseline's cost TinyViT ties it at threshold 0.5; per FLOP its frame-count curve is level with or above MobileNetV2's, not significantly at any single point. Single checkpoint and seed, so development evidence.

## F004 — Shared-weight frame-count operating points (T010)

Can the D027 frame-count axis run as one deployable model with shared weights and runtime switching, and does fine-tuning across K improve it? Ran on DAS-5 on 2026-10-07, validation only (254 units, 76 abnormal), no test-partition inference. F004a wraps the D026 checkpoint as `AdaptiveTinyViT` at K ∈ {1, 2, 4, 8, 12, 16} with an anytime clip cache, a per-layer weight check, strict fvcore FLOPs for points and upgrades, round-robin timing on three GTX 1080 Ti hosts and one CPU node, per-device results, GPU memory, an all-frames reference and an ONNX export. F004b fine-tuned D026 with a loss averaged over five K against a K16-only control (seeds 2026–2028), under an adoption rule fixed before running. A first pass with five points and section-by-section timing was redone after an internal review. Design, commands, tables and figures: `experiments/F004/README.md`; outputs in ignored `artifacts/F004/`.

Measured: macro-F1 0.816 / 0.829 / 0.860 / 0.868 / 0.870 / 0.874 and accuracy 0.846 / 0.854 / 0.882 / 0.890 / 0.890 / 0.894 at 1.29 / 2.57 / 5.15 / 10.30 / 15.45 / 20.60 GFLOPs (equal to F003); model-only 9.9 / 10.2 / 10.2 / 12.1 / 17.4 / 22.7 ms on GPU and 32 / 49 / 81 / 201 / 381 / 538 ms on CPU. All points read the same 215 parameter tensors of one 20.4 MB checkpoint. Anytime upgrades encode only added frames (K4 → K16: 15.45 GFLOPs) and match direct inference to 2.2e-5. Warm switching changes per-K latency by ≤ 0.13 ms; the first request at a new K costs up to about 150 ms. Only K1 and K2 are resolved below K16. Every frame: macro-F1 0.870 at 240 GFLOPs on average. ONNX export matches PyTorch to 4e-6. F002b: 11.3 ms GPU, 254 ms CPU model-only. F004b (seed 2026, the rule's seed): mean macro-F1 over five K 0.852 vs 0.849 (D026) and 0.851 (control), K16 0.842; 3-seed means 0.858 (multi-K) and 0.848 (control).

Interpretation: the D026 family meets T010's criteria. Fine-tuning across K is not adopted: the pre-specified test failed, and the 3-seed hint of a low-K gain is unresolved. Accepted as D028. Single checkpoint, validation reused for all selections, so development evidence.

## F005 — Compressing the shared frame-count family (T011)

F005a (2026-10-07, validation, CPU): INT8 dynamic quantization of the 41 Linear layers of the F004 family. Macro-F1 within −0.009/+0.008 of FP32 at every K (all paired intervals include zero), file 20.4 → 6.4 MB, FLOPs unchanged, CPU latency 2–6% slower (1.06–1.62× slower when frames are encoded one at a time, the mode that keeps anytime upgrades exact, measured against batched FP32). F005b: nested structured pruning of the hidden channels (Taylor importance, chosen over L1 by a recorded train-split rule) at widths 100/75/50/25% (1.29/1.10/0.91/0.72 GFLOPs per frame), fine-tuned from D026 with the sandwich rule and distillation, three seeds plus a 50%-only control. Mean macro-F1 over five K: 0.853/0.857/0.856/0.846 (D026 0.849; control 0.854); the recorded adoption rule holds (D029, accepted as compute compression). The shared nested checkpoint remains 20.4 MB. AUROC at or above the unpruned family at every cost. F005c: static INT8 in ONNX Runtime, matrix operators only, Percentile calibration: AUROC 0.01–0.02 lower and 1.5–1.8× slower than FP32 on the CPU, and worse with `reduce_range`; not kept. Follow-ups after review: width × K switching per request costs nothing measurable (GPU ≤ 0.6%); a width-aware anytime cache (`NestedAnytimeClip`) is tested; recalibrating all 19 downstream BNs per width after training lowered the narrow widths by 0.008 (not adopted); ONNX export of every width matches to 8e-6; nested file 20.4 MB vs 62.5 MB for four copies; calibration (ECE) unchanged by pruning; a random channel order trained the same way scores 0.03 (75/50%) to 0.10 (25% width) lower than Taylor order, three seeds each. Details: `experiments/F005/README.md`; outputs in ignored `artifacts/F005/`.

## F006a — Resource-aware controller design (T012a)

Question: Which operating configurations of the deployed F005b seed-2026
checkpoint should a deterministic FLOP-budget controller consider? Validation-
only analysis of all 24 width × K cells, using full-precision selected-epoch
macro-F1/accuracy from its canonical run and strict fvcore counts recomputed
from the canonical checkpoint. The original raw per-unit predictions and exact
AUROC are absent locally; AUROC is reported only at the three-decimal precision
in F005 Table 3b. No training, runtime simulation, or test-image inference.

Measured/reconstructed: nine cells are strict macro-F1/FLOPs Pareto points and
15 are dominated. With proposed `Cmax = full-width K16 = 20.596068563`
GFLOPs, the provisional max-F1-under-budget rule maps b = 0.25/0.50/0.75/1.00
to 0.50×K2 / 0.75×K8 / 0.75×K8 / 1.00×K16. The proposal and cheap-near-best
and cost-aware sensitivity grids are in
`experiments/F006a/README.md`, with reproducible tables and a figure. These
small validation differences do not establish pairwise superiority; the
normalization and selection rule await Charalampos and Salo's acceptance
before T012b.

## F006a_2 — Quality screen against frozen MobileNetV2 (T012a)

Question: Which of the **24** deployable F005b seed-2026 width × K cells meet
initial predictive-quality requirements before comparing compute? This
validation-only analysis uses the same selected-epoch F005b confusion counts
and strict checkpoint-derived FLOPs as F006a. It reconstructs the frozen F002b
MobileNetV2 reference from its documented 254 duplicate-aware validation units:
168/178 normal and 57/76 abnormal correct, giving accuracy 225/254 =
0.885827, macro-F1 0.858875 and abnormal recall 57/76 = 0.750. The three
inclusive screening conditions use these exact values, not the rounded README
figures. They are provisional empirical requirements, **not clinical safety
thresholds** or an accepted controller policy.

Measured/reconstructed: 12 cells meet accuracy, 10 macro-F1, 14 abnormal
recall, and 10 all three. The cheapest qualifying cell is 50% width × K2 at
1.816575299 GFLOPs: accuracy 227/254, macro-F1 0.869706, abnormal recall
59/76 and normal recall 168/178. Full width × K16 costs 20.596068563 GFLOPs
and detects 64/76 abnormal units, five more than that cheaper cell; efficiency
alone does not resolve the quality trade-off. Under the **proposed**, still
unapproved `B(b) = b × Cmax` normalization, the counts meeting both compute
and quality at b = 0.25/0.50/0.75/1.00 are 1/3/7/10; b = 0.00 defers by
definition. Tightening the accuracy requirement by three correct units makes
b = 0.25 infeasible; tightening abnormal recall by three detected units makes
b = 0.25 and 0.50 infeasible. No requirement was relaxed to force inference.

The selected-run confusion counts give exact threshold metrics but not paired
unit-level comparisons with F002b; AUROC is available only to three decimals
from F005's published table. F005's three-seed summaries are secondary
robustness context, not the deployable checkpoint's scores. All screening and
budget choices still require Charalampos and Salo's review before T012b.
Tables, script, reproduction and limitations:
[`experiments/F006a_2/README.md`](../experiments/F006a_2/README.md). No
training, runtime controller or T006 test-image inference occurred.

## F006b — Provisional resource-aware controller (T012b)

Question: Can the accepted F005b seed-2026 shared-weight checkpoint serve a
deterministic controller across continuous normalized FLOP budgets while the
three F006a_2 quality requirements are enforced? The user-directed
**provisional** rule filters all 24 cells by exact F002b-referenced validation
accuracy, macro-F1 and abnormal recall, then `B(b) = b × 20,596,068,563`
integer FLOPs. Among qualified and feasible cells it prioritizes abnormal
recall, then lower FLOPs, then a stable tie-breaker; otherwise it defers.
This extends the accepted D021 demonstration states without replacing them.
Implementing this testable prototype before team acceptance does not
retroactively change the T012a analysis records or accept their proposals.

Reconstructed from the tracked selected-run counts and checkpoint-derived
costs: four points are selected over the continuous range, at exact threshold
fractions `1,816,575,299/Cmax`, `10,298,034,451/Cmax`,
`15,447,051,507/Cmax` and `1`. They are `w0.5_k2`, `w1.0_k8`,
`w1.0_k12` and `w1.0_k16`, respectively, and are nondominated on FLOPs,
macro-F1 and abnormal recall jointly. F006a's earlier two-dimensional
macro-F1/FLOPs frontier therefore cannot filter the recall-priority choices.
At D021 `b = 0/0.25/0.50/0.75/1`, the choices are DEFER, `w0.5_k2`,
`w0.5_k2`, `w1.0_k8`, `w1.0_k16`; exact costs put full-width K8 and K12
just above nominal half- and three-quarter-budget boundaries.

In three deterministic five-request resource traces (decreasing, increasing,
fluctuating), the recall-priority policy makes 4/4/5 inferences and 1/1/0
deferrals, with 3/3/4 active model switches. Summed full configuration work
is 48.158/48.158/49.974 GFLOPs; modeled incremental work after same-clip
feature reuse is 22.413 GFLOPs in each case (rounded). All 12 scenario ×
policy comparison rows, including minimum-compute, F006a Rule A and fixed
full-width K16, have zero attempted budget violations. Fixed K16 can run
only at `b = 1` in each trace (four deferrals). These are policy/cost
simulations, not measured mixed-policy macro-F1.

Actual local serving on three distinct validation-group clips saved 60 trace
rows: 42 non-DEFER predictions, all matching direct logits exactly in the
saved trace. These three repeated clips are insufficient for a mixed-policy
quality estimate. One comparable local CPU job (Windows 11 Ryzen AI 9 HX 370,
four threads, 50 validation requests, five warm-ups and five repeats) found
chunk-4 median model-only latency of 31.66/168.62/249.80/333.01 ms for the
four selected points in cost order. Their direct/chunked logits matched
exactly; interleaved-minus-grouped switching deltas ranged from −0.85 to
+2.74 ms, with no consistent positive penalty. This is supporting local
latency evidence, not a FLOP-budget substitute or a DAS-5/edge-device result.

The reproducible runtime traces, comparison with deterministic alternatives,
CPU evidence and limitations are in
[`experiments/F006b/README.md`](../experiments/F006b/README.md).
This is a validation-stage policy implementation, not team acceptance of the
quality thresholds or controller and not held-out test evidence. No checkpoint
or baseline was retrained or changed.

## F007 — Proposed final comparison protocol (T013a; no measured result)

Question: Does the adaptive TinyViT family provide a useful quality,
coverage and cost trade-off against frozen static alternatives, and under
which resource states? [`experiments/F007/README.md`](../experiments/F007/README.md)
is a **proposed preregistration**, not an accepted decision or executed
experiment. It preselects six systems, D021 statewise and test-independent
mixed-budget comparisons, duplicate-aware unit metrics with explicit
deferrals, exact FLOPs, a single comparable CPU benchmark, known-group paired
intervals, fixed descriptive cost/quality contrasts and test-leakage guards.
PR #27 merged the historical T012 implementation. D030 subsequently proposes
10% simulated budget headroom for F007 only and removes formal predictive
noninferiority margins; Charalampos approved the direction, Salo's explicit
policy approval and independent F007 review remain pending before T013b.
No held-out test images, labels or predictions were accessed; no raw F007
results exist.
