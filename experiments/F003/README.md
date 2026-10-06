# F003 — Adaptive dimensions of the D026 TinyViT (T009)

**Question:** which ways of changing the compute of the frozen D026 TinyViT
checkpoint give useful accuracy–cost trade-offs, and which should T010 build
operating points from?

**Status:** ran on DAS-5 on 2026-10-06 (issue #11). All numbers are on the
T006 validation partition; no model has seen the test partition. Validation
also tuned the cascade threshold here, so cascade numbers are optimistic.

## Short answer

- **Frame count K is the primary adaptive axis.** Cost is linear in K (1.29
  GFLOPs per frame), macro-F1 degrades smoothly from K16 to K4 and drops
  clearly only at K1–K2. End-to-end latency scales with K because JPEG
  decoding dominates: 16.7 ms at K1 against 115.1 ms at K16 on the 1080 Ti.
- **A K4 → K16 confidence cascade is the input-adaptive candidate.** Escalating
  only the uncertain K4 predictions matches K16's macro-F1 (0.874) at about a
  third of K16's FLOPs. The K16 pass reuses the four K4 frames because the frame
  sets are nested.
- **Rejected:** early exits on the frozen trunk (below the frame-count curve at
  every cost except the very cheapest K1 points) and lower input resolution (160 px collapses towards predicting
  normal; 192 px trades abnormal recall for normal recall).
- **Against the static baseline:** TinyViT K4 costs the same FLOPs as the frozen
  MobileNetV2 F002b K16 (5.15 vs 5.11 GFLOPs) and has the same macro-F1 (paired
  difference +0.001, 95% CI −0.057 to +0.064). The adaptive system's advantage
  has to come from the cheaper points and the cascade, not from beating the
  baseline at equal cost.

## Design

Everything uses the frozen D026 checkpoint
`checkpoints/f002c_tinyvit5m_epoch8.pt` (SHA-256 checked on load), F002c's
bicubic full-square preprocessing, threshold 0.5 and the duplicate-aware
evaluation units. Three axes, all sharing the same weights:

- **Frame count K ∈ {1, 2, 4, 8, 12, 16}.** K4/K8/K16 use the frozen F002
  sampler. K12 is the same widest-gap bisection stopped at 12 frames, K2 keeps
  the two interior thirds anchors and K1 the first of them. That keeps every
  set nested: K1 ⊂ K2 ⊂ K4 ⊂ K8 ⊂ K12 ⊂ K16 (`adaptive.frame_indices`, tested
  against the frozen sampler for every length 16–399).
- **Exit depth.** Linear heads after stage 1 (128-d), after block 3 of stage 2
  (160-d) and after stage 2 (160-d), on spatially pooled features averaged over
  K. Each head is a standardized L2 logistic regression (C = 1, no class
  weights, duplicate-aware sample weights) fitted on the train split's K16
  features of the frozen trunk, then folded into one `nn.Linear`. The trunk is
  not retrained. `final` is the checkpoint's own head.
- **Input resolution 160 / 192 / 224 px.** The same weights; TinyViT pads its
  attention windows, so no retraining or interpolation is needed.

**One cached pass.** The model mean-pools per-frame features before a linear
head, and every K set sits inside the K16 frames. One K16 pass that stores the
pooled per-frame features at every exit therefore gives every (exit, K)
prediction exactly. The cache reproduces the direct model's validation logits to
9.5e-7.

**Cascade (descriptive).** Run a cheap configuration and escalate a unit to
K16 when |p − 0.5| < τ, for τ from 0 to 0.5. Expected FLOPs = cheap cost +
escalation rate × (K16 cost − reused work). The reused work is the cheap
frames' encoder passes (K cascade) or the trunk up to the exit (exit cascade).
The threshold is chosen on the same validation data it is scored on.

**Cost.** Strict fvcore counting with the F002 handlers (one multiply-add = one
FLOP). TinyViT adds operators the F002 allowlist never saw. `view`,
`split_with_sizes`, padding/contiguous copies and `index` (the attention-bias
gather) are data movement. `rsub`, `remainder` and `floor_divide` appear only as
0-d window-padding arithmetic; their handler fails if they ever touch a real
tensor. Runtime follows the F002 §9 protocol: the first 200 validation rows,
50 warm-up requests before each of 5 repeats, model-only (preloaded tensors)
and JPEG pipeline paths.

## Commands

From the repository root on DAS-5 (`myenv`, CUDA 12.6 module):

```bash
sbatch experiments/F003/das5_f003.sbatch flops
sbatch experiments/F003/das5_f003.sbatch features --partition train
sbatch experiments/F003/das5_f003.sbatch features --partition validation
sbatch experiments/F003/das5_f003.sbatch features --partition validation --resolution 160
sbatch experiments/F003/das5_f003.sbatch features --partition validation --resolution 192
sbatch experiments/F003/das5_f003.sbatch analyze
sbatch -w node201 experiments/F003/das5_f003.sbatch benchmark-reference
sbatch -w node201 experiments/F003/das5_f003.sbatch benchmark
python experiments/F003/run_f003.py report      # needs matplotlib for the plot
```

Tests: `python -m unittest tests.test_f003`. `features` refuses to overwrite a
cached feature file, and `benchmark` skips configurations that already have a
result, so a rerun cannot mix two feature passes or two timing sessions.

## Environment

- Feature passes ran in parallel on whatever GPU SLURM assigned (no model
  constraint, following the teaching-team scheduling advice Charalampos
  recorded for F002c): train 224 px on node219 (Titan RTX, 62 s), validation
  224 px on node202 (GTX TITAN X, 22 s), validation 160/192 px on node220
  (Titan RTX, 13–14 s). FP32 with TF32 off, so the GPU model changes features
  only at float rounding level.
- All timings ran in one job (110931) on node201, GTX 1080 Ti, driver 580.95.05,
  the same host as the F002b timings. Another user held the other three GPUs on
  node201 for 7-day jobs, so the node was shared, not exclusive. The job
  therefore also re-timed the frozen MobileNetV2 F002b checkpoint
  (`benchmark-reference`, read-only) for a same-conditions reference: 11.33 ms
  model-only (11.40 ms in the exclusive F002b run) and 81.8 ms pipeline
  (89.2 ms then).
- Python 3.12.14, torch 2.14.0+cu126, timm 1.0.30, fvcore 0.1.5.post20221221.
  Every artifact records host, GPU, versions and source hashes.
- `run_f003.py` grew during the run: the `benchmark-reference` command was
  added before the timing job, and the report switched to unfused-for-all and
  gained the cascade curve afterwards. So `flops.json`, the feature files and
  `analysis.json` record an earlier `run_f003.py` hash than the committed file.
  The functions they ran (`cmd_flops`, `cmd_features`, `cmd_analyze` and their
  helpers) did not change; `adaptive.py` is identical throughout.

## Measured results

Duplicate-aware validation, 254 units (178 normal, 76 abnormal). CI = unit
bootstrap, 1000 resamples. GFLOPs per sequence request. Latency is the median of
5 repeat medians.

**Frame count (224 px, final head).**

| K | Macro-F1 [95% CI] | Acc. | Normal rec. | Abnormal rec. | GFLOPs | Model-only ms | Pipeline ms | Pipeline seq/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.816 [0.764, 0.865] | 0.846 | 0.893 | 0.737 | 1.29 | 10.72 | 16.7 | 59.8 |
| 2 | 0.830 [0.778, 0.878] | 0.854 | 0.882 | 0.789 | 2.57 | 10.99 | 22.6 | 44.3 |
| 4 | 0.860 [0.813, 0.907] | 0.882 | 0.910 | 0.816 | 5.15 | 10.99 | 34.2 | 29.2 |
| 8 | 0.868 [0.823, 0.912] | 0.890 | 0.927 | 0.803 | 10.30 | 12.20 | 58.8 | 17.0 |
| 12 | 0.870 [0.823, 0.915] | 0.890 | 0.916 | 0.829 | 15.45 | 17.47 | 86.8 | 11.5 |
| 16 | 0.874 [0.828, 0.918] | 0.894 | 0.921 | 0.829 | 20.60 | 22.93 | 115.1 | 8.7 |
| MobileNetV2 F002b K16 | 0.859 | 0.886 | 0.944 | 0.750 | 5.11 | 11.33 | 81.8 | 12.2 |

The K16 row reproduces F002c's recorded K16 result. Paired difference to K16
(unit bootstrap): K1 −0.057 [−0.107, −0.012], K2 −0.044 [−0.083, −0.008],
K4 −0.014 [−0.045, +0.015], K8 −0.006 [−0.030, +0.014], K12 −0.004
[−0.023, +0.011]. Against MobileNetV2 F002b on the same units: K4 +0.001
[−0.057, +0.064], K8 +0.009 [−0.049, +0.077], K16 +0.015 [−0.040, +0.077].

Model-only latency uses canonical unfused attention for every configuration.
Fused attention is 0.2–1.8 ms faster, but its logits differ from unfused by
0.5e-5 to 2.2e-5, which straddles the 1e-5 limit borrowed from D024. Pipeline
latency used fused attention where it passed that limit (K8, K12, K16, both
lower resolutions and the mid-stage-2 exit) and unfused elsewhere, which moves
pipeline numbers by about 1 ms.

**Early exits (224 px).** At K16: stage 1 0.761 (10.85 GFLOPs, 13.83 ms), mid
stage 2 0.817 (14.78, 17.72 ms), stage 2 0.847 (18.35, 21.31 ms). At K4: 0.736,
0.801 and 0.837 (2.71, 3.69 and 4.59 GFLOPs). Except at K1, every exit scores
below a full-model configuration of similar or lower cost. At K1 the mid-stage-2
exit matches the full model (0.815 vs 0.816) at 72% of its FLOPs (0.92 vs 1.29),
and the stage-1 exit is the cheapest point measured (0.777 at 0.68 GFLOPs). Pipeline latency barely moves (105–114 ms
at K16) because decoding dominates.

**Resolution (final head, K16).** 160 px: 0.778, normal/abnormal recall
1.000/0.487, 12.54 GFLOPs, 14.27 ms. 192 px: 0.876, recall 0.966/0.750,
16.71 GFLOPs, 18.89 ms. Smaller K shows the same pattern; for example K8 at
192 px scores 0.868 with abnormal recall 0.750, against 0.803 at 224 px.

**Cascades.** K4 → K16: τ = 0.25 escalates 6% of units, 6.00 GFLOPs, macro-F1
0.873; τ = 0.35 escalates 11%, 6.85 GFLOPs, 0.874. K2 → K16 at τ = 0.40
escalates 15%, 5.34 GFLOPs, 0.864. K8 → K16 needs 20% escalation (12.3 GFLOPs)
to reach 0.874. Exit cascades from stage 1 reach 0.886 at τ = 0.40, but only
after escalating 45% of units (15.3 GFLOPs). Full curves are in
`analysis.json`.

`pareto_f003.png` plots every static configuration and the K4 → K16 cascade.

## Interpretation

- **K is the axis to build on.** It is the only one that keeps quality
  close to full while cutting cost by large factors, and it needs no new
  weights. Cost and pipeline time fall together because frames are both
  the arithmetic and the decoding work. The useful range is K4–K16; K1 and K2
  are real low-budget points but lose 0.04–0.06 macro-F1.
- **On this GPU, model-only latency is flat from K1 to K4** (about 11 ms) and
  only grows from K8. A 1080 Ti is overhead-bound for a small TinyViT batch,
  just as it was for MobileNetV2. Runtime savings from small K therefore show up
  end to end (decode) and in throughput, which is what D020 asked us to report,
  more than in model-only latency.
- **The cascade is the most interesting result, but not yet a proven one.**
  It reaches K16 quality at a third of the cost. The K4-versus-K16 gap it
  closes is itself only 0.014 and not statistically resolved, and τ was picked
  on the scored data. It deserves a proper policy evaluation (T012) with τ fixed
  before the test-partition evaluation, which the brief allows: thresholds are
  tuned on validation, the Pareto frontier is reported on test.
- **Early exits failed because the trunk was never trained for them.** The
  frozen intermediate features carry little class signal, and stage 1 already
  costs 53% of a frame because of its 28 × 28 attention. Jointly trained exits
  might do better, but that changes the weights and K already gives the
  trade-off for free.
- **Resolution shifts the decision threshold.** At lower resolution the
  checkpoint predicts normal more often (normal recall 1.000 at 160 px). Macro-F1 at 192 px
  hides a drop in abnormal recall, the clinically costly error. Rejected
  without retraining.
- **Matched-compute parity with MobileNetV2** means the adaptive story is "same
  quality as the static baseline at its cost, plus cheaper points and an
  input-adaptive path", not "better at every cost". TinyViT does keep higher
  abnormal recall at K4 (0.816 vs 0.750) and the static baseline higher normal
  recall (0.944 vs 0.910).
- Single checkpoint, single seed, 76 abnormal units: differences under about
  0.03 macro-F1 are within noise.

## Passed to T010

Candidate operating points, all from one checkpoint with nested frames:

| Point | GFLOPs | Macro-F1 | Pipeline ms | Role |
|---|---:|---:|---:|---|
| K1 | 1.29 | 0.816 | 16.7 | lowest budget |
| K2 | 2.57 | 0.830 | 22.6 | low budget (optional) |
| K4 | 5.15 | 0.860 | 34.2 | matches static baseline cost |
| K8 | 10.30 | 0.868 | 58.8 | middle |
| K16 | 20.60 | 0.874 | 115.1 | full |
| K4 → K16 cascade | ~6–7 | ~0.873 | derived | input-adaptive candidate for T012 |

K12 adds little over K8 and K16. Exits and resolution are rejected for now;
the only exit result worth remembering is the mid-stage-2 exit at K1, if T010
ever needs a point below 1.29 GFLOPs. If
T010 wants to improve the cheap end, the most direct option is training the
shared model across several K values (it changes the weights, so it needs a
documented team decision).

## Artifacts

Ignored, under `artifacts/F003/` (local copy and DAS-5
`/var/scratch/bde2621/amea/artifacts/F003/`): `flops.json` (all 36
configurations with by-operator counts), `features/*.pt` (cached per-frame
features, keys, nested positions, direct K16 logits), `exit_heads.pt`,
`analysis.json` (metrics, CIs, paired comparisons, cascades),
`predictions/*.csv` (+ units JSON), `benchmark/*.json` (raw per-request
latencies, both attention paths, equivalence, F002b reference), `summary.json`,
`pareto_f003.png` and `logs/`.
