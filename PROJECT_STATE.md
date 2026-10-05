# Current phase

The architecture and data foundation is complete. The project is entering adaptive
experimentation, with T009 next after the F002c phase-transition branch merges.
F002b MobileNetV2 remains the frozen static comparison; D026 selects the F002c
TinyViT-5M checkpoint as the model to develop adaptively. No model has run on the
T006 held-out test partition.

## Frozen foundation

The scenario is resource-aware inference for prehospital/ambulance lung POCUS,
using OpenPOCUS as a technical experimental proxy rather than evidence of trauma,
ambulance-population or clinical-deployment effectiveness. The D019 task predicts
normal versus definite abnormal (B-lines and/or consolidation) for one exam × lung
zone sequence. Accuracy and macro-F1 are the primary classification metrics.

T006 fixes the cohort, labels, known-family/shared-recording-group-disjoint split
and evaluation units (D023). The retained snapshot has 1,762 sequences: 1,232
train, 265 validation and 265 test. The split protects against known dependencies,
but the released identifiers cannot prove biological-patient independence. T009
through T012 use train and validation only; the test partition remains reserved for
a later pre-specified final evaluation.

The frozen static reference is F002b MobileNetV2 (D025), seed 2026 at K16:

- duplicate-aware validation macro-F1 0.859 and accuracy 0.886;
- normal recall 0.944 and abnormal recall 0.750;
- 5.11 GFLOPs under the F002 convention;
- checkpoint SHA-256
  `5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823`.

Its checkpoint, protocol, results and historical decisions remain immutable. See
[`experiments/F002/README.md`](experiments/F002/README.md) for the full record.

## Adaptive backbone

Charalampos and Salo selected timm
`tiny_vit_5m_224.dist_in22k_ft_in1k` as the adaptive-system backbone on
2026-10-05 (D026). The canonical shared frozen copy is
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](checkpoints/f002c_tinyvit5m_epoch8.pt)
and has SHA-256
`30b70734426e84f429b83011abca04511540149d1fc3e416274c6fbcb14853b2`.

The same checkpoint, without K-specific retraining, produced the following
duplicate-aware validation results at threshold 0.5:

| K | Macro-F1 | Accuracy | Normal recall | Abnormal recall |
|---:|---:|---:|---:|---:|
| 4 | 0.8602 | 0.8819 | 0.9101 | 0.8158 |
| 8 | 0.8676 | 0.8898 | 0.9270 | 0.8026 |
| 16 | 0.8737 | 0.8937 | 0.9213 | 0.8289 |

At K16, TinyViT exceeded the frozen MobileNetV2 reference on validation macro-F1,
accuracy and abnormal recall, while MobileNetV2 retained higher normal recall.
K4 and K8 also retained useful predictive quality with fewer input frames. This is
single-seed development evidence under different training recipes, not statistical
or universal superiority. TinyViT cost, latency and throughput still need the
consistent measurement work in T009. Full evidence and caveats are in
[`experiments/F002c/README.md`](experiments/F002c/README.md).

## Current objective — T009

T009 empirically characterizes a useful adaptive configuration space around the
selected TinyViT checkpoint. Frame count is already a promising dimension, so the
first work is to measure predictive and cost behavior across useful K values under
one consistent FLOPs/runtime/throughput convention. T009 may also run bounded
experiments on TinyViT-supported depth or exits, spatial resolution, compression
interactions, and input-difficulty or confidence signals when evidence justifies
them. It does not need to implement every possible dimension.

The output is an evidence-backed choice of the configurations and adaptive axes to
carry into T010. Exploratory work should stay reproducible, but it does not require
a new decision-log entry or full preregistration for every small idea.

## Guardrails

- Keep the T006 cohort, labels, train/validation/test split and evaluation units
  frozen.
- Keep the F002b MobileNetV2 checkpoint as the immutable static reference.
- Use the F002c epoch-8 TinyViT checkpoint as the adaptive backbone unless a later
  documented team decision changes direction.
- Do not use the T006 test partition during T009–T012 development.
- Measure final candidate operating points with a consistent FLOPs, runtime and
  throughput convention.
- Give each experiment an ID, reproducible configuration/commands, raw evidence,
  and a clear separation between measurements and interpretation.

## Next steps

T009 characterizes adaptive dimensions → T010 selects at least three useful
shared-model operating points → T011 integrates one compression method → T012
implements the resource-aware policy → T013 performs the final static-versus-
adaptive evaluation.
