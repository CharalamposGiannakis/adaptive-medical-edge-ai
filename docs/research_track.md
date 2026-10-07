# Research track

## Current direction

Develop adaptive edge inference for prehospital/ambulance lung POCUS with
OpenPOCUS as a technical experimental proxy. D025 keeps F002b MobileNetV2 as the
frozen static reference; D026 selects the F002c TinyViT-5M epoch-8 checkpoint as
the adaptive-system backbone, with its canonical shared frozen copy at
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](../checkpoints/f002c_tinyvit5m_epoch8.pt).
F003 (T009) characterized the adaptive dimensions; F004 (T010) built the
shared-weight frame-count family that T011 and T012 build on.

## Evidence so far

- F001–F001c established pilot evidence for shared-depth operation and per-input
  complementarity on DermaMNIST, without using its official test split. Those
  studies remain historical feasibility work rather than the selected POCUS model.
- T006 fixed the OpenPOCUS cohort and known-dependency-group-disjoint split.
- F002b established the frozen MobileNetV2 static reference at K16: validation
  macro-F1 0.859, abnormal recall 0.750 and 5.11 GFLOPs.
- F002c's selected TinyViT checkpoint reached duplicate-aware validation macro-F1
  0.8602/0.8676/0.8737 at K4/K8/K16 without K-specific retraining. At K16 its
  abnormal recall was 0.8289. This supports the D026 adaptive-backbone choice but
  is single-seed development evidence under a different training recipe.
- F003 measured TinyViT's FLOPs, latency and throughput and established frame
  count as the adaptive axis (D027, accepted).
- F004 serves K1/K2/K4/K8/K12/K16 on the D026 weights as one family with anytime
  upgrades and per-request switching (D028, accepted). Fine-tuning across K
  did not pass its pre-specified rule.
- F005 compresses inference work: nested structured pruning with distillation
  keeps macro-F1 at 29–44% fewer FLOPs per frame in one 20.4 MB weight file
  (D029, accepted). Dynamic INT8 gives file-size reduction only on this hardware
  and was not selected.

## Next research decision

T012 next maps resource states to the measured width × K
points of the F005b file and T013 performs the final static-versus-adaptive
comparison, including a statically pruned single-width model and static TinyViT
K16 as further fixed baselines. The T006 held-out test
partition remains reserved until a final protocol is fixed.

## Interpretation guardrails

Separate validation measurements, validation-driven model/checkpoint selection,
per-input analyses, cost measurements and deployable policy performance. Do not
infer clinical significance, statistical superiority or edge-device behavior from
the present validation results.
