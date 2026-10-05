# Research track

## Current direction

Develop adaptive edge inference for prehospital/ambulance lung POCUS with
OpenPOCUS as a technical experimental proxy. D025 keeps F002b MobileNetV2 as the
frozen static reference; D026 selects the F002c TinyViT-5M epoch-8 checkpoint as
the adaptive-system backbone, with its canonical shared frozen copy at
[`checkpoints/f002c_tinyvit5m_epoch8.pt`](../checkpoints/f002c_tinyvit5m_epoch8.pt).
T009 now characterizes useful adaptive dimensions and their costs before T010
selects operating points.

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
- TinyViT FLOPs, latency and throughput have not yet been measured under the
  consistent convention needed for final operating-point comparisons.

## Next research decision

T009 should determine which bounded TinyViT configurations form a useful adaptive
configuration space. Frame count is the first promising dimension; additional K
values, architecture-supported depth/exits, spatial resolution, compression
interactions or input-difficulty signals may be investigated when they can change
the operating-point decision. Evidence should reject unnecessary complexity.

T010 will select at least three measured shared-model operating points. T011 adds
one compression method, T012 maps resource states to measured points, and T013
performs the final static-versus-adaptive comparison. The T006 held-out test
partition remains reserved until a final protocol is fixed.

## Interpretation guardrails

Separate validation measurements, validation-driven model/checkpoint selection,
per-input analyses, cost measurements and deployable policy performance. Do not
infer clinical significance, statistical superiority or edge-device behavior from
the present validation results.
