# Research track

## Current direction

Investigate resource-aware inference using a shared-weight DeiT-Tiny with selectable transformer depth on DermaMNIST-224. The selected scenario and utility objective are still open.

## Evidence so far

- F001 established three shared-backbone operating points (4/8/12) without using the official test split.
- F001b measured each depth from 4 through 12. Validation accuracy peaks at depth 9, with non-monotonic behavior thereafter; compute is explicitly reported as fvcore FLOPs.
- F001c shows per-input complementarity but not a straightforward case for always extending inference: 9→12 yields 28 rescues and 35 regressions. Five examples are first correct at depth 12. A perfect hindsight oracle covers 92.2%, but this is an unattainable upper bound. Confidence/entropy patterns merit cautious follow-up, not a controller claim.

## Next research decision

Use the F001c per-sample evidence to discuss whether input-adaptive depth is worth pursuing, then choose the scenario and define utility/constraints with Dolly/TA. Only after those choices should a simple candidate policy be compared with fixed-depth baselines. Reserve the official test set for a final pre-specified evaluation.

## Interpretation guardrails

Separate aggregate validation metrics, per-input rescue/regression behavior, theoretical oracle headroom, and deployable policy performance. Do not call a depth optimal, infer clinical significance from errors, or treat confidence as a validated routing signal.
