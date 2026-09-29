# Research track

## Current direction

Investigate adaptive edge inference for prehospital/ambulance lung POCUS with OpenPOCUS as a technical experimental proxy. Variable frame count per zone-sequence prediction is a supported primary direction; the DermaMNIST-224/DeiT-Tiny depth work remains pilot evidence, not the selected architecture. Exact frame counts, utility and policy mapping remain open (D020–D021).

## Evidence so far

- F001 established three shared-backbone operating points (4/8/12) without using the official test split.
- F001b measured each depth from 4 through 12. Validation accuracy peaks at depth 9, with non-monotonic behavior thereafter; compute is explicitly reported as fvcore FLOPs.
- F001c shows per-input complementarity but not a straightforward case for always extending inference: 9→12 yields 28 rescues and 35 regressions. Five examples are first correct at depth 12. A perfect hindsight oracle covers 92.2%, but this is an unattainable upper bound. Confidence/entropy patterns merit cautious follow-up, not a controller claim.

## Next research decision

Use F002 to test whether OpenPOCUS supports useful predictive behaviour, then characterize variable-frame configurations before selecting operating points or mapping them to simulated resource states. Compare any later policy with fixed baselines under one pre-specified protocol. Clarify the throughput unit with the teaching team on Thursday. Reserve the official test set for a final pre-specified evaluation.

## Interpretation guardrails

Separate aggregate validation metrics, per-input rescue/regression behavior, theoretical oracle headroom, and deployable policy performance. Do not call a depth optimal, infer clinical significance from errors, or treat confidence as a validated routing signal.
