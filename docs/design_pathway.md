# Current experimental pathway

## Purpose and authority

This is the living pathway for the adaptive phase. Accepted decisions in
[`decision_log.md`](decision_log.md) take precedence, while historical planning
and experiment details remain in the decision log and experiment READMEs. Update
this pathway when evidence redirects the work; do not force new experiments to
follow assumptions from the completed architecture-selection phase.

## Frozen foundation

- **Scenario and task (D019–D020):** OpenPOCUS is a technical proxy for
  resource-aware prehospital/ambulance lung POCUS. Predict normal versus definite
  abnormal for one exam × lung-zone sequence; do not make trauma, pneumothorax or
  clinical-deployment claims.
- **Data (D023):** T006 fixes the cohort, labels, known-dependency groups,
  train/validation/test split and duplicate-aware evaluation units. The released
  identifiers cannot certify biological-patient independence.
- **Static reference (D025):** F002b MobileNetV2, seed 2026, K16, is immutable:
  validation macro-F1 0.859, abnormal recall 0.750 and 5.11 GFLOPs under the F002
  convention.
- **Adaptive backbone (D026):** The selected F002c TinyViT-5M epoch-8 checkpoint
  is the backbone for T009 onward. Use its canonical shared frozen copy at
  [`checkpoints/f002c_tinyvit5m_epoch8.pt`](../checkpoints/f002c_tinyvit5m_epoch8.pt).
  Its same-checkpoint K4/K8/K16 validation macro-F1 values are
  0.8602/0.8676/0.8737. This choice does not replace the MobileNetV2 static
  reference or rewrite F002 history.
- **Held-out test:** T009–T012 use train and validation only. The T006 test
  partition remains untouched until a later pre-specified final evaluation.

## Active pathway

```text
Frozen T006 cohort, split and evaluation units
                    |
Frozen MobileNetV2 F002b static reference (D025)
                    |
TinyViT F002c adaptive backbone (D026)
                    |
T009 adaptive-dimension playground and cost characterization (done)
                    |
T010 select at least three useful shared-model operating points (done)
                    |
T011 integrate one compression method (done)
                    |
T012a design the resource-aware policy from validation evidence (next)
                    |
T012b implement and simulate the accepted policy
                    |
T013 final static-versus-adaptive evaluation
```

## T009 — characterize the adaptive configuration space

T009 is exploratory. Start from the selected TinyViT checkpoint and measure which
configurations offer useful quality–cost trade-offs. Frame count K is already a
promising axis: the same F002c checkpoint retained strong validation macro-F1 at
K4, K8 and K16. T009 should extend or refine those measurements where doing so can
change the operating-point choice.

Bounded investigations may include:

- frame count, temporal work and additional useful K values;
- architecture-supported depth or exit points;
- spatial resolution;
- compression interactions;
- input-difficulty or confidence signals;
- another small dimension supported by TinyViT and the dataset.

This list is permission to experiment, not a requirement to implement every axis.
Reject dimensions that add cost or complexity without useful evidence. Avoid a
model zoo or broad search. Small exploratory ideas need an experiment ID,
configuration/command and raw evidence, but do not need a new decision entry or a
large preregistration before each run.

For candidate configurations, measure predictive behavior and use a consistent
method for FLOPs, latency, completed sequence throughput and processed-frame
throughput. Input-frame percentages alone are not compute or runtime results. Keep
threshold-0.5 metrics available for comparison and keep tuned thresholds, if
explored, clearly separate.

T009 ends with an evidence-backed description of the adaptive configuration space:
which axes are useful, which configurations remain candidates, their measured
costs, and which complexity was rejected. It does not map the normalized resource
states to configurations; that belongs to T012.

## T010–T013 — build and evaluate the adaptive system

- **T010:** Select at least three useful operating points supported by the shared
  TinyViT model/checkpoint or a documented shared-weight extension. Record exact
  configurations and raw measurements. Do not manufacture points to match the
  number of controller states. Done in F004: K = 1, 2, 4, 8, 12, 16 on the D026
  weights as one family with anytime upgrades (D028, accepted).
- **T011:** Integrate one evidence- and backend-supported compression technique and
  compare it with the appropriate uncompressed reference. Done in F005: nested
  structured pruning of the hidden channels with distillation, widths 100/75/50/25%
  in one 20.4 MB weight file (D029, accepted). This reduces inference compute,
  not the shared file size; dynamic INT8 was measured but not adopted.
- **T012a:** Analyze the actual deployable F005b seed-2026 width × K grid and
  propose the FLOP-budget definition, useful configurations and deterministic
  selection rule. The accepted normalized states are `b = 0.25, 0.50, 0.75,
  1.00`; `b = 0.00` means defer/no inference. The proposed normalization and
  rule in [`experiments/F006a/README.md`](../experiments/F006a/README.md)
  require human acceptance; this analysis does not implement the controller.
- **T012b:** Implement and simulate the accepted policy over measured frontier
  points, serve width changes with `NestedAnytimeClip`, and compare fixed
  alternatives. Measure chunked CPU latency for the selected pruned width × K
  points before turning FLOPs into latency budgets.
- **T013:** Run the pre-specified final comparison between the frozen MobileNetV2
  static reference and the adaptive/compressed system. Report predictive metrics,
  consistent FLOPs, throughput and relevant latency, with uncertainty and dataset
  limitations. Any use of the held-out T006 test partition requires the final
  protocol to be fixed first.

## Continuing guardrails

Keep T006 and F002b immutable, keep the F002c epoch-8 checkpoint as the selected
adaptive backbone (D029 deploys its pruned and fine-tuned descendant), protect
the held-out test partition during development, and use
one comparable cost/runtime convention for final candidate operating points.
Reproducibility and clear measured-versus-interpreted reporting remain mandatory;
the adaptive phase does not otherwise inherit arbitrary restrictions from the
completed F002 architecture-selection funnel.
