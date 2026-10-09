# Active task queue

GitHub Issues carry live assignment, discussion and pull-request status. This file
keeps the current technical queue and acceptance outcomes without repeating the
full history of completed work. The active pathway is in
[`design_pathway.md`](design_pathway.md).

## Completed foundation

- **T001–T003:** Integrated and audited OpenPOCUS, and documented the proxy
  prehospital/ambulance POCUS scenario. See the dataset README and audit.
- **T004–T005:** Fixed the supported binary zone-sequence task and recorded
  teaching-team feedback/resource-state conventions (D019–D021).
- **T006:** Froze the reproducible cohort, known-dependency-group-disjoint split
  and manifests (D023; PR #20; issue #8).
- **T007:** Fixed the F002 qualification/static-reference protocol (D024; PR #22;
  issue #9).
- **T008:** Ran F002a/F002b and froze the MobileNetV2 static reference (D025;
  PR #23; issue #10 closed).
- **F002c:** Ran the bounded TinyViT exploration. Charalampos and Salo selected
  the epoch-8 checkpoint as the adaptive-system backbone (D026). MobileNetV2
  remains the static reference.
- **T009:** Characterized frame count as the primary adaptive axis (F003, D027;
  PR #24, issue #11).
- **T010:** Built the six-point shared frame-count family on unchanged D026
  weights (F004, D028; PR #26, issue #12).
- **T011:** Added nested structured pruning with distillation as compute
  compression (F005b, D029; PR #26, issue #13). The shared checkpoint stays
  about 20.4 MB; narrower widths reduce inference FLOPs.

Historical acceptance criteria and evidence remain in
[`decision_log.md`](decision_log.md), [`experiment_log.md`](experiment_log.md),
[`experiments/F002/README.md`](../experiments/F002/README.md) and
[`experiments/F002c/README.md`](../experiments/F002c/README.md).

## T009 — Characterize useful adaptive dimensions (issue #11; completed)

- **Priority:** P1
- **Owner:** Salom
- **Status:** Done (PR #24; issue #11 closed). F003 (2026-10-06) established frame
  count as the primary axis (D027); see [`experiments/F003/README.md`](../experiments/F003/README.md).
- **Foundation:** Use the canonical shared frozen F002c TinyViT epoch-8 checkpoint
  at [`checkpoints/f002c_tinyvit5m_epoch8.pt`](../checkpoints/f002c_tinyvit5m_epoch8.pt),
  selected in D026. Keep the T006 split and duplicate-aware evaluation units
  frozen; use train/validation only.
- **Goal:** Empirically and analytically characterize a useful adaptive
  configuration space for T010.
- **Scope:** Frame count K is already supported by preliminary same-checkpoint
  K4/K8/K16 evidence. Measure useful K values and their predictive/FLOPs/runtime/
  throughput behavior under a consistent convention. Bounded experiments may
  also examine TinyViT-supported depth/exits, spatial resolution, compression
  interactions, input-difficulty/confidence signals or another justified
  dimension. The list is optional; evidence should eliminate unnecessary
  complexity.
- **Acceptance criteria:** Reproducible configurations/commands and raw validation
  evidence; no T006 test inference; consistent cost/runtime measurements for
  candidate configurations; explicit measured-versus-interpreted conclusions; an
  evidence-backed selection of the adaptive axes and candidate configuration space
  passed to T010.

## T010 — Construct shared adaptive operating points (issue #12; completed)

- **Priority:** P1
- **Owner:** Salo
- **Status:** Done with PR #26. F004 (2026-10-07) builds K1/K2/K4/K8/K12/K16 on
  the D026 weights as one family and does not adopt fine-tuning across K (D028,
  accepted); see
  [`experiments/F004/README.md`](../experiments/F004/README.md).
- **Dependency:** T009.
- **Scope:** Select and implement at least three useful operating points from the
  T009 evidence, sharing the relevant TinyViT model/weights or a documented
  shared-weight extension.
- **Acceptance criteria:** Exact configurations, shared-weight relationship and
  raw predictive/cost measurements are reproducible. Do not invent points merely
  to mirror the number of budget states.

## T011 — Integrate one compression technique (issue #13; completed)

- **Priority:** P1
- **Owner:** Salo
- **Status:** Done with PR #26. F005b (nested structured pruning with
  distillation) satisfies its recorded adoption rule and is the accepted compute
  compression step (D029); F005a dynamic INT8 demonstrated file-size reduction
  but was not selected. See
  [`experiments/F005/README.md`](../experiments/F005/README.md).
- **Dependency:** Stable operating points from T010.
- **Scope:** Choose one technique supported by model/backend evidence and evaluate
  it against the appropriate uncompressed reference.
- **Acceptance criteria:** Reproducible method and artifacts; predictive and cost
  effects measured under the same convention. INT8 post-training quantization and
  structured pruning remain candidates, not commitments.

## T012a — Design resource-aware controller configurations

- **Priority:** P1
- **Owner:** Charalampos
- **Status:** Draft analysis ready for review on
  `T012a-resource-aware-controller-design`; the
  [`F006a design analysis`](../experiments/F006a/README.md) is a proposal
  awaiting Charalampos and Salo's review, not an accepted policy decision.
- **Dependencies:** T010 and relevant T011 results.
- **Scope:** Analyze all 24 configurations of the actual deployable F005b
  seed-2026 checkpoint. Report strict macro-F1/FLOPs dominance, practical
  trade-offs, uncertainty, and candidate deterministic mappings for accepted
  states `b = 0.25/0.50/0.75/1.00`; `b = 0.00` defers inference. Propose what
  b measures, but do not freeze the normalization or the rule without human
  acceptance. Use validation only; no training or runtime controller here.
- **Acceptance criteria:** Reproducible source-backed comparison table, Pareto
  figure, rule sensitivity and proposed state mapping; focused logic tests;
  exact unavailable evidence and human decisions stated.

## T012b — Implement and simulate the approved controller

- **Priority:** P1
- **Owner:** Charalampos
- **Status:** Awaiting acceptance of T012a's budget normalization, configuration
  set and selection rule.
- **Dependencies:** T012a's reviewed design and accepted decision.
- **Scope:** Implement a deterministic policy over the selected Pareto points
  and compare with fixed/static alternatives under the same protocol. Use F005's
  `NestedAnytimeClip`, whose cache keys include both width and frame; F004's
  `AnytimeClip` is not suitable across widths. Log the selected point per request
  for a runtime trace. Before setting CPU latency budgets, measure the selected
  F005b width × K points with chunked encoding in one comparable CPU job. F004
  Table 8 measured chunks of four only for the unpruned family.
- **Acceptance criteria:** Reproducible simulation, scenarios and comparisons;
  no held-out-test use during development. Learned or optimization-based control
  is optional research work, not required for the core.

## T013 — Final static-versus-adaptive evaluation

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T010–T012 and a frozen final evaluation protocol.
- **Scope:** Compare the frozen F002b MobileNetV2 static reference with the selected
  TinyViT adaptive/compressed system.
- **Acceptance criteria:** Pre-specify any held-out-test use before inference;
  report predictive metrics, consistent FLOPs, throughput and relevant latency;
  preserve raw evidence and state uncertainty, proxy-dataset limits and any
  validation-driven selection bias. From the T010/T011 reviews: freeze the
  operating points and the policy on validation before touching test; add the
  static TinyViT points (D026 at K16, the 50%-only model at a fixed K) as
  further fixed baselines next to MobileNetV2; report one matched-accuracy and
  one matched-FLOPs comparison with intervals; report per-device results
  (device and label are entangled); and use CPU latencies from a single job.

## Research backlog

- Optimization-based or learned resource control after the simple T012 policy.
- Broad architecture searches beyond the selected TinyViT backbone.
- More complex temporal selection or joint frame/depth routing if the course core
  is complete and T009 evidence warrants it.
