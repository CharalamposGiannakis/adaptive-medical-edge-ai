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

Historical acceptance criteria and evidence remain in
[`decision_log.md`](decision_log.md), [`experiment_log.md`](experiment_log.md),
[`experiments/F002/README.md`](../experiments/F002/README.md) and
[`experiments/F002c/README.md`](../experiments/F002c/README.md).

## T009 — Characterize useful adaptive dimensions (issue #11)

- **Priority:** P1
- **Owner:** Salom
- **Status:** Done (PR #24; issue #11 closed). F003 (2026-10-06) proposes frame
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

## T010 — Construct shared adaptive operating points

- **Priority:** P1
- **Owner:** Charalampos
- **Dependency:** T009.
- **Scope:** Select and implement at least three useful operating points from the
  T009 evidence, sharing the relevant TinyViT model/weights or a documented
  shared-weight extension.
- **Acceptance criteria:** Exact configurations, shared-weight relationship and
  raw predictive/cost measurements are reproducible. Do not invent points merely
  to mirror the number of budget states.

## T011 — Integrate one compression technique

- **Priority:** P1
- **Owner:** Charalampos
- **Dependency:** Stable operating points from T010.
- **Scope:** Choose one technique supported by model/backend evidence and evaluate
  it against the appropriate uncompressed reference.
- **Acceptance criteria:** Reproducible method and artifacts; predictive and cost
  effects measured under the same convention. INT8 post-training quantization and
  structured pruning remain candidates, not commitments.

## T012 — Implement a resource-aware policy

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T010 and relevant T011 results.
- **Scope:** Map active normalized resource states to measured operating points;
  `b = 0.00` defers inference. Compare with fixed/static alternatives under the
  same protocol.
- **Acceptance criteria:** Reproducible policy, scenarios and comparisons; no
  held-out-test use during policy development. A learned or optimization-based
  controller is optional research work, not required for the core.

## T013 — Final static-versus-adaptive evaluation

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T010–T012 and a frozen final evaluation protocol.
- **Scope:** Compare the frozen F002b MobileNetV2 static reference with the selected
  TinyViT adaptive/compressed system.
- **Acceptance criteria:** Pre-specify any held-out-test use before inference;
  report predictive metrics, consistent FLOPs, throughput and relevant latency;
  preserve raw evidence and state uncertainty, proxy-dataset limits and any
  validation-driven selection bias.

## Research backlog

- Optimization-based or learned resource control after the simple T012 policy.
- Broad architecture searches beyond the selected TinyViT backbone.
- More complex temporal selection or joint frame/depth routing if the course core
  is complete and T009 evidence warrants it.
