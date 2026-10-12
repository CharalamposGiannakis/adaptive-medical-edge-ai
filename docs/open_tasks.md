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
- **Status:** F006a and F006a_2 analyses were included in the independently
  reviewed and merged T012 PR #27. Their measured validation evidence is
  historical; final numerical controller-policy acceptance remains pending
  independent PR re-review and Charalampos's recorded sign-off under the
  reported Salo delegation (D030).
- **Dependencies:** T010 and relevant T011 results.
- **Scope:** Analyze all 24 configurations of the actual deployable F005b
  seed-2026 checkpoint. Report strict macro-F1/FLOPs dominance, practical
  trade-offs, uncertainty, and candidate deterministic mappings for accepted
  states `b = 0.25/0.50/0.75/1.00`; `b = 0.00` defers inference. Screen all
  24 cells against provisional quality requirements anchored to the frozen
  MobileNetV2 validation reference before comparing compute among qualified
  cells. This is a research benchmark, not a clinical safety threshold. Propose
  what b measures, but do not freeze quality thresholds, normalization or the
  rule without human acceptance. Use validation only; no training or runtime
  controller here.
- **Acceptance criteria:** Reproducible source-backed comparison table, Pareto
  figure, full 24-cell quality screen, budget feasibility, sensitivity and
  proposed state mapping; focused logic tests; exact unavailable evidence and
  human decisions stated.

## T012b — Implement and simulate the provisional controller

- **Priority:** P1
- **Owner:** Charalampos
- **Status:** Provisional implementation and validation-only evaluation
  completed in [`F006b`](../experiments/F006b/README.md) and merged in PR #27.
  Merge did not accept the numerical policy. D030 proposes F007 headroom and
  retains the quality filter and recall-first rule; Charalampos approved that
  direction; independent PR re-review and Charalampos's recorded final sign-off
  under the reported Salo delegation remain pending.
- **Dependencies:** T012a's F006a and F006a_2 analyses; final team decision
  before T013.
- **Scope:** Implement a deterministic policy over all 24 measured F005b cells:
  apply the three provisional F002b-referenced quality requirements, then an
  exact `b × Cmax` FLOP budget, then prefer abnormal recall, lower FLOPs and a
  stable tie-breaker. Support continuous `b` while retaining D021's accepted
  `0/0.25/0.50/0.75/1.00` demonstration states. Use F005's
  `NestedAnytimeClip`, whose cache keys include width and frame; compare with
  minimum-compute, F006a Rule A and fixed full-width K16 policies. Log
  selections, deferrals, configuration changes and cached incremental work.
  Measure selected pruned points with chunked encoding in one comparable CPU
  job before making CPU-latency claims; FLOPs remain the resource constraint.
- **Acceptance criteria:** Reproducible simulation, scenarios, comparisons and
  CPU evidence or executable commands where the job is unavailable; no
  held-out-test use. Team review is required before treating this policy as
  fixed for T013. Learned control is optional research, not required here.

## T013a — Propose the final evaluation protocol

- **Priority:** P1
- **Owner:** Charalampos
- **Status:** [`F007`](../experiments/F007/README.md) is a proposed
  preregistration based on the merged T012 implementation. Canonical
  `verify_t006()` has structurally verified test-manifest metadata; no held-out
  test images have been decoded for evaluation and no test-image inference or
  predictive evaluation has occurred. B1/B2 from the first independent review
  and the later cross-platform protocol-hash blocker have been addressed in
  the proposed docs/config; D030 and F007 await final independent PR re-review
  and Charalampos's recorded sign-off under the reported Salo delegation.
- **Scope:** Fix six primary systems, the D021 resource states and a
  test-independent mixed schedule, duplicate-aware prediction/coverage and
  cost rules, one same-job CPU benchmark, group-cluster uncertainty, a fixed
  descriptive MobileNetV2-versus-`w0.5_k2` comparison and preselected
  matched-FLOP diagnostics before any predictive final-test access. The static
  width-0.5
  comparator is the **canonical nested F005b** width slice at validation-
  selected K8, not the separate historical 50%-only training control.
- **Acceptance criteria:** Independent PR review and Charalampos's recorded
  final sign-off under the reported Salo delegation on
  D030's numerical policy and the F007 protocol, including simulated budget
  assignment and descriptive comparison rules; record the accepted version
  in the repository before T013b execution. Do not treat this proposal as
  held-out evidence.

## T013b — Execute the accepted final comparison

- **Priority:** P1
- **Owner:** Charalampos
- **Status:** Phase A1 validation infrastructure is ready for review:
  exact-capacity routing, six canonical model paths, request/unit
  evidence, and a fail-closed final-test gate passed a bounded validation
  smoke. Held-out execution remains blocked on final independent PR review,
  Charalampos's recorded D030/F007 sign-off under the reported Salo delegation,
  and a committed approval receipt. Phase A2 analyses and CPU measurements
  remain open.
- **Dependencies:** Frozen T006/F002b/F002c/F005b handoffs and accepted F007
  protocol.
- **Acceptance criteria:** Use the approved six-system matched requests;
  report quality **with coverage**, exact full/incremental FLOPs, same-job CPU
  latency and throughput, dependency-group intervals, the fixed cost–quality
  and matched-FLOP comparisons, and descriptive per-device results. Keep raw
  evidence ignored and document proxy-dataset, validation-selection and
  device/label limitations. No test-driven retuning or comparison changes.

## Research backlog

- Optimization-based or learned resource control after the simple T012 policy.
- Broad architecture searches beyond the selected TinyViT backbone.
- More complex temporal selection or joint frame/depth routing if the course core
  is complete and T009 evidence warrants it.
