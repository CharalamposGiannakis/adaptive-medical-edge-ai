# F007 — final static-versus-adaptive evaluation (T013a protocol proposal)

**Status: proposed, not approved or executable.** This records the T013a design
before any held-out T006 test labels, images or predictions are accessed. PR #27
still contains the T012 evidence and provisional F006b controller. Charalampos
and Salo must review T012, freeze its policy, and approve this protocol in the
repository before T013b may implement a final-test command. No F007 result or
raw test artifact exists yet.

**Question.** Does the shared-weight adaptive TinyViT system provide a useful
predictive-quality, coverage and computational-cost trade-off against frozen
static alternatives? At which resource states does changing width and frame
count add value? OpenPOCUS is a technical proxy, not evidence of clinical or
prehospital effectiveness.

## 1. Frozen inputs and six primary systems

Use the unchanged D019 binary normal (0) versus definite abnormal (1) target,
the D023 T006 cohort/split, and the D024 duplicate-aware evaluation unit. Do
not retrain, retune a decision threshold, select a checkpoint, or change the
T006 manifests. All predictions use the relevant canonical model loader and
`probability >= 0.5` for abnormal. The three checkpoint identities are:

| Bundle | Frozen SHA-256 | Input path |
|---|---|---|
| F002b MobileNetV2 seed 2026 | `5bad132633fd6188c0e449a75f73294612479bb1f0dc1820d8e6c459f7bdb823` | D024 RGB 224 bilinear, ImageNet normalization |
| F002c TinyViT-5M epoch 8 | `30b70734426e84f429b83011abca04511540149d1fc3e416274c6fbcb14853b2` | F003/F004 RGB 224 bicubic, ImageNet normalization |
| F005b nested TinyViT seed 2026 | `18d601e3ea04a1ba572ef99484c8681c5a3011dd7d3eb28309cb32906ff9e702` | F003/F005 RGB 224 bicubic, ImageNet normalization |

The F005b canonical loader must also verify the bundle's `run.json` and
`importance.pt`. All K use the frozen nested spread-frame sampler; F002b uses
its accepted K16 D024 path. Different accepted resize kernels are retained
for predictive evaluation and recorded in cross-system comparisons.

| ID | System and locked selection | Full request FLOPs before alias multiplicity |
|---|---|---:|
| A | F002b MobileNetV2, fixed K16 | 5,107,788,033 |
| B | Original frozen F002c TinyViT, fixed K16 | 20,596,068,563 |
| C | Canonical **nested F005b** at fixed width 0.5, K8 | 7,266,300,179 |
| D | F006b minimum-quality-cost policy: the fixed cheapest of the quality-qualified F005b cells, `w0.5_k2`, or DEFER when over budget | 1,816,575,299 when served |
| E | F006a Rule A: highest **validation macro-F1** among all 24 budget-feasible F005b cells; no F006a_2 quality gate | selected cell's exact cost |
| F | F006b proposed rule: quality gate, then highest abnormal recall, lower FLOPs and stable name tie-break | selected cell's exact cost |

For C, the pre-test selection rule is maximum duplicate-aware validation
macro-F1 among the six width-0.5 K cells; exact ties go to lower FLOPs, then
lower K. The tracked F006a/F006a_2 table selects K8 (validation macro-F1
0.8740079365). C uses the canonical nested bundle's 50%-width slice; it is
**not** the separately trained 50%-only historical control, which has no
canonical checkpoint handoff. A's exact cost was reconstructed with D024's
strict `count_sequence_flops` on the frozen MobileNetV2 architecture; all
costs and checkpoint hashes must be verified again in the final preflight.

D, E and F use their tracked validation-derived selection algorithms with no
test-derived quality table. D/F's three inclusive validation requirements are accuracy
`>=225/254`, exact F002b macro-F1 `>=(336/365 + 114/143)/2`, and abnormal
recall `>=57/76`. `Cmax = 20,596,068,563` is the F005b full-width K16 cost. These
requirements, normalization, and F's rule remain **provisional** until T012
acceptance; the approved values and source hashes must be recorded before test
access. At each budget, a static A/B/C point runs only if its own full cost is
within the cap; otherwise it defers. No FLOP tolerance is allowed.

## 2. Matched resource requests and coverage

The primary resource states are D021's accepted
`b = {0, 0.25, 0.50, 0.75, 1.00}` with exact rational per-row limit
`B(b) = b × Cmax`. At zero, every system defers. **Primary statewise design:**
each frozen duplicate-aware test unit is requested once at each of these five
states, with a fresh clip cache for each row/request. All six systems face
identical unit × state pairs. This is a factorial stress test, not an estimate
of a real ambulance resource distribution. Report every state separately;
do not pool the five repeated observations as independent patients.

One **secondary heterogeneous schedule** uses each test unit once. Construct
unit IDs from the frozen T006 shared-recording graph without using labels for
assignment. Sort them by `(SHA256(UTF-8("F007-T013a-v1\n" + unit_id)), unit_id)`;
give sorted rank `i` the state in `(0, 0.25, 0.50, 0.75, 1.00)[i mod 5]`.
State counts therefore differ by at most one. Process requests in frozen
manifest order, applying one unit's state to every retained alias row in that
unit. The schedule is fixed for all systems, independent of labels, model
scores or device. Report the realized state counts, coverage and cost; do not
describe this deliberately balanced scenario as a deployment frequency.

Conditional on team acceptance of continuous `b`, test-independent
**sensitivity states** `0.05`, `0.67` and `0.83` may be evaluated statewise,
separately from the primary five-state and heterogeneous summaries. They were
already used in F006b's validation-only sweep/traces: 0.05 probes an
insufficient active budget, 0.67 an interior middle budget and 0.83 the
region where F's K12 choice becomes feasible. Exact one-FLOP boundary checks
are analytical/controller tests, not extra held-out quality searches. If the
team rejects continuous-budget use, omit this entire sensitivity set before
any test access and update the approved protocol accordingly.

For each unit × state, every retained alias receives the same budget and
configuration or the same DEFER. Each alias is one zone-sequence inference
request, so the cap applies to **each row**; duplicate aliases each incur
their own actual work. A technical inference failure fails the run and is not
silently turned into a policy deferral. Record row requests and unit votes
separately. At each system/state and for the heterogeneous schedule report:

- requested, completed and deferred rows **and** duplicate-aware units;
- coverage = completed units / requested units, with each deferral reason;
- accuracy, macro-F1, normal/abnormal recall and AUROC on completed units,
  always beside coverage and class support;
- a paired quality comparison on the **common completed units** when two
  systems have different coverage, with its denominator shown.

No prediction metric is assigned to a zero-coverage cell. DEFER is not an
incorrect classification. Do not hide deferrals by calculating only selective
accuracy. A separate clinical utility for deferral is not asserted here.

## 3. Prediction aggregation and cost

Use `src/f002_protocol.py`'s frozen `recording_units` mapping: a retained
same-target shared-recording component is one unit, otherwise a row is a
singleton. Keep every alias prediction, average its **probability** within
the unit, then apply threshold 0.5 once. Validate target and split_group
agreement within every unit and exact row coverage. Compute TN/FP/FN/TP,
accuracy, macro-F1 and both recalls with
the frozen `binary_metrics` formulas; calculate AUROC from unit targets and
mean probabilities only when both classes occur. For zero completed units,
all predictive metrics are unavailable. For a class absent in a descriptive
stratum, show that class's recall, macro-F1 and AUROC as unavailable in the report,
along with the confusion counts; never infer class performance from a zero
denominator. Adaptive metrics come from actual routed per-unit predictions,
never averages of configuration-level F1 values.

Use the same strict one-FMA-one-FLOP fvcore convention for A/B/C and every
F005b cell selected by D/E/F. For each system/state report configuration and
its exact FLOPs, total **full** FLOPs over completed alias-row requests,
mean full FLOPs per completed row (unavailable when none completes), and mean
full FLOPs over **all** requested rows (deferred rows contribute zero). Count any exact-budget violation as a
failed run. Report active-to-active configuration switches and transitions
to/from DEFER separately; the heterogeneous schedule's switch count depends
on the frozen manifest order and is descriptive, not a deployment forecast.

The primary factorial and heterogeneous requests each start with an empty
clip cache, so incremental work equals full request work. Separately, serve
**every retained alias clip** under each of D, E and F through both increasing
and decreasing D021 sequences with F005b `NestedAnytimeClip`. Start a fresh
cache for each alias × policy × order; never share cached features across
aliases, policies or orders. Key features by width and frame, verify every
cached logit against direct inference, and report newly encoded frames and
exact modeled incremental FLOPs. These
within-clip savings must not be substituted for the independent-request full
cost in the primary comparison. FLOPs are not energy, battery life or latency.

## 4. Comparable CPU latency and throughput

Run **one** CPU benchmark job for the union of A/B/C, every distinct D/E/F
configuration reached in the approved states, and the supplementary
matched-FLOP point below. Preload the same 200 frozen D024 validation timing
IDs on each model's canonical preprocessing path; use D024's first 50 warm-up
IDs, then all 200 timed IDs, for five repeats per point. Use one named CPU,
one process and fixed four threads; record CPU model/node, OS, library
versions, torch interop threads and the exact request IDs/order. Rotate point
order between repeats. The primary CPU path encodes frames in chunks of at
most four for **every** per-frame encoder, with an unchunked direct pass as a
secondary measurement. Before timing, require direct/chunked logit agreement
within `1e-4` and no threshold-class change on all timed requests; otherwise
stop and seek a documented methodology decision. F002b receives bilinear
tensors, while TinyViT systems receive bicubic tensors from the same raw IDs.

Time frame selection, width setting, encoding, pooling and head, with already
decoded/normalized tensors; exclude JPEG decode and preprocessing from this
**model-only** clock. Report median and p95 per-request latency, raw repeats,
completed zone-sequence predictions/s and processed frames/s using timed-pass
wall time. Measure routing overhead and grouped-versus-interleaved switching
on an identical point × request multiset, with cache reset between clips.
Run a **separate** practical JPEG decode/resize/normalize/model path on the
same frozen 200 timed IDs and 50 warm-up IDs, five repeats, host and thread
settings, after an untimed read pass, and label its latency/throughput
end-to-end. Deferral overhead and completed throughput accompany coverage.
Existing F002 DAS-5, F004 Xeon and F006b Ryzen absolute times are context,
not a controlled cross-system comparison.

## 5. Uncertainty and preselected matched comparisons

First aggregate alias probabilities into one record per duplicate-aware unit,
retaining that unit's alias-row costs and frozen split_group. Resample the
**test split_group dependency groups**, not alias rows or the five repeated
states, with replacement for 5,000 paired bootstrap draws using NumPy PCG64
seed `13013`. Each draw carries all member unit records, budget states and
predictions, preserving multiplicity when a group is drawn twice; use
identical sampled groups for compared systems. Report point estimates and
2.5/97.5 percentile intervals for coverage, accuracy, macro-F1, both recalls,
AUROC where defined, mean full FLOPs/request, and prespecified paired quality
differences. A draw without
both classes is invalid for class-dependent metrics; report valid-draw counts,
and suppress an interval with fewer than 4,000 valid draws. These intervals
reflect known-group sampling uncertainty on this proxy dataset, not
biological-patient independence or model-seed uncertainty. No unadjusted
multiple-comparison p-values determine a winner.

**Near-matched computation (supplementary, chosen before test):** accept at
most a 1% relative difference in exact full FLOPs, measured against A's cost.
Compare A at K16 (5,107,788,033 FLOPs) with the **same original F002c
checkpoint** at K4 (5,149,017,395 FLOPs), a 0.807% cost difference. K4 is an additional
diagnostic point, not a replacement for primary B at K16. Use identical
units, probability aggregation and paired group-bootstrap differences at
an unrestricted budget; report cost, quality and coverage. F002c K4 exceeds
the exact D021 `b=0.25` cap by 254.25 FLOPs, so it must not be counted as
feasible at that state. This pair meets the proposed 1% matching tolerance
and isolates a near-matched FLOP comparison. It does not by itself measure
the benefit of adaptivity.

**Adaptive-value contrasts (preselected):** for each informative D021 state
`b = {0.25, 0.50, 0.75, 1.00}` and the secondary heterogeneous schedule,
report paired F-minus-D and F-minus-E differences in coverage, completed-unit
accuracy, macro-F1, abnormal and normal recall, AUROC where supported, and
mean full FLOPs per requested row. Predictive differences use common completed
units, with that denominator shown; coverage and cost use all matched
requests. Report these jointly with bootstrap intervals and configurations;
do not select favorable states after seeing test results. At `b=0`, all
systems defer, so report the common zero coverage/cost without a predictive
contrast. No single winner rule is specified.

**Candidate matched predictive quality (chosen on validation):** compare A
against D's `w0.5_k2` at `b=0.25`; both pass the exact cap. Validation
macro-F1 is 0.858875 versus 0.869706, with D's exact cost 1,816,575,299
FLOPs. This pair is fixed regardless of its held-out result. Proposed
descriptive noninferiority margins are `-0.02` for D-minus-A macro-F1 and
accuracy, and `-0.05` for abnormal recall; these are **not clinical safety
margins** and require team approval. Claim “at least matched predictive
quality on these three metrics at lower cost” only if the lower bounds of all
three paired 95% group-bootstrap intervals exceed their respective margins,
both systems have complete coverage on the pair, and the intervals are
available.
Otherwise report **match not established**, the observed gaps and cost ratio;
do not search test configurations or relax a margin after seeing results.

Per-device results are descriptive: assign a unit to its common device code,
or `mixed` if aliases disagree; show units, class counts, coverage, confusion
counts and supported metrics for each code. Device and label are entangled in
OpenPOCUS, and small strata may have no abnormal examples. Do not infer
device-adjusted or causal superiority.

## 6. Outputs, limits and execution gate

T013b should write ignored `artifacts/F007/` raw row probabilities, unit
aggregations, resource assignments, deferrals, per-request FLOPs, cache
diagnostics, CPU raw timings and a run record. The run record must contain
the accepted protocol's committed SHA, code/checkpoint/manifest hashes,
environment, exact request IDs, seed, status and every attempted execution;
fail on mismatches. Version concise derived tables and figures only after
review: system/configuration provenance, per-state and mixed-schedule
quality–coverage–cost, deferral reasons, paired contrasts, per-device
descriptives, CPU timing/throughput, and quality-versus-FLOPs plots annotated
with coverage. Separate measurements from interpretation.

No T013b final-test command exists in this branch. Its future command must
first verify (1) PR #27 was reviewed and merged or an explicitly revised T012
policy was accepted, (2) Charalampos and Salo approved the fixed K8 static
comparator, budget schedule, quality margins, benchmark path and this entire
protocol in a tracked decision, (3) an independent protocol review was done,
and (4) frozen source/protocol/checkpoint hashes match. Only after those gates
may it read the held-out test manifest **as bytes solely to check its frozen
hash**, without parsing rows or labels. A manifest-hash mismatch fails closed.
Only after that check may it parse labels or images and run inference. No
model/threshold fitting, post-test configuration selection, budget
reassignment or
metric-driven rerun is allowed. A technical failure may be rerun with the
identical frozen configuration and all attempts retained.

The available identifiers protect only known family/shared-recording
dependencies, not biological patients. One test split and one selected seed
per frozen checkpoint, validation reuse in prior selection, device/label
confounding, simulated FLOP budgets, and absence of edge-device energy or
clinical data limit the conclusions. The team must decide whether it wants a
separate scalar utility for deferral; this protocol makes no such claim.
