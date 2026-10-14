# F007 — final static-versus-adaptive evaluation (T013a protocol proposal)

**Status: proposed, not approved for held-out execution.** This records the T013a design
before held-out test-image inference or predictive evaluation. The canonical
`verify_t006()` has already parsed the frozen test manifest for structural
counts and fingerprints; this is permitted provenance verification, not model
evaluation. PR #27 merged the T012 implementation and validation evidence;
that merge did not accept its numerical controller policy. Charalampos reports
that Salo delegated completion of the remaining work to him. D030 and F007
remain proposed pending final independent PR review and Charalampos's sign-off
under that delegation. No held-out F007 prediction or raw test-result artifact exists.
The JSON status required for a future accepted final protocol is
`accepted_final_protocol`; the current proposed status and
`authorizes_test_inference=false` grant no test-inference authorization.
This README governs methodology details not duplicated in the compact JSON
contract. Resolve any disagreement before accepting the protocol.

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
`probability >= 0.5` for abnormal.

The canonical `test.csv` SHA-256 in
[`configs/f002_protocol.json`](../../configs/f002_protocol.json) is
`439315130efb419032d1a686c1a871f0b60443e635a0845e03ebf4325db55c38`.
The proposed [`F007 machine-readable protocol`](../../configs/f007_protocol.json)
pins and references that F002 protocol's T006 fingerprints; it does not define
a second test-manifest hash. Its F002 protocol-file SHA-256 is
`84762d37e9de78fb906a05f8f9a5b7daeb65226d59821393db6517a1e4faa70f`,
computed from the canonical LF bytes stored in Git. Verify against the Git
blob at the committed F007 revision selected for execution rather than raw
checkout bytes, which can have CRLF on an existing Windows checkout. Compare
a local copy after CRLF-to-LF
normalization against that blob to detect other changes. `.gitattributes`
requires LF for protocol JSON on future checkouts. The three checkpoint
identities are:

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
test-derived quality table. D/F's three inclusive validation requirements are
accuracy `>=225/254`, exact F002b macro-F1
`>=(336/365 + 114/143)/2`, and abnormal recall `>=57/76`. These limits and
F's recall-priority rule remain **provisional** pending independent PR re-review
and Charalampos's recorded final sign-off under the reported delegation.
F006b historically used `Cmax = Cmodel = 20,596,068,563` FLOPs,
the full-width K16 cost. D030 proposes a subsequent F007-only budget scale:
`Ccapacity = (11/10) × Cmodel = 22,655,675,419.3` FLOPs. This 10% headroom
is a **simulated per-request allowance**, not measured hardware capacity,
battery level, CPU utilization or energy. F006b's original measurements and
controller code retain their original normalization. At each revised budget,
a static A/B/C point runs only if its own full cost is within the cap;
otherwise it defers. No FLOP tolerance is allowed.

## 2. Matched resource requests and coverage

The primary resource states are D021's accepted
`b = {0, 0.25, 0.50, 0.75, 1.00}` with exact rational per-row limit
`B(b) = b × Ccapacity = b × (11/10) × Cmodel`. The same revised cap applies
to **all six systems** at each state. At zero, every system defers.
**Primary statewise design:** Each frozen duplicate-aware test unit is requested
once at each of these five
states, with a fresh clip cache for each row/request. All six systems face
identical unit × state pairs. This is a factorial stress test, not an estimate
of a real ambulance resource distribution. Report every state separately;
do not pool the five repeated observations as independent patients.

Reapply the unmodified [F006b selection rule](../F006b/controller.py) to all
24 tracked [F006a_2 validation profiles](../F006a_2/quality_screen.csv)
(CSV SHA-256 `726038faa90e86a405367fb1cfdd5043be68fe8672cb2bd248a7b2205447027d`).
Compare integer point costs with exact rational budgets. The resulting D/E/F
selections and static feasibility under the **proposed** 10% headroom are:

| D021 `b` | Exact `B(b)` FLOPs | Static A/B/C feasible | D | E | F |
|---:|---:|---|---|---|---|
| 0 | 0 | none | DEFER | DEFER | DEFER |
| 0.25 | 5,663,918,854.825 | A | `w0.5_k2` | `w0.5_k2` | `w0.5_k2` |
| 0.50 | 11,327,837,709.65 | A, C | `w0.5_k2` | `w0.75_k8` | `w1.0_k8` |
| 0.75 | 16,991,756,564.475 | A, C | `w0.5_k2` | `w0.75_k8` | `w1.0_k12` |
| 1.00 | 22,655,675,419.3 | A, B, C | `w0.5_k2` | `w1.0_k16` | `w1.0_k16` |

F reaches exactly four active configurations over continuous `b`, at these
inclusive entry thresholds (fractions are exact; decimals are display only):

| Selected point | First selected `b = FLOPs/Ccapacity` |
|---|---:|
| `w0.5_k2` | `18,165,752,990/226,556,754,193` ≈ 0.080181908744 |
| `w1.0_k8` | `102,980,344,510/226,556,754,193` ≈ 0.454545462027 |
| `w1.0_k12` | `154,470,515,070/226,556,754,193` ≈ 0.681818185559 |
| `w1.0_k16` | `10/11` ≈ 0.909090909091 |

Validation-profile-only headroom sensitivity, with the same rule and five
states, confirms the effect of normalization without image inference:

| Headroom above `Cmodel` | `b=0` | `b=0.25` | `b=0.50` | `b=0.75` | `b=1` |
|---:|---|---|---|---|---|
| 0% (historical F006b) | DEFER | `w0.5_k2` | `w0.5_k2` | `w1.0_k8` | `w1.0_k16` |
| 5% | DEFER | `w0.5_k2` | `w1.0_k8` | `w1.0_k12` | `w1.0_k16` |
| 10% (F007 proposal) | DEFER | `w0.5_k2` | `w1.0_k8` | `w1.0_k12` | `w1.0_k16` |
| 20% | DEFER | `w0.5_k2` | `w1.0_k8` | `w1.0_k12` | `w1.0_k16` |

The mapping is stable across 5–20% headroom **at these states**. This does
not establish physical device headroom. Across the examined 0%, 5%, 10% and
20% assumptions, A/B/C static feasibility and E's maximum-validation-F1
mapping stay unchanged at the five states; the headroom changes **F's**
evaluated mapping at `b=0.50/0.75`. Every selected point is within its exact
budget. The `w1.0_k12` choice is validation-sensitive: its TN=163,
FP=15, normal recall=163/178 and accuracy=225/254 place accuracy exactly at
the provisional threshold. Requiring one more correct validation unit would
disqualify it. Relative to `w1.0_k8`, it detects two more abnormal units but
adds four false positives and roughly 50% more FLOPs. Recall priority is not
a clinical safety guarantee.

One **secondary heterogeneous schedule** uses each test unit once. Obtain
`unit_id` from `src.f002_protocol.recording_units` for every retained row.
For a verified shared-recording component, it is `shared:` followed by the
first 16 lowercase hex characters of SHA-256 over the UTF-8 string formed by
sorting **all** component `(exam_id, zone_id)` nodes (including excluded
aliases), formatting each as `exam_id/zone_id`, and joining them with `|`.
For a singleton it is `singleton:exam_id/zone_id`. The same returned ID is
used for all retained aliases of one component; do not derive a different ID
from the retained subset. Assignment uses IDs only, never labels or scores.
Sort units by `(SHA256(UTF-8("F007-T013a-v1\n" + unit_id)), unit_id)`;
give sorted rank `i` the state in `(0, 0.25, 0.50, 0.75, 1.00)[i mod 5]`.
State counts therefore differ by at most one. Process requests in frozen
manifest order, applying one unit's state to every retained alias row in that
unit. The schedule is fixed for all systems, independent of labels, model
scores or device. Report the realized state counts, coverage and cost; do not
describe this deliberately balanced scenario as a deployment frequency.

The final predictive-evaluation schedule contains only these five states.
Continuous `b ∈ [0,1]` remains a controller capability; exact boundary
checks and headroom sensitivity use validation profiles without adding
held-out states.

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
Because each policy depends on `b` and frozen validation profiles, factorial
coverage is deterministically 0 or 1 for every system × state. A structural
coverage interval of `[0,0]` or `[1,1]` adds no statistical information;
report the exact value without a bootstrap interval. Mixed-schedule coverage
differs by system; show requested/completed unit counts, class support and
the completed-unit denominator beside **every** predictive metric. Its AUROC
is descriptive because it mixes predictions from different configurations.

## 3. Prediction aggregation and cost

Use `src/f002_protocol.py`'s frozen `recording_units` mapping: a retained
same-target shared-recording component is one unit, otherwise a row is a
singleton. Keep every alias prediction, average its **probability** within
the unit, then apply threshold 0.5 once. Validate target and split_group
agreement within every unit and exact row coverage. Compute TN/FP/FN/TP,
accuracy, macro-F1 and both recalls with
the frozen `binary_metrics` formulas; calculate AUROC from unit targets and
mean probabilities only when both classes occur. For zero completed units,
all predictive metrics are unavailable. T013b must wrap `binary_metrics` with
an explicit class-support check: when one class is absent, retain defined
accuracy and present-class recall but return unavailable for the absent-class
recall, macro-F1 and AUROC instead of the helper's zero placeholder. Show
confusion counts and support; never infer class performance from a zero
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

Run **one** CPU benchmark job for exactly these nine preselected points:

1. F002b MobileNetV2 K16 (A).
2. Original F002c TinyViT K16 (B).
3. Original F002c TinyViT K4 (near-matched FLOP diagnostic).
4. Nested F005b `w0.5_k8` (C).
5. Nested F005b `w0.5_k2` (D and F/E at `b=0.25`).
6. Nested F005b `w0.75_k8` (E at `b=0.50/0.75`).
7. Nested F005b `w1.0_k8` (F at `b=0.50`).
8. Nested F005b `w1.0_k12` (F at `b=0.75`).
9. Nested F005b `w1.0_k16` (F/E at `b=1`; secondary equal-FLOP pair).

Preload the same 200 frozen D024 validation timing IDs on each model's
canonical preprocessing path; use D024's first 50 warm-up
IDs, then all 200 timed IDs, for five repeats per point. Use one named CPU,
one process and fixed four threads; record CPU model/node, OS, library
versions, torch interop threads and the exact request IDs/order. Rotate point
order between repeats. The primary CPU path encodes frames in chunks of at
most four for **every** per-frame encoder, with an unchunked direct pass as a
secondary measurement. Before timing, require direct/chunked logit agreement
within `1e-4` and no threshold-class change on all timed requests; otherwise
stop and seek a documented methodology decision. F002b receives bilinear
tensors, while TinyViT systems receive bicubic tensors from the same raw IDs.

For each point, the **model-only** clock starts with frame selection from
already decoded/normalized tensors and ends when its probability is ready;
it includes width setting, encoding, pooling and head, and excludes JPEG
decode, preprocessing and policy lookup. Report median and p95 per-request
latency, raw repeats, completed zone-sequence predictions/s and processed
frames/s using timed-pass
wall time. Separately time routing from receipt of a budget and policy ID to
the selected point or DEFER result, without image decoding or model work.
For served requests, also time routing plus model work from that same start
to probability ready; for DEFER, stop when the decision is returned and do
not decode or infer. Grouped-versus-interleaved switching uses the identical
point × request multiset of the **six nested configurations**
`w0.5_k2`, `w0.5_k8`, `w0.75_k8`, `w1.0_k8`, `w1.0_k12` and `w1.0_k16`,
with cache reset between clips. It excludes the two separate static
backbones and DEFER, whose routing time is reported separately.
Run a **separate** practical JPEG decode/resize/normalize/model path on the
same frozen 200 timed IDs and 50 warm-up IDs, five repeats, host and thread
settings, after an untimed read pass. Its clock starts at budget/policy receipt
before JPEG decode and ends at probability ready, or at an early DEFER decision;
label its latency/throughput end-to-end. Deferral overhead and completed
throughput accompany coverage.
Existing F002 DAS-5, F004 Xeon and F006b Ryzen absolute times are context,
not a controlled cross-system comparison.

## 5. Uncertainty and preselected matched comparisons

First aggregate alias probabilities into one record per duplicate-aware unit,
retaining that unit's alias-row costs and frozen split_group. Resample the
**test split_group dependency groups**, not alias rows or the five repeated
states, with replacement for 5,000 paired bootstrap draws using NumPy PCG64
seed `13013`. Sort dependency-group IDs, generate the group-index draws
**once**, and reuse the same indices across every system, state and contrast.
Each draw carries all member unit records, budget states and predictions,
preserving multiplicity when a group is drawn twice. Report point estimates and
2.5/97.5 percentile intervals for coverage, accuracy, macro-F1, both recalls,
AUROC where defined, mean full FLOPs/request, and prespecified paired quality
differences. A draw without
both classes is invalid for class-dependent metrics; report valid-draw counts,
and suppress an interval with fewer than 4,000 valid draws. These intervals
reflect known-group sampling uncertainty on this proxy dataset, not
biological-patient independence or model-seed uncertainty. These are
approximate, unadjusted per-comparison 95% intervals; multiple comparisons
are not familywise corrected, and no p-value determines a winner.

**Near-matched computation (supplementary, chosen before test):** accept at
most a 1% relative difference in exact full FLOPs, measured against A's cost.
Compare A at K16 (5,107,788,033 FLOPs) with the **original frozen F002c
TinyViT checkpoint** at K4 (5,149,017,395 FLOPs), a 0.807% cost difference.
K4 is an additional diagnostic point, not a replacement for primary B at K16.
Use identical units, probability aggregation and paired group-bootstrap
differences at
the common `b=0.25` budget, where both points now fit under the revised cap;
report cost, quality and coverage. This pair meets the proposed 1% matching
tolerance and isolates a near-matched FLOP comparison. It does not by itself
measure the benefit of adaptivity.

**Equal-FLOP training-recipe comparison (secondary, chosen before test):**
compare B's original F002c TinyViT K16 checkpoint with the canonical F005b
nested checkpoint at full width, K16. Both cost exactly 20,596,068,563
FLOPs and fit at `b=1`. Use the same units and report paired accuracy,
macro-F1, both recalls and AUROC where supported, coverage and cost with
dependency-group intervals. This investigates the predictive effect of the
F005b training/distillation procedure at unchanged inference FLOPs. The
training recipes and checkpoints differ, so it does not isolate pruning as a
causal mechanism or add a seventh primary system.

**Adaptive-value contrasts (preselected):** for each informative D021 state
`b = {0.25, 0.50, 0.75, 1.00}` and the secondary heterogeneous schedule,
report paired F-minus-D and F-minus-E differences in coverage, completed-unit
accuracy, macro-F1, abnormal and normal recall, AUROC where supported, and
mean full FLOPs per requested row. Predictive differences and paired
common-request cost use common completed units, with that denominator shown;
coverage and total cost use all assigned requests. Report these jointly with
bootstrap intervals and configurations;
do not select favorable states after seeing test results. At `b=0`, all
systems defer, so report the common zero coverage/cost without a predictive
contrast. No single winner rule is specified.

**Adaptive versus static contrasts (preselected):** compare F with every
static system that serves at the same D021 state:

| State | F comparisons on common completed units |
|---:|---|
| `b=0.25` | F−A |
| `b=0.50` | F−A, F−C |
| `b=0.75` | F−A, F−C |
| `b=1.00` | F−A, F−B, F−C |

In the predefined heterogeneous schedule, compare F with A, B and C on
each pair's common completed units. For every contrast report F-minus-static
accuracy, macro-F1, abnormal recall, normal recall and AUROC differences
where supported, plus the common-unit denominator, both systems' coverage and
their coverage difference,
full FLOPs difference on those common requests and paired dependency-group
bootstrap intervals. Also report total full FLOPs over all assigned requests
beside coverage so deferrals remain visible. The same prediction and cost
rules apply to F−D and F−E. No new winner rule or post-test pair selection
is permitted. F, D and E all select `w0.5_k2` at `b=0.25`; F and E both
select `w1.0_k16` at `b=1.00`. With the same checkpoint, frames and input
path, these pairs have identical predictions by construction. State the
identity and cost directly; do not run redundant statistical tests for it.

**Fixed cost–quality comparison (chosen on validation):** compare A
against D's `w0.5_k2` at `b=0.25`; both pass the exact cap. Validation
macro-F1 is 0.858875 versus 0.869706, with D's exact cost 1,816,575,299
FLOPs. This pair is fixed regardless of its held-out result. Report paired
D-minus-A accuracy, macro-F1, abnormal-recall and normal-recall differences,
the full-work difference/ratio, coverage and dependency-group bootstrap
intervals. Describe the observed differences and uncertainty without an
equivalence or noninferiority claim: no justified margins have been fixed.
A nonsignificant difference does not establish equivalent predictive quality.
Do not search test configurations after seeing the results.

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

A guarded T013b final-test command is implemented but currently denies execution.
Phase A1 parameterized the
F006b controller's exact-rational capacity while retaining historical
`Cmodel` as its default; F007 applies `Ccapacity` to all six systems. The
future command must strictly require F007 JSON
`status == "accepted_final_protocol"` and
`authorizes_test_inference=true`, the accepted committed protocol and frozen
source/checkpoint identities. The proposed JSON currently has
`authorizes_test_inference=false`; changing its status alone is insufficient.
Structural T006 verification may occur before final sign-off: canonical
`verify_t006()` hashes and parses test-manifest
metadata, checks counts and fingerprints against `configs/f002_protocol.json`,
and must fail closed on mismatch. That check does not authorize predictive
evaluation. Before **any test-image decoding for evaluation, model inference,
test-derived predictive metric or raw result**, require completed independent
PR review and Charalampos's recorded final sign-off under the reported Salo
delegation. No test predictions may choose a model or configuration, and no
threshold, budget assignment or protocol rule may be retuned from held-out
evidence. A technical failure may be rerun with the identical frozen
configuration and all attempts retained.

The available identifiers protect only known family/shared-recording
dependencies, not biological patients. One test split and one selected seed
per frozen checkpoint, validation reuse in prior selection, device/label
confounding, simulated FLOP budgets, and absence of edge-device energy or
clinical data limit the conclusions. The team must decide whether it wants a
separate scalar utility for deferral; this protocol makes no such claim.

## 7. T013b Phase A1 validation infrastructure

`experiments/F007/phase_a1.py` implements exact-capacity routing, the frozen
recording-unit schedule, guarded alias-row requests, canonical model dispatch,
and duplicate-aware unit aggregation. Every served primary request starts a
fresh sequence/clip cache; encoded-frame counts are diagnostics while full
FLOPs remain the independent-request accounting. `src/f007_protocol.py` checks the
proposed JSON, canonical Git LF identity of the F002 protocol and validation
profile, T006 manifests via `verify_t006()`, and the frozen model hashes.
The controller's historical default remains `Cmodel`; F007 explicitly passes
`Ccapacity`. This implementation changes no F007 scientific selection rule.

The bounded reproducible validation smoke is:

```powershell
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.phase_a1 smoke-validation --device cpu
```

It creates a new ignored `artifacts/F007/phase_a1_validation_*` directory with
`requests.jsonl`, `units.jsonl`, `summary.json`, and `run.json`; an existing
attempt is never overwritten. The 2026-10-12 UTC smoke used one normal and
one abnormal validation singleton. Across six systems and the `b=1` and
`b=0` diagnostic states, all 12 full-resource requests completed within
their exact budgets and all 12 zero-resource requests deferred. The two-row
predictive figures only verify plumbing; they are not an F007 quality result
or held-out evidence. Synthetic tests cover intermediate states, shared
aliases, missing classes, and unauthorized test attempts. The full official
CPU job and held-out execution remain pending.

The proposed JSON retains `status=proposed_pending_team_approval` and
`authorizes_test_inference=false`. A future final-test run requires both
accepted values, frozen identity checks, D030's committed acceptance and
sign-off, and a separately committed `configs/f007_approval.json` receipt
that references the earlier accepted protocol commit and independent review.
No receipt is provided in Phase A1 or A2. A receipt is kept separate because a
commit cannot embed its own SHA in its committed bytes.

## 8. T013b Phase A2 validation analysis and CPU runner

`validation_run.py` writes complete or explicitly bounded validation-only A1
request/unit evidence for all six systems, the five fixed states, the
heterogeneous schedule and the two prespecified supplementary points.
`analysis.py` rechecks frozen request and checkpoint identities, reconstructs
duplicate-aware units from alias records, generates the shared sorted-group
PCG64 seed-13013 draw matrix once, and uses it for all point and paired
intervals. Drawn groups retain all aliases and repeated states; absent-class
macro-F1/AUROC and recall are unavailable as specified above. Results include
valid-draw counts and suppress intervals below 4,000 valid draws. CSV tables,
an SVG/PNG quality-versus-full-FLOPs figure, raw JSON and the bootstrap indices
go to a new ignored `artifacts/F007/` directory. Identity-by-construction
comparisons report the exact equality without redundant bootstrap tests.
The heterogeneous report gives requested units, alias rows, completed units,
deferred units and coverage for each realized resource state and system.
Factorial coverage and factorial paired coverage differences are exact
structural values without bootstrap intervals; mixed coverage retains its
group-bootstrap interval.

`cache_diagnostic.py` independently runs D/E/F in increasing and decreasing
resource order with a fresh width-and-frame cache per alias/policy/order. It
verifies every cached/direct logit and records incremental FLOPs separately
from full independent-request FLOPs. `benchmark_cpu.py` checks direct/chunk4
logit and class agreement before timing all nine fixed points. It implements
the frozen 200-ID/50-warm-up/five-repeat official protocol, rotating point
order, model-only and JPEG pipeline clocks, policy/deferral clocks and
grouped/interleaved nested switching on one CPU process with four threads.
Outputs include raw repeats, request IDs, software/hardware identity and
checkpoint hashes. Existing F002/F004/F006b times are not matched evidence.

`provenance.py` records the Git commit and dirty state, local and committed
SHA-256 for the execution modules, protocol/checkpoint/input identities,
Python and package versions, OS/CPU/device/thread settings, UTC start time and
exact command parameters in every new Phase A2 run/report. Official validation,
analysis, cache and CPU runs require a clean committed working tree and fail if
source or protocol changes during execution. Diagnostic runs record a dirty
tree explicitly. Each output uses a new directory, preserving prior raw
attempts. Validation-only CPU inference uses one fixed Torch thread and still
executes every alias/system/state as a fresh request; the runner checks exact
prediction identity whenever the frozen checkpoint, configuration and alias
repeat. Full independent-request FLOPs are charged for every served request.

Reproducible commands from the repository root in `bde-project`:

```powershell
$runDir = "artifacts/F007/validation_full_$(Get-Date -Format yyyyMMddTHHmmss)"
$reportDir = "artifacts/F007/report_full_$(Get-Date -Format yyyyMMddTHHmmss)"
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.validation_run --all-validation-units --output $runDir
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.analysis --evidence $runDir --output $reportDir
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.cache_diagnostic --all-validation-aliases
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.benchmark_cpu --official
```

For a bounded plumbing check, use `validation_run --diagnostic-units 2`, then
`analysis ... --diagnostic`, `cache_diagnostic --diagnostic-units 1`, and
`benchmark_cpu --diagnostic-requests 1`. On 2026-10-12 UTC, these
real-validation checks created 76 A1 request records (48 served), 38 report
cells and all 24
prespecified contrasts across five states and the heterogeneous schedule.
Both selected units were normal, so this subset does not estimate two-class
predictive performance. The one-unit cache run served 24 of 30 requests with
zero maximum cached/direct logit difference. The one-request, one-repeat CPU
diagnostic covered all nine points with zero maximum direct/chunk4 logit
difference.
These subset timings are **diagnostic only**. The subsequent complete
validation run and official CPU job were verified locally as follows; both
record clean commit `ffac85119713fc8d8cd4fde53ea56fedd2e44c38` and the
same frozen input/checkpoint identities as the current protocol:

- `artifacts/F007/validation_full_20261013T015903/`: completed, 265 retained
  validation alias rows, 254 duplicate-aware units, all six systems and five
  states plus the heterogeneous schedule; its
  `artifacts/F007/report_full_20261013T015903/` analysis has 38 cells and 24
  preselected contrasts. SHA-256: `run.json`
  `8dcaa47a211c2a224756dd3a58cad11db5c359893a1086afda86fe66e2d0e199`,
  `report.json`
  `b92318ecaf78d3d8e0b9fbee77d0f2e8f9c2c0392d7b7652b96a05a822a1e3cc`.
  The report's recorded source-file hashes match the local raw request, unit
  and run files.
- `artifacts/F007/cpu_validation_20261014T004948183271Z/benchmark.json`:
  official same-job nine-point benchmark on an AMD Ryzen AI 9 HX 370, one
  process/four CPU threads, 200 timed validation requests, 50 warm-ups and
  five repeats. Maximum direct/chunk4 absolute logit difference was `0.0`
  for every point. Its locally verified SHA-256 is
  `223fd2655acb54b6b0ea775d8f220b83d607ec864aaa5a014d9867fdaa0e1de8`.

These are validation and CPU measurements, not held-out predictive evidence.
No held-out test image was decoded or inferred, and the JSON remains proposed
with test inference false.

## 9. Guarded final-test command (not yet authorized)

`final_test.py` calls the committed F007 authorization gate before selecting
test rows for execution, constructing a model predictor, decoding an image,
or producing a prediction. The gate requires both accepted JSON values, the
earlier accepted protocol commit, a committed approval receipt referencing the
independent PR review and D030 sign-off, and matching frozen manifest/profile/
checkpoint hashes. The run also requires committed clean source and records the
approval receipt, protocol commit, source hashes, environment, exact parameters,
request identities and output hashes. The present proposed JSON and absent
receipt make this command fail closed; this section grants no authorization.
After authorization, a no-decode preflight resolves both the MobileNetV2 and
TinyViT K16 frame inventories for every frozen test row before model inference.
The F007-only TinyViT test loader inherits F003's nested positions, bicubic
RGB224/ImageNet preprocessing and `frames_at` behavior; F002c's original
train/validation guard is unchanged.

After independent review and formal recorded approval, the exact command from
the repository root is:

```powershell
$testRunDir = "artifacts/F007/final_test_$(Get-Date -Format yyyyMMddTHHmmss)"
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.final_test --output $testRunDir --device cpu
```

It creates a new ignored attempt directory and refuses an existing one. An
`attempt.json` records success, failure or interruption; denial leaves no
predictions. An
authorized run writes `evidence/requests.jsonl`, `units.jsonl`, `summary.json`
and `run.json`, then uses the same Phase A2 analysis to write `report/` tables,
bootstrap indices and figures. The five factorial states, mixed schedule, all
six systems and both fixed supplementary comparisons use the same canonical
request, aggregation and statistical functions as validation. Each request
retains fresh-cache full-FLOP accounting. Technical failures preserve the new
attempt and any completed raw evidence for investigation; a rerun uses another
directory with the identical frozen configuration. This implementation has
only been exercised with synthetic and validation data; no held-out predictive
result exists.

If complete authorized predictions were written but report generation failed,
recover analysis without another model pass, using a **new** output directory:

```powershell
$recoveryDir = "artifacts/F007/final_report_recovery_$(Get-Date -Format yyyyMMddTHHmmss)"
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.final_test --recover-evidence "$testRunDir/evidence" --output $recoveryDir
```

Recovery repeats the authorization gate, verifies complete raw evidence and
the original committed source/approval provenance, then applies the unchanged
F007 analysis. It does not construct models or decode images.
