# Experimental Design Pathway

## Purpose and authority

This file describes the current evidence-driven experimental pathway. It is a living design, not a gospel or permanent architecture specification. Experimental evidence may invalidate or redirect it; update this file when that happens rather than forcing experiments to follow an obsolete plan. If it conflicts with an accepted entry in [`decision_log.md`](decision_log.md), the decision log wins. A recommendation becomes a locked project decision only when explicitly promoted through the normal decision process. Status words below distinguish accepted foundations from hypotheses, decision gates, fallbacks and research backlog.

## LOCKED foundation

- **Dataset and scenario (D010, D020):** OpenPOCUS is the selected public technical experimental proxy for resource-aware prehospital/ambulance lung POCUS. Its ED/clinic data do not establish ambulance or trauma representativeness or clinical deployment effectiveness.
- **Prediction task (D019):** One exam × lung-zone sequence is classified as normal (source code 0) or definite abnormal (codes 1/2/3). Codes 4/5 and missing labels are excluded. `abnormal` means supplied B-lines and/or consolidation annotations, not trauma, pneumothorax, FAST/free fluid, urgency or intervention. Source labels and raw data remain immutable.
- **Known leakage constraints (D018, D019, D023):** Keep each known A/B/C family and cross-ID shared/re-encoded recording connected group in one partition. T006 has frozen the cohort/split from 212 source groups to 211 retained groups. Released identifiers cannot certify biological-patient independence.
- **Order and evaluation (D013, D020):** Establish a reproducible static baseline before the final adaptive system. Variable-frame/input-work adaptivity is a valid direction to investigate, not a proven benefit. Accuracy and macro-F1 are predictive metrics; the assignment's FLOPs Pareto evaluation remains required with one consistent method, and throughput is required supporting evidence for this design. Simulated resource scenarios are acceptable; on-device evaluation is not expected.
- **Resource convention (D021):** Normalized `b ∈ [0, 1]` has active states `0.25`, `0.50`, `0.75`, `1.00`. `b = 0.00` means defer/no inference, never an accuracy or FLOPs model point. No state maps to a frame count yet.

## CURRENT HYPOTHESIS and bounded funnel

`K` denotes the number of frames supplied for one zone-sequence prediction. Variable `K` may offer useful predictive gains for additional input work even if the network itself is fixed. OpenPOCUS size, bias and sample properties could instead make the curve unhelpful. The static proof of concept and T009 measurements decide that question.

The architecture choice is a bounded qualification, not a tournament or a selection of MobileNetV2 by size alone. A higher-capacity model might learn the task better or improve more with additional frames, enough to justify its cost. MobileNetV2 is an edge-oriented lightweight CNN with straightforward implementation and course/lab and quantization compatibility. ResNet-18 is a conventional higher-capacity CNN reference with a possible depth/early-exit fallback. DeiT-Tiny tests whether the project's F001 transformer experience yields value that justifies its added complexity. All three are **candidates**, with no expected winner. D022 accepts the funnel, not an architecture or `K`.

```text
T006 common cohort + known-group-disjoint split
  -> T007 frozen methodology and qualification rules
  -> F002a: 3 backbones at one fixed K -> at most 2 at low/mid/high K
  -> DECISION GATE: select 1 backbone
  -> F002b: frozen static reference
  -> T009: dense/targeted K characterization
  -> DECISION GATE: useful K frontier?
       yes: variable-frame family
       no: cheapest adequate K + one bounded fallback axis
  -> T010 shared operating points -> T011 one compression method
  -> T012 simple resource-budget lookup policy -> T013 final comparison
```

`F002a` and `F002b` are conceptual phases of F002 under T008, not completed experiments, new issue IDs or separately allocated experiment IDs. T008's implementation organization is decided when claimed. Its execution follows the T007-approved method.

## T006 DECISION GATE: common cohort and split

**Purpose:** Produce one reproducible cohort and one known-family/shared-recording-group-disjoint development/test foundation, consumed unchanged by every architecture candidate. Quality and exclusion decisions precede model results. No candidate gets a different split, and the split is not tuned after performance is seen.

**Required T006 outputs and exit checks:**

1. Document derived cohort eligibility, explicit exclusion/reason codes, and an attrition table from T004 image-backed candidates to the retained cohort. Decide Pt45 rendition handling and known duplicate-associated conflicting eligibility/labels without changing source labels or raw files.
2. Build connected groups for the known A/B/C families and every known cross-ID shared/re-encoded recording set; preserve their provenance and check that no known linked group crosses partitions.
3. Produce deterministic train/validation/test or equivalently approved manifests with the split seed/configuration and source/manifest fingerprints or hashes.
4. Report class and group counts, device distribution and retained sequence-length distribution by partition; run leakage checks and reproducibility assertions.
5. State the residual limit: this protects against **known** family/shared-recording dependence but cannot certify biological-patient independence from the released identifiers. Keep raw data outside Git.

T006 does not select or favor an architecture, and its outcomes are not conditioned on model results. It must finish before T007 finalizes data-dependent choices and before F002 candidate training.

## T007 DECISION GATE: pre-specify methodology

T007 is specification only, before any candidate trains on the final T006 split. It fixes the following for the common F002a screen and F002b reference:

The concrete proposed contract is [`f002_methodology.md`](f002_methodology.md) with machine values in `configs/f002_protocol.json` (D024, pending independent review). This living pathway does not override that reviewed contract.

- **Data and model protocol:** T006 manifests; sequence prediction unit and target; one fixed initial `K`; deterministic frame sampler; input resolution and preprocessing; pretrained-weight identifiers; a comparable sequence-aggregation concept where technically reasonable; binary sequence head; loss, class weighting if any, optimizer/schedule and one bounded comparable training/fine-tuning budget; seeds and hardware/software record.
- **Selection protocol:** Checkpoint-selection metric, tie-break, classification-threshold policy, partition use and held-out-test protection; pre-specified learnability/validity criterion; dominance/reduction rule; low/mid/high `K` choices made only after T006 sequence-length statistics; final one-backbone decision rule. Exploratory candidates are not final static baselines.
- **Measurement protocol:** Zone-sequence accuracy and macro-F1, abnormal recall and normal recall/specificity where useful; FLOPs convention and operator-coverage checks; model-only latency and feasible practical pipeline latency; completed zone-sequence predictions/s and processed frames/s; parameter/artifact size where useful; justified device/source-label diagnostic slices. Record hardware/software conditions. The teaching team's preferred throughput presentation remains a non-blocking Thursday clarification.
- **Frozen baseline protocol:** What F002b will hold fixed: architecture, preprocessing, static `K`, sampler, training recipe, checkpoint rule and evaluation protocol. Later adaptive results must not silently replace this reference.

This pathway document itself does not set these values. T006 has fixed the split; T007 proposes exact K, threshold and training values in the linked protocol. No backbone winner or final adaptive operating point has been selected.

## F002a: bounded architecture qualification and backbone gate

Screen at most **MobileNetV2, ResNet-18 and DeiT-Tiny** at a common fixed `K`, on the same T006 partitions, unit and target, with the same evaluation and cost methodology and comparable training budget. Keep sequence aggregation comparable where technically reasonable, and record any unavoidable differences. Eliminate clearly weak/dominated candidates; retain at most two for a small low/mid/high `K` response screen under comparable sampling and evaluation. Select **one** backbone for F002b and record why it was preferred and why the others were discarded.

Use predictive usefulness (accuracy, macro-F1, abnormal and normal recall), FLOPs, latency, both defined throughput quantities, size, compatibility with one shared model artifact and later compression, implementation burden and remaining schedule. Use evidence and Pareto/dominance reasoning, without a fabricated weighted score or raw-accuracy-only rule. Slightly lower static accuracy may be acceptable for a better adaptive frontier; a larger model may win if measured benefit justifies cost.

Stop at three initial backbones, two `K`-screen candidates and one final backbone. Do not broaden search, tune each indefinitely or carry all three through T009–T013. A fourth candidate requires all three to fail T007's pre-specified learnability/validity criterion and an explicit team-approved scope change.

## F002b: frozen static reference

T008 trains/evaluates the selected backbone under T007's frozen protocol and saves an immutable reference artifact, configuration, raw results and measurement conditions for downstream comparison. The F002a exploratory checkpoints do not become the final static baseline merely because they exist. No F002 model, result or final choice exists at the time of this document.

## T009 DECISION GATE: frame-count response

After F002b, characterize evidence-driven candidate `K` values densely or where the response changes. They are **not** the four controller budget percentages. Ask whether additional frames produce a useful accuracy–compute frontier, including throughput and latency effects under the fixed measurement convention.

- **Useful response:** Continue variable `K` as the primary adaptive axis and pass measured candidates to T010.
- **Collapsed/unhelpful response — FALLBACK:** Do not manufacture four frame-count operating points. Retain the cheapest adequate `K` and investigate **one** bounded alternative: preferably network depth/early exits, otherwise spatial resolution. Do not jump to a full elastic supernet or learned/neural controller. Record the evidence and update this pathway and the decision log before changing the core direction.

## T010–T013: course-critical adaptive core

- **T010:** Construct the required at-least-three shared-model adaptive operating points from the useful measured axis; keep configurations and raw results. The controller states do not dictate how many candidate configurations must be measured. If evidence supports too few useful points, revisit the design instead of inventing them.
- **T011:** Integrate **one** compression technique and compare with an uncompressed reference. INT8 post-training quantization is a likely first candidate; structured pruning is a fallback. The method is not selected until model/backend evidence supports it.
- **T012:** Build a simple lookup policy mapping active normalized resource states to useful measured operating points; `b = 0.00` defers inference. Compare against fixed/static alternatives. Optimization-based or learned controllers remain research extensions.
- **T013:** Compare the frozen static reference and adaptive/policy/compressed system on the pre-specified task and partitions, with accuracy, macro-F1, consistent FLOPs, throughput and relevant latency. Report limitations and uncertainty. T011, T012 and T013 are course-critical, not optional research work.

## Decision gates and fallback rules

| Gate | Evidence required | Primary path | Fallback / response |
|---|---|---|---|
| T006 cohort | Eligibility, groups, leakage and attrition | Freeze one shared data foundation | Resolve quality/grouping before model results |
| F002a initial screen | Common fixed-`K` predictive/cost/runtime evidence | Retain at most two | Fourth candidate only after all fail pre-specified validity and team approves scope change |
| F002a `K` screen | Small low/mid/high response for retained candidates | Select one backbone for F002b | Record limits; do not select by accuracy alone |
| T009 frame-count curve | Dense/targeted predictive and cost response | Variable `K` family | Cheapest adequate `K` plus one bounded depth/exit or resolution axis |
| T011 compression | Measured backend compatibility and cost/quality change | One suitable method | Structured pruning if INT8 PTQ is unsuitable; document choice |

## Deliberately undecided and scope control

No winning architecture, final adaptive operating points, compression method, resource-to-`K` mapping, learned controller or ViT final architecture has been selected. T006 fixes the cohort and split; T007 proposes exact initial/screen K in its preregistration. The core stays bounded to three initial backbones, at most two in the small `K` screen, one final architecture, one primary adaptive axis, one compression method and one simple policy. Decisions follow measured evidence and accepted decision-log entries.

**RESEARCH BACKLOG:** Marginal-benefit routing, learned frame selection, attention pooling, frame × depth combinations, queue-aware optimization, learned controllers and broad transformer searches wait until the course-critical core is established. DeiT-Tiny's bounded qualification is not a commitment to broad ViT work.
