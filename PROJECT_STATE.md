# Current Phase

F001, F001b, and F001c are completed pilot feasibility studies. The project scenario is adaptive edge inference for prehospital/ambulance lung POCUS, with OpenPOCUS as a technical experimental proxy accepted in TA Sudaksh's feedback (T005, D020); actual dataset fit and architecture remain unproven. The OpenPOCUS lung dataset is downloaded and labeled locally (T001 completed; PR #16 merged). The T002 audit has passed independent review and is merged (issue #4 closed); T002b found cross-ID shared recordings and provisional split groups (D018). The T004 decision (issue #6, D019) is binary normal versus definite abnormal at the exam × lung-zone sequence level; its deterministic implementation has passed independent review.

## Leading hypothesis

Explore resource-aware inference for prehospital/ambulance lung POCUS using OpenPOCUS only as a technical proxy; it does not establish ambulance/trauma representativeness or clinical effectiveness. Keep D019's binary zone-sequence task. Variable frame count per prediction is now a supported primary adaptive direction: it changes input work even if the network stays fixed. F002 must test whether the dataset's size, bias and sample properties permit useful predictive behaviour; later characterization must test adaptive operating points. ViTs and optimization-based controllers remain research possibilities, not current commitments.

## Feasibility experiments

- **F001:** ImageNet-pretrained DeiT-Tiny on DermaMNIST-224, with shared-backbone operating points at depths 4, 8, and 12. The official test split was not loaded.
- **F001b:** Dense characterization of depths 4–12 using the F001 backbone. The F001 depth-4/8 heads and depth-12 classifier were reused; six intermediate heads were trained with the backbone frozen. Train/validation only; test untouched. Validation accuracy peaks at depth 9 (85.14%), then varies non-monotonically; macro-F1 peaks at depth 10 (0.7061). Measured **fvcore FLOPs** range from 0.3792G (depth 4) to 1.0794G (depth 12).
- **F001c:** Analysis-only validation inference through all existing depth heads (1,003 examples × 9 depths); no weights trained and F001/F001b inputs verified unchanged by SHA-256. At 9→12, 28 errors are rescued and 35 correct predictions regress (net −7). Depth 12 is the first correct depth for 5 samples; the all-depth hindsight oracle covers 92.2% of samples, explicitly as an unreachable upper bound. Confidence/entropy patterns are descriptive and conditional bins are sparse/undefined at confident shallow predictions, so no routing signal is validated.
- F001c source: `experiments/F001c_per_input_depth_utility/`; generated outputs are ignored under `artifacts/F001c_per_input_depth_utility/`.

These DermaMNIST/DeiT experiments remain historical feasibility pilots. They do not lock the POCUS scenario, prediction task, or final architecture.

## Dataset: OpenPOCUS lung ultrasound

- **Source:** Kumar et al. 2026 (`docs/OpenPOCUS_paper_Kumar2026.pdf`), CC BY-NC-SA 4.0. Fetch with `bash data/openpocus/download_openpocus.sh` (~5.3 GB into ignored `data/openpocus/raw/`); labels are committed as `data/openpocus/labels.csv`, rebuilt and checked against the spreadsheet's 94 summary totals by `build_labels.py`.
- **Contents:** 226 exam folders; paper population includes emergency-department respiratory cases and ED/clinic controls, 2020–2022. The local CSV has 114 COVID-positive exams (50.44%). There are 325,731 readable 512×512 RGB JPEGs (5.84 GB raw logical storage). Of 1,871 defined zone labels, 1,862 have frames; 1,871 exam-zone groups have files, including 3 undefined and 6 unlabeled groups. Zone groups are not verified independent source clips.
- **Labels:** per exam per zone, not per frame: normal, B-lines, consolidation, both, indeterminate. Also per exam: COVID status, abnormal flag, device (five device codes observed, 59% Butterfly). **No pneumothorax or trauma labels.**
- **Caveats:** 67.14% of frames inherit normal labels. `Pt146/151/156/157/160 A/B/C` are potential repeat-exam families that must stay together across train/validation/test; biological patient identity remains unverified. The audit found 515 exact duplicate-image groups with conflicting inherited zone labels within ED2/Pt160C, eight frame-count discrepancies, and overlapping Pt45 renditions. FPS/duration are unavailable. No exact duplicates crossed exam folders, but a follow-up search found 10 folder pairs sharing re-encoded recordings (Pt67/Pt81 and Pt147/Pt163 are complete duplicate exams), some with conflicting labels or device entries; with the A/B/C families this gives 212 provisional split groups. This still does not establish patient independence. See `docs/openpocus_dataset_audit.md` for measurements and uncertainties.

## Not yet decided

- Whether later provenance changes T006's known-group/quality assumptions; biological-patient identity remains unverified
- Model input processing, exact frame counts and sampling strategy within each zone sequence; F002/T009 must characterize useful candidate configurations
- Whether OpenPOCUS's size, bias and sample properties support useful prediction/adaptivity; its patients are emergency-department respiratory/COVID cases with no pneumothorax or trauma labels
- Scenario-specific utility/constraints
- Model architecture (a ViT is a research possibility, not a commitment)
- Compression technique
- Runtime adaptivity signal and deployable policy/controller
- Whether to pursue an optimization-based controller (research possibility, not a commitment)
- Target device, latency/energy budgets, final evaluation protocol and throughput unit (frames/s, completed zone-sequence predictions/s, or both; Thursday clarification)
- Whether input-dependent exits outperform a fixed externally selected depth under the chosen scenario
- Pre-specified use of the reserved official test set

## Current objective

T005 is complete: D020 records Sudaksh's feedback and resulting project decisions; D021 records the project's normalized simulated resource-budget convention. The current living technical plan is [`docs/design_pathway.md`](docs/design_pathway.md), subordinate to accepted decisions (D022). T006's common cohort and known-group-disjoint split have been implemented on the task branch and await independent review (D023, issue #8). T007 must pre-specify F002 methodology before model training. Keep F001/F001b/F001c as pilot evidence and the DermaMNIST official test set untouched.

The intended bounded F002 qualification considers MobileNetV2, ResNet-18 and DeiT-Tiny, then selects one architecture before freezing a static baseline. None is selected now. Exact frame counts remain open; variable-frame adaptivity is the primary hypothesis, to be tested at T009.

T006's local snapshot retains **1,762** of 1,765 T004 image-backed candidates (1,225 normal, 537 abnormal) after excluding three aliases of genuinely conflicting shared recordings; missing-label aliases remain T004-excluded without creating semantic conflicts. It omits 98 Pt45 alternate frames without losing a sequence. The corrected seed-42 group allocation gives train/validation/test sequence counts **1,232/265/265** from 211 retained dependency groups; ED4 is the one zero-retained source group. The tracked manifests, diagnostics and reproduction commands are in `data/openpocus/README.md`. This establishes no verified biological-patient independence.

## T005 evaluation and runtime design

- **Evaluation:** Accuracy and macro-F1 remain predictive metrics. Use one consistent FLOPs methodology for the required Pareto evaluation; throughput is a required supporting metric for this variable-frame design. Its exact unit awaits Thursday clarification. No energy, battery or thermal measurements are claimed.
- **Simulated budgets:** Normalized `b ∈ [0, 1]` has active states `0.25`, `0.50`, `0.75`, `1.00`; `b = 0.00` means unavailable / defer / no inference and is not an accuracy/FLOPs operating point. A later policy maps active states to measured frame-count operating points. No frame counts, architecture or mapping are selected yet.

## Decided task: T004

- **Target:** normal (source zone code 0 → target 0) versus definite abnormal (codes 1/2/3 → target 1). Codes 4, 5 and missing labels are explicitly excluded. `abnormal` means B-lines and/or consolidation according to the supplied zone annotation, not trauma, pneumothorax, FAST/free fluid, urgency, critical illness or need for intervention.
- **Prediction unit:** one exam × lung-zone sequence, containing all available ordered frame files. Frames are not independently adjudicated examples. This does not decide a model input/sampling strategy.
- **Reproducible representation:** `data/openpocus/build_binary_task.py` reads immutable `labels.csv` and inventories raw JPEG paths; ignored `artifacts/T004_binary_task/` holds all 2,712 records and a summary. Source-label eligibility is separate from frame availability and later data-quality/split eligibility.
- **Measured candidates:** 1,227 normal + 538 abnormal = **1,765** sequences before quality/split decisions, exactly matching T002. Another nine source-eligible zones lack frames (six normal, three abnormal). Known overlaps, rendition ambiguity and patient-identity uncertainty are not resolved by the binary mapping.
- **Primary classification metrics:** accuracy and macro-F1 at the zone-sequence unit, because the candidate classes are imbalanced. The final evaluation protocol belongs to T007. See D019 and `data/openpocus/README.md` for semantics, limits and reproduction commands.

## Plan

Integrate (T001, PR #16 merged) and complete the independently reviewed audit (T002, issue #4)
→ use the reviewed task (T004) and recorded feedback (T005; D020–D021)
→ specify and run the F002 static baseline (T006–T008)
→ characterize adaptive options and construct shared operating points (T009–T010)
→ complete compression, resource policy and final evaluation (T011–T013)
→ consider research extensions only after core course work
