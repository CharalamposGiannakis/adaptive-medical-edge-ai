# Current Phase

F001, F001b, and F001c are completed pilot feasibility studies. The leading scenario is provisionally prehospital/ambulance POCUS; dataset fit and architecture remain open. The OpenPOCUS lung dataset is downloaded and labeled locally (T001 completed; PR #16 merged). The T002 audit has passed independent review and is merged (issue #4 closed). The T004 decision (issue #6, D018) is binary normal versus definite abnormal at the exam × lung-zone sequence level; its deterministic implementation is prepared for independent PR review.

## Leading hypothesis

Explore resource-aware inference for a provisional prehospital/ambulance POCUS scenario. OpenPOCUS may serve as a technical proxy and does not establish ambulance/trauma representativeness. The annotation-derived task is now defined; scenario fit and architecture remain open. ViTs and optimization-based controllers are research possibilities, not current commitments.

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
- **Caveats:** 67.14% of frames inherit normal labels. `Pt146/151/156/157/160 A/B/C` are potential repeat-exam families that must stay together across train/validation/test; biological patient identity remains unverified. The audit found 515 exact duplicate-image groups with conflicting inherited zone labels within ED2/Pt160C, eight frame-count discrepancies, and overlapping Pt45 renditions. FPS/duration are unavailable. No exact duplicates crossed exam folders, but this does not establish patient independence. See `docs/openpocus_dataset_audit.md` for measurements and uncertainties.

## Not yet decided

- Final data-quality exclusions, patient/group mapping and train/validation/test split protocol (T006), including the separate re-encoded cross-ID duplicate follow-up for Pt67/Pt81 and Pt147/Pt163 before the final split
- Model input processing, frame counts and sampling strategy within each zone sequence
- Whether and how OpenPOCUS can serve as a technical proxy for prehospital/ambulance use; its patients are emergency-department respiratory/COVID cases with no pneumothorax or trauma labels
- Final medical scenario and scenario-specific utility/constraints
- Model architecture (a ViT is a research possibility, not a commitment)
- Compression technique
- Runtime adaptivity signal and deployable policy/controller
- Whether to pursue an optimization-based controller (research possibility, not a commitment)
- Target device, latency/energy budgets, and evaluation protocol
- Whether input-dependent exits outperform a fixed externally selected depth under the chosen scenario
- Pre-specified use of the reserved official test set

## Current objective

Review the T004 implementation and record Dolly/TA feedback (T005; pending). T006 can use the explicit exam/zone IDs and derived labels after the duplicate follow-up informs group/exclusion rules; no split protocol or F002 methodology has been selected. Keep F001/F001b/F001c as pilot evidence and the DermaMNIST official test set untouched.

## Decided task: T004

- **Target:** normal (source zone code 0 → target 0) versus definite abnormal (codes 1/2/3 → target 1). Codes 4, 5 and missing labels are explicitly excluded. `abnormal` means B-lines and/or consolidation according to the supplied zone annotation, not trauma, pneumothorax, FAST/free fluid, urgency, critical illness or need for intervention.
- **Prediction unit:** one exam × lung-zone sequence, containing all available ordered frame files. Frames are not independently adjudicated examples. This does not decide a model input/sampling strategy.
- **Reproducible representation:** `data/openpocus/build_binary_task.py` reads immutable `labels.csv` and inventories raw JPEG paths; ignored `artifacts/T004_binary_task/` holds all 2,712 records and a summary. Source-label eligibility is separate from frame availability and later data-quality/split eligibility.
- **Measured candidates:** 1,227 normal + 538 abnormal = **1,765** sequences before quality/split decisions, exactly matching T002. Another nine source-eligible zones lack frames (six normal, three abnormal). Known overlaps, rendition ambiguity and patient-identity uncertainty are not resolved by the binary mapping.
- **Primary classification metrics:** accuracy and macro-F1 at the zone-sequence unit, because the candidate classes are imbalanced. The final evaluation protocol belongs to T007. See D018 and `data/openpocus/README.md` for semantics, limits and reproduction commands.

## Plan

Integrate (T001, PR #16 merged) and complete the independently reviewed audit (T002, issue #4)
→ review the implemented task (T004) and record feedback (T005; pending)
→ specify and run the F002 static baseline (T006–T008)
→ characterize adaptive options and construct shared operating points (T009–T010)
→ consider extensions only after core course work
