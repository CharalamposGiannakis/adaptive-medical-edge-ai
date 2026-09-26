# Decision Log

This document records project-level decisions that affect the scientific
methodology, implementation direction, evaluation, or team workflow.

Decisions marked **Provisional** may be revisited when new evidence becomes
available. Historical decisions are kept even if later superseded.

---

### D001 — Build the project around a safe core and an optional research track

**Date:** 2026-09-22  
**Status:** Accepted

- **Decision**: Develop the project in two layers:

1. **Core course project** — designed to satisfy the assignment robustly and
   target a strong grade.
2. **Research extensions** — pursued only after the core methodology is secure
   and only when experimental evidence justifies the additional work.

- **Rationale**: The assignment rewards system design, correct adaptive behaviour, shared/nested weights, Pareto evaluation, and presentation in addition to novelty. Research ambition should not make the grade-critical implementation fragile.

- **Consequence**: Research ideas must not block completion of the required adaptive system, compression method, operating points, static baseline, and Pareto analysis.

---

### D002 — Repository is the shared project source of truth

**Date:** 2026-09-22  
**Status:** Accepted

- **Decision**: Important project state, assumptions, experiments, decisions, and tasks must be represented in the repository rather than existing only in conversations.

- **Rationale**: The project involves multiple people and multiple tools. Persistent documentation is necessary for reproducibility and synchronization.

- **Consequence**: The repository contains project-state, experiment, task, research, and presentation documentation from the beginning of the project.

---

### D003 — Do not lock the final scenario before a feasibility study

**Date:** 2026-09-22  
**Status:** Accepted

- **Decision** : Run a small technical pilot before selecting the final medical scenario, dataset, and model architecture.

- **Rationale**: We wanted evidence that adaptive shared-weight inference could produce useful accuracy–compute operating points before investing in a complete scenario.

- **Consequence**: Experiments F001–F001c are treated as **pilot/feasibility studies**, not as the final project model.

---

### D004 — F001 uses DermaMNIST-224 and pretrained DeiT-Tiny only as a pilot

**Date:** 2026-09-22  
**Status:** Accepted

- **Decision**: Use the following for the initial feasibility experiment:

  - DermaMNIST-224
  - ImageNet-pretrained DeiT-Tiny
  - shared transformer backbone
  - intermediate exits

- **Rationale**: The combination is small enough for rapid experimentation while still testing the mechanism of interest: shared-weight inference at different compute depths.
- **Consequence**: The DermaMNIST-trained weights are not assumed to transfer to the eventual medical scenario. The experimental methodology and code may transfer even if the final dataset/model changes.

---

### D005 — Characterize the configuration space before selecting operating points

**Date:** 2026-09-24  
**Status:** Accepted

- **Decision**: Extend the sparse F001 depths 4/8/12 into the F001b dense sweep over depths 4–12.
- **Rationale**: Connecting three measured points does not establish the shape of the accuracy–compute relationship. Operating points should be chosen based on measurements rather than aesthetically convenient depths.
- **Evidence**: F001b showed strong gains at earlier depths followed by diminishing and non-monotonic returns. Validation accuracy peaked at depth 9 rather than at the full depth, demonstrating why the dense characterization was useful.
- **Consequence**: Future adaptive configurations should be selected from measured trade-offs rather than assuming that deeper inference is always better.

---

### D006 — Aggregate accuracy is insufficient for judging adaptive depth

**Date:** 2026-09-24  
**Status:** Accepted

- **Decision**: Run F001c to evaluate predictions at the individual-input level rather than judging deeper exits only by aggregate accuracy.
- **Rationale**: A deeper exit can correct some shallow errors while simultaneously introducing new errors. Aggregate accuracy hides this complementary behaviour.
- **Evidence**: For depth 9 versus depth 12, depth 12 rescued 28 depth-9 errors but regressed 35 previously correct predictions. The all-depth hindsight oracle correctly covered 92.2% of validation samples, showing theoretical headroom for input-dependent routing.
- **Consequence**: Input difficulty/uncertainty remains a potentially useful adaptive signal, but no deployable routing method has yet been validated.

---

### D007 — Do not declare an optimal model depth from the pilot experiments

**Date:** 2026-09-24  
**Status:** Accepted

- **Decision**: F001–F001c may identify promising operating regions but must not define an “optimal” depth.
- **Rationale**: Optimality depends on the eventual deployment scenario, resource constraints, latency, energy, predictive requirements, and adaptive policy.
- **Consequence**: Depths such as 6, 7, 8, or 9 remain candidate operating points rather than final architectural choices.

---

### D008 — Move the leading scenario toward prehospital / ambulance POCUS

**Date:** 2026-09-24  
**Status:** Provisional

- **Decision**: Use **prehospital point-of-care ultrasound (POCUS)** in ambulance / field settings as the leading deployment scenario. Retinal screening remains a possible fallback.
- **Rationale**: The ambulance environment gives a direct motivation for edge inference: rapid response requirements; limited portable compute; battery constraints; potentially unreliable or unavailable network connectivity; continuous sensory/video data; and realistic need for local inference. The scenario is also immediately understandable for a presentation and aligns well with the course's emphasis on sensory data and changing edge conditions.
- **Consequence**: Dataset and model selection should now be evaluated primarily against this deployment story.

---

### D009 — Do not commit yet to ViT or the optimization controller

**Date:** 2026-09-24  
**Status:** Accepted

- **Decision**: Keep both of the following as research possibilities rather than architecture requirements: Vision Transformer / transformer-based model; optimization-based or learned runtime controller.
- **Rationale**: The strongest contribution may come from the complete adaptive edge case study rather than from adding a complicated neural architecture or optimization formulation. The selected POCUS dataset and task should determine the appropriate model.
- **Consequence**: Architecture follows the data and problem, not the CV goal or an earlier prototype choice.

---

### D010 — Public POCUS data may be used as a technical proxy

**Date:** 2026-09-24  
**Status:** Provisional

- **Decision**: If a sufficiently large trauma-specific ultrasound dataset cannot be obtained, use an appropriate public POCUS dataset while clearly documenting that it is a proxy for the target ambulance-trauma deployment.
- **Rationale**: The course project is primarily about adaptive edge inference rather than clinical validation. The project timeline is too short to make access to restricted medical data a critical dependency.
- **Limitation**: The model must only claim to predict labels that actually exist in the chosen dataset. For example, a model trained on general lung-ultrasound findings must not be presented as a validated trauma pneumothorax detector unless the data support that claim.
- **Consequence**: Conclusions will concern **adaptive inference methodology**, not clinical effectiveness in ambulance trauma patients.

---

### D011 — Restricted clinical datasets must not block the project

**Date:** 2026-09-24  
**Status:** Accepted

- **Decision**: Request access to relevant trauma datasets, including human FAST and pneumothorax data, but wait only briefly for access before proceeding with public data.
- **Rationale**: The remaining project duration is less than three weeks.
- **Consequence**: Restricted datasets may later supplement or strengthen the experiments, but the core implementation must be executable using accessible data.

---

### D012 — Audit the dataset before defining F002

**Date:** 2026-09-25  
**Status:** Accepted

- **Decision**: Before selecting the final prediction task or model, inspect the selected POCUS dataset in detail. The audit must determine at least:
  - available labels;
  - number of patients;
  - number of clips/images;
  - class balance;
  - patient identifiers;
  - file format;
  - video/frame characteristics;
  - licensing;
  - storage requirements;
  - whether patient-level splitting is possible.
- **Rationale**: The prediction task and architecture must be based on what the dataset actually contains.
- **Consequence**: No final POCUS model architecture is selected until the dataset audit is complete.

---

### D013 — Establish a static POCUS baseline before adaptivity

**Date:** 2026-09-25  
**Status:** Accepted

- **Decision**: The first experiment after dataset/task lock will be a reproducible **static POCUS baseline (F002)**.
- **Rationale**: The adaptive system needs a fixed baseline against which its benefits can be measured. The assignment also explicitly recommends fixing the static baseline before developing the adaptive system.
- **Consequence**: F002 will establish the data pipeline, split, predictive metrics, FLOPs and latency methodology before introducing adaptive behaviour.

---

### D014 — Temporal information is a legitimate candidate adaptive dimension

**Date:** 2026-09-25  
**Status:** Provisional / Research candidate

- **Decision**: For ultrasound video, investigate whether computation can adapt not only through model depth but also through temporal processing, for example:
  - number of frames;
  - window duration;
  - frame sampling rate.
- **Rationale**: Ultrasound naturally produces temporal sensory data. A fixed temporal window may spend unnecessary computation on easy cases while difficult cases may benefit from more temporal information.
- **Consequence**: Temporal/window adaptivity remains in the research backlog until F002 shows what the dataset and baseline model require.

---

### D015 — GitHub Issues are the operational task queue

**Date:** 2026-09-25  
**Status:** Accepted

- **Decision**: Use GitHub Issues as the live collaboration queue between Charalampos and Salom. Issue states follow the lightweight workflow: `[READY]`, `[IN PROGRESS]`, `[REVIEW]`, and `[BLOCKED]`; close issues when complete.
- **Rationale**: This workflow worked well in the Evolutionary Computing project and gives both team members a clear view of what can be worked on immediately.
- **Consequence**: `docs/open_tasks.md` remains a high-level roadmap rather than duplicating live issue status. Work is performed through branches and pull requests where appropriate.

---

### D016 — First collaborative task is the existing POCUS download pipeline

**Date:** 2026-09-25  
**Status:** Accepted

- **Decision**: T001 is to integrate Salom's existing POCUS dataset download script rather than implementing another downloader independently.
- **Rationale**: The work already exists locally and should become the first task moving through the shared issue → branch → PR → review workflow.
- **Consequence**: The dataset itself remains outside Git. Once T001 is merged, T002 (dataset audit) becomes ready.

---

### D017 — Version `data/openpocus/labels.csv` despite being generated

**Date:** 2026-09-25  
**Status:** Accepted

- **Decision**: Commit `data/openpocus/labels.csv` to Git, although `build_labels.py` generates it from the OpenPOCUS metadata spreadsheet. This is an intentional exception to the rule that generated artifacts stay untracked.
- **Rationale**: The file is small (about 9 KB), human-readable metadata rather than the dataset itself. Keeping it in Git lets anyone audit the label mapping, and review changes to it in diffs, without downloading the full OpenPOCUS dataset.
- **Consequence**: `build_labels.py` remains the reproducible source. If the labeling logic changes, rerun it and commit the regenerated `labels.csv` together with the script change. The raw dataset and temporary download files stay untracked.

---

### D018 — Define the OpenPOCUS binary zone-sequence task

**Date:** 2026-09-26

**Status:** Accepted task definition; implementation subject to independent PR review (T004, issue #6)

- **Decision**: Classify one **exam × lung-zone sequence** as normal or definite abnormal. A sequence contains all available ordered frame files for one exam folder and zone; the prediction unit is not an individual JPEG. Source zone code 0 maps to target 0 (`normal`); codes 1/2/3 map to target 1 (`abnormal`). Code 4 (indeterminate), code 5 (undefined) and blank/missing labels are explicitly excluded. The source `labels.csv` and raw files remain immutable during task derivation.
- **Meaning**: `abnormal` means **B-lines and/or consolidation according to the supplied OpenPOCUS zone-level annotation**. It does not mean trauma, pneumothorax, FAST/free fluid, urgency, critical illness or need for intervention. The supplied exam-level abnormal flag is a different target and is not used for this mapping. Frames are not independently adjudicated ground-truth examples.
- **Rationale**: This user-directed task follows existing zone annotations and combines the sparse definite finding classes without adding clinical labels. It supports investigation of a binary lung-finding task; it does not establish the dataset's suitability for the provisional prehospital/ambulance scenario.
- **Primary classification metrics**: Accuracy and macro-F1, at the zone-sequence unit. The 1,227 normal versus 538 abnormal image-backed candidates are imbalanced, so accuracy alone is insufficient. Final F002 evaluation methodology remains T007 work.
- **Measured reconciliation**: The local snapshot yields 2,712 exam × zone records. Of 1,774 source-label-eligible records (1,233 normal, 541 abnormal), nine lack frames. The remaining **1,765 candidates (1,227 normal, 538 abnormal)** agree with T002 before quality/split exclusions. There are 97 indeterminate, 3 undefined-code and 838 unavailable-label records. These are snapshot measurements, not hardcoded eligibility totals.
- **Implementation**: `data/openpocus/build_binary_task.py` produces deterministic, ignored `artifacts/T004_binary_task/zones.jsonl` and `summary.json`. Every source slot is retained with original exam/zone IDs, label, mapped target, exclusion reason and ordered JPEG paths/count. Source-label eligibility is separate from frame availability and later quality/split eligibility; missing frames do not change a target. Details and runnable checks are in `data/openpocus/README.md`.
- **Unresolved quality constraints**: ED2/Pt160C inherited-label conflicts, Pt45 alternate renditions, A/B/C repeat-exam structure and biological-patient identity are not repaired. The reported re-encoded cross-ID recordings Pt67/Pt81 and Pt147/Pt163 are a separate follow-up before the final split. Preserved identifiers allow T006 to incorporate that evidence; this mapping neither merges patients nor establishes independent observations.
- **Consequence**: Target semantics and prediction unit are available for T006/T007. No partition, patient/group assignment, model architecture, frame count/sampling strategy, F002 specification/training, adaptive mechanism or compression technique is selected here. The final scenario and dataset fit remain open.

---

# Open Decisions

The following are intentionally **not decided yet**:

- Suitability of OpenPOCUS as a technical proxy for the final deployment scenario; T004 defines its annotation-derived prediction task only.
- Final data-quality exclusions, patient/group mapping and split protocol, including the separate cross-ID duplicate follow-up before splitting.
- Model input processing and frame counts/sampling within the decided exam × zone prediction unit.
- Final backbone architecture (CNN, ViT, video/temporal architecture, etc.).
- Primary adaptive dimension.
- Final set of operating points.
- Compression technique.
- Whether resource conditions are simulated, measured, or both.
- Whether input uncertainty becomes part of the runtime policy.
- Whether an optimization-based controller is worth implementing.
- Final scenario-specific utility / global performance measure.
- Exact latency, battery, thermal, and network assumptions.
- Final held-out test protocol.
- Whether restricted FAST/pneumothorax data become available.
- Dolly/TA feedback that materially changes the design.
