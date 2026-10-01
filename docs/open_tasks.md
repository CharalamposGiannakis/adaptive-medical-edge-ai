# Open task queue

Course-critical tasks are listed first. Priorities: **P0** = resolve before committing to the final task/scenario; **P1** = core implementation/evaluation, including compression, policy and final comparison; **P2** = research extensions after the core. Owners are coordination assignments, not evidence that work has started. Do not infer completion from discussion; update this queue from repository-backed work. The conditional technical sequence is in [`design_pathway.md`](design_pathway.md).

## Course-critical tasks

### T001 — Integrate Salome's POCUS download script through a PR

- **Priority:** P0
- **Owner:** Salome
- **Dependencies:** None
- **Scope:** Integrate Salome's existing script; do not reimplement it here before reviewing her PR.
- **Acceptance criteria:** Reproducible download command/script; downloaded dataset is gitignored; no credentials or machine-specific paths; short usage documentation.

### T002 — Audit downloaded POCUS dataset

- **Priority:** P0
- **Owner:** Charalampos / Salome
- **Dependencies:** T001
- **Audit evidence:** Independent review and full rerun passed 2026-09-26; see `docs/openpocus_dataset_audit.md` and `data/openpocus/audit_openpocus.py`. T002b later documented shared cross-ID recordings and provisional split groups (D018). Patient identity and source metadata limitations remain; task definition follows in T004 (D019), while F002 remains unspecified.
- **Scope:** Determine labels, patients, clip count, class distribution, video/image format, available resolutions/frame rates, patient identifiers, licensing, disk size, and feasibility of patient-level splits.
- **Acceptance criteria:** Repository-backed audit records each available property, source/license, missing metadata, and whether leakage-safe patient-level splitting is possible.

### T003 — Reflect provisional prehospital/ambulance POCUS scenario in the repository

- **Priority:** P0
- **Owner:** Charalampos
- **Dependencies:** None
- **Scope:** Make the scenario visible while explicitly labeling it provisional and distinguishing the public dataset as a technical proxy where appropriate.
- **Acceptance criteria:** Relevant project-facing documentation does not present the scenario, dataset fit, or clinical population as finalized.

### T004 — Lock the exact prediction task supported by the selected public dataset

- **Priority:** P0
- **Owner:** Charalampos
- **Dependencies:** T002
- **Repository evidence:** D019 defines binary normal versus definite abnormal at the exam × lung-zone sequence level, with accuracy and macro-F1 as primary classification metrics. `data/openpocus/build_binary_task.py` implements the mapping and explicit exclusions; tests reconcile 1,227 normal + 538 abnormal candidates with T002. See `data/openpocus/README.md` for reproduction and limitations; GitHub issue #6 carries completion status.
- **Scope:** Define target, input unit, labels, and evaluation unit only from the audited dataset and documented project goals.
- **Acceptance criteria:** Prediction task and rationale are documented; task scope is supported by available labels/data and does not overstate ambulance/trauma representativeness.

### T005 — Record teaching-team feedback and resolve project assumptions

- **Priority:** P0
- **Owner:** Charalampos
- **Dependencies:** None
- **Status:** Completed (GitHub issue #7). D020 records TA Sudaksh's feedback separately from project decisions; D021 records the normalized simulated resource-budget convention. `PROJECT_STATE.md` summarizes the current design. Thursday's non-blocking clarification is whether throughput means frames/s, completed zone-sequence predictions/s, or both.
- **Acceptance criteria:** Feedback is recorded faithfully in project documentation, including unresolved assumptions; do not fabricate or paraphrase feedback that has not been received.

### T006 — Build reproducible POCUS cohort and known-group-disjoint split pipeline

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T002, T004 (complete)
- **Status:** Implemented on the T006 task branch; awaiting independent review in GitHub issue #8. D023 records the quality/split policy. `data/openpocus/t006/` holds the small tracked shared foundation; `data/openpocus/README.md` gives reproduction and measured diagnostics.
- **Scope:** Decide quality/exclusion rules before model results, including Pt45 and duplicate-associated label conflicts. Produce one common cohort and known-family/shared-recording-group-disjoint partition foundation for every architecture; no architecture-specific split. Biological-patient independence remains unverified.
- **Acceptance criteria:** Deterministic manifests/config and source hashes; attrition and class/device/group/sequence-length summaries; connected-group and leakage assertions; immutable source labels/raw data and no dataset files committed. See the T006 gate in `design_pathway.md`.

### T007 — Pre-specify F002 architecture qualification and frozen static baseline

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T002, T004, T005 (complete); planning can run alongside T006, but final data-dependent protocol requires T006 manifests.
- **Scope:** Specify a common fixed-`K` screen for MobileNetV2, ResNet-18 and DeiT-Tiny; an at-most-two low/mid/high `K` screen; a one-backbone decision rule; and the frozen F002b reference protocol. T007 does not train models.
- **Acceptance criteria:** Pre-specified data use, sampler/preprocessing, comparable bounded training, checkpoint/threshold selection, predictive metrics, consistent FLOPs with operator checks, latency, both throughput quantities, held-out-test protection and hardware/software record. Exact `K` follows T006 sequence-length evidence; see `design_pathway.md`.

### T008 — Implement and run F002 static POCUS baseline

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T006, T007
- **Acceptance criteria:** Execute T007's approved F002 methodology: conceptual F002a bounded qualification selects one backbone, then F002b saves the frozen static reference. Reproducible configurations, raw results and measurements distinguish evidence from interpretation; exploratory candidate checkpoints are not the final baseline.

### T009 — Characterize candidate adaptive dimensions

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** F002 (T008)
- **Scope:** Densely or selectively characterize frame count `K` on the selected F002 model as the primary adaptive hypothesis, without mapping resource percentages to `K`.
- **Acceptance criteria:** Measured predictive/cost response decides whether variable `K` yields useful operating points. If it does not, retain the cheapest adequate `K` and examine one bounded fallback axis, preferably depth/early exits or spatial resolution; do not manufacture four frame-count points.

### T010 — F003: construct at least three shared adaptive operating points

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** To be specified after T009 and the F002 findings.
- **Acceptance criteria:** At least three reproducible operating points share the relevant model/weights as defined by the approved design; configurations and raw measurements are saved.

### T011 — Integrate one compression technique

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** To be specified after core operating points are established.
- **Acceptance criteria:** One evidence/backend-dependent technique is implemented and evaluated against a documented uncompressed baseline; methodology and results are reproducible. INT8 PTQ is a candidate, structured pruning a fallback, with neither selected yet.

### T012 — Implement a simple runtime resource-aware policy

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** Scenario, operating points, and resource signals to be specified from preceding tasks.
- **Acceptance criteria:** A simple lookup policy maps active normalized budget states to measured operating points, defers at `b = 0.00`, and is compared with fixed/static alternatives under the same protocol. Optimization-based or learned control remains research work.

### T013 — Final static-versus-adaptive Pareto evaluation

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T008, T010, T011, T012; final scenario and protocol must be locked first.
- **Acceptance criteria:** Static and adaptive/compressed systems are compared on the pre-specified task, splits, predictive metrics, FLOPs and throughput; limitations and uncertainty are reported.

## Research backlog (non-course-critical)

- **R001 — Optimization-based controller:** Explore only after the core task, policy, and evaluation are established.
- **R002 — Broad ViT/transformer exploration:** Research possibility, not a current architecture commitment. DeiT-Tiny's bounded F002a candidacy does not select a transformer.
- **R003 — Other temporal/window variants:** Explore beyond the primary variable-frame direction after the dataset-supported task and baseline are established.
