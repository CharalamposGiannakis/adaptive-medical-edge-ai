# Open task queue

Course-critical tasks are listed first. Priorities: **P0** = resolve before committing to the final task/scenario; **P1** = core implementation/evaluation; **P2** = extensions after the core baseline. Owners are coordination assignments, not evidence that work has started. Do not infer completion from discussion; update this queue from repository-backed work.

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
- **Scope:** Define target, input unit, labels, and evaluation unit only from the audited dataset and documented project goals.
- **Acceptance criteria:** Prediction task and rationale are documented; task scope is supported by available labels/data and does not overstate ambulance/trauma representativeness.

### T005 — Record Dolly/TA feedback and resolve project assumptions

- **Priority:** P0
- **Owner:** Charalampos
- **Dependencies:** None
- **Status:** Pending; no Dolly/TA feedback is recorded in the repository yet.
- **Acceptance criteria:** Feedback is recorded faithfully in project documentation, including unresolved assumptions; do not fabricate or paraphrase feedback that has not been received.

### T006 — Implement reproducible POCUS loader and patient-level split pipeline

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T002, T004
- **Acceptance criteria:** Reproducible loader and patient-level split generation with documented configuration and leakage checks; no dataset files committed.

### T007 — Specify F002 static POCUS baseline methodology before training

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T002, T004, T005
- **Acceptance criteria:** Documented research question, splits, model/baseline, metrics, configuration, and evaluation protocol approved before training.

### T008 — Implement and run F002 static POCUS baseline

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** T006, T007
- **Acceptance criteria:** Reproducible implementation and run; raw results and configuration saved; results distinguish measurement from interpretation.

### T009 — Characterize candidate adaptive dimensions

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** F002 (T008)
- **Scope:** Consider depth, temporal window length, frame sampling, and other task-appropriate dimensions without presuming a ViT architecture.
- **Acceptance criteria:** Candidate dimensions are compared against the locked task and F002 baseline; proposed measurements and trade-offs are documented.

### T010 — F003: construct at least three shared adaptive operating points

- **Priority:** P1
- **Owner:** Charalampos
- **Dependencies:** To be specified after T009 and the F002 findings.
- **Acceptance criteria:** At least three reproducible operating points share the relevant model/weights as defined by the approved design; configurations and raw measurements are saved.

### T011 — Integrate one compression technique

- **Priority:** P2
- **Owner:** Charalampos
- **Dependencies:** To be specified after core operating points are established.
- **Acceptance criteria:** One selected technique is implemented and evaluated against a documented uncompressed baseline; methodology and results are reproducible.

### T012 — Implement a simple runtime resource-aware policy

- **Priority:** P2
- **Owner:** Charalampos
- **Dependencies:** Scenario, operating points, and resource signals to be specified from preceding tasks.
- **Acceptance criteria:** A simple, documented policy uses an agreed resource signal and is compared with fixed/static alternatives under the same protocol.

### T013 — Final static-versus-adaptive Pareto evaluation

- **Priority:** P2
- **Owner:** Charalampos
- **Dependencies:** T008, T010, T012; final scenario and protocol must be locked first.
- **Acceptance criteria:** Static and adaptive systems are compared on the pre-specified task, splits, predictive metrics, and resource measures; limitations and uncertainty are reported.

## Research backlog (non-course-critical)

- **R001 — Optimization-based controller:** Explore only after the core task, policy, and evaluation are established.
- **R002 — ViT/transformer architecture:** Research possibility, not a current architecture commitment.
- **R003 — Temporal/window-length adaptivity:** Explore as a research dimension after the dataset-supported task and baseline are established.
