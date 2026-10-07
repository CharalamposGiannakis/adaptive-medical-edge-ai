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

### D018 — Version `data/openpocus/split_groups.csv` despite being generated

**Date:** 2026-09-28

**Status:** Accepted

- **Decision**: Commit `data/openpocus/split_groups.csv`, which `find_shared_recordings.py` generates from the raw frames. It maps each of the 226 exam folders to one of 212 provisional split groups: the A/B/C suffix families plus folders that share re-encoded recordings. This is a second exception to the rule that generated artifacts stay untracked, following D017.
- **Rationale**: The grouping is what the later split (T006) must respect, so everyone should split on the same groups. The file is small (226 rows) and readable, and regenerating it requires the full raw dataset. Keeping it in Git makes the grouping reviewable in diffs. The match evidence behind it (`matches.csv`) stays in ignored `artifacts/`, since the audit summarizes it.
- **Consequence**: `find_shared_recordings.py` remains the reproducible source; rerun it and commit the regenerated file together with any change to the search. The grouping is provisional: T006 decides the final split protocol and may add groups if patient identity is resolved.

---

### D019 — Define the OpenPOCUS binary zone-sequence task

**Date:** 2026-09-26

**Status:** Accepted (T004, issue #6; independently reviewed in PR #17)

- **Decision**: Classify one **exam × lung-zone sequence** as normal or definite abnormal. A sequence contains all available ordered frame files for one exam folder and zone; the prediction unit is not an individual JPEG. Source zone code 0 maps to target 0 (`normal`); codes 1/2/3 map to target 1 (`abnormal`). Code 4 (indeterminate), code 5 (undefined) and blank/missing labels are explicitly excluded. The source `labels.csv` and raw files remain immutable during task derivation.
- **Meaning**: `abnormal` means **B-lines and/or consolidation according to the supplied OpenPOCUS zone-level annotation**. It does not mean trauma, pneumothorax, FAST/free fluid, urgency, critical illness or need for intervention. The supplied exam-level abnormal flag is a different target and is not used for this mapping. Frames are not independently adjudicated ground-truth examples.
- **Rationale**: This user-directed task follows existing zone annotations and combines the sparse definite finding classes without adding clinical labels. It supports investigation of a binary lung-finding task; it does not establish the dataset's suitability for the provisional prehospital/ambulance scenario.
- **Primary classification metrics**: Accuracy and macro-F1, at the zone-sequence unit. The 1,227 normal versus 538 abnormal image-backed candidates are imbalanced, so accuracy alone is insufficient. Final F002 evaluation methodology remains T007 work.
- **Measured reconciliation**: The local snapshot yields 2,712 exam × zone records. Of 1,774 source-label-eligible records (1,233 normal, 541 abnormal), nine lack frames. The remaining **1,765 candidates (1,227 normal, 538 abnormal)** agree with T002 before quality/split exclusions. There are 97 indeterminate, 3 undefined-code and 838 unavailable-label records. These are snapshot measurements, not hardcoded eligibility totals.
- **Implementation**: `data/openpocus/build_binary_task.py` produces deterministic, ignored `artifacts/T004_binary_task/zones.jsonl` and `summary.json`. Every source slot is retained with original exam/zone IDs, label, mapped target, exclusion reason and ordered JPEG paths/count. Source-label eligibility is separate from frame availability and later quality/split eligibility; missing frames do not change a target. Details and runnable checks are in `data/openpocus/README.md`.
- **Unresolved quality constraints**: ED2/Pt160C inherited-label conflicts, Pt45 alternate renditions, A/B/C repeat-exam structure and biological-patient identity are not repaired. T002b found five linked cross-ID sets and conflicting labels on shared recordings (for example Pt67/Pt81 z8 is code 4 versus code 0, changing T004 eligibility). See the dataset audit and D018 for the provisional grouping; T006 must define quality and split rules. This mapping neither merges patients nor establishes independent observations.
- **Consequence**: Target semantics and prediction unit are available for T006/T007. D018's provisional split groups remain in force, but no partition, final patient grouping, model architecture, frame count/sampling strategy, F002 specification/training, adaptive mechanism or compression technique is selected here. The final scenario and dataset fit remain open.

---

### D020 — Record TA feedback and investigate variable-frame adaptive inference

**Date:** 2026-09-29
**Status:** Accepted (T005, issue #7)
**Source:** Email feedback from TA Sudaksh; no feedback from Dolly is recorded here.

- **Sudaksh explicitly confirmed:** The dataset may differ from the target deployment scenario; the adaptive pipeline matters more. Dataset size or strong bias might prevent useful model or adaptive behaviour, which can only be established after training a proof of concept. He considered the proposed use-case design appropriate. Adaptivity along input difficulty/complexity is valid and actively researched, so variable-frame input can be pursued. Since the model may stay fixed while input work changes, throughput is especially important; for this design, the teaching team would expect a Pareto-style evaluation including throughput alongside the assignment-required FLOPs evaluation. Simulated resource, budget and input-fluctuation scenarios are acceptable; on-device evaluation is not expected.
- **Project decision:** Continue studying adaptive edge inference for prehospital/ambulance lung POCUS, with OpenPOCUS explicitly a **technical experimental proxy**. Keep D019's binary normal versus definite abnormal exam × lung-zone sequence task. Results will not claim trauma, pneumothorax, FAST/free-fluid or clinical deployment effectiveness. Investigate variable frame count per zone-sequence prediction as a primary adaptive input/workload complexity direction; the neural network need not shrink. F002 must test whether dataset bias and sample properties permit useful model behaviour and provide evidence for later adaptivity; characterization (T009) must determine useful frame-count operating points. No exact frame counts or architecture are selected.
- **Evaluation decision:** Retain zone-sequence accuracy and macro-F1; preserve the required FLOPs Pareto evaluation with one consistent FLOPs methodology. Add throughput as a required supporting metric for this design. Do not imply energy, battery or thermal measurements have been made.
- **Open Thursday lab clarification:** Whether throughput should be reported as frames per second, completed zone-sequence predictions per second, or both. This does not block F002 planning.

---

### D021 — Use normalized simulated resource-budget states for later control

**Date:** 2026-09-29
**Status:** Accepted (T005, issue #7)
**Source:** Project design decision following TA Sudaksh's confirmation that simulated runtime scenarios are acceptable; these numerical states were selected by the project, not prescribed by the TA.

- **Decision:** Represent the simulated resource budget as normalized `b ∈ [0, 1]`. Active runtime states are `b = 0.25`, `0.50`, `0.75` and `1.00`; `b = 0.00` means unavailable / defer / no inference.
- **Meaning:** The 0% state is not an ML operating point and has no model accuracy or FLOPs point. A later controller/policy will map an active resource state to a measured frame-count operating point, then produce one zone-sequence prediction.
- **Deferred:** Do not map the four active states to frame counts now. F002 and T009 must characterize candidate frame configurations first; there may be more measured candidate configurations than controller states. T009/T012 will determine the useful operating points and policy mapping from measured evidence.

---

### D022 — Use a bounded architecture-selection funnel before freezing F002

**Date:** 2026-09-29
**Status:** Accepted (experimental planning pathway)

- **Decision:** After T006 establishes one common cohort/split and T007 pre-specifies comparable methodology, qualify at most three initial backbone candidates: MobileNetV2, ResNet-18 and DeiT-Tiny. Screen them at one common fixed frame count `K`, retain at most two for a small low/mid/high `K`-response screen, then select exactly one backbone for the frozen F002 static baseline. Downstream adaptive work uses that selected backbone.
- **Rationale:** A small model should not win by size alone, and a more capable model may justify its cost through learning or a better adaptive frontier. Carrying three architectures through dense characterization, compression, policy and final evaluation would divert effort from the course-critical system. Selection uses pre-specified predictive, compute, runtime and adaptive-potential evidence with Pareto/dominance reasoning, not raw accuracy alone.
- **Limit:** This decision selects **no architecture or exact `K`**. The pathway is conditional: if T006/F002/T009 evidence invalidates its assumptions, document and decide the change rather than forcing an obsolete plan. The living detail and decision gates are in `docs/design_pathway.md`.

---

### D023 — Define the T006 cohort quality and known-group split policy

**Date:** 2026-10-01
**Status:** Accepted; independently reviewed and merged in PR #20 (issue #8 closed)

- **Decision:** Keep the standardized/masked `image_...` Pt45 rendition for the primary cohort and omit the overlapping files ending ` 2.jpg` from model input, without changing raw files. Within a verified shared-recording component, missing/unavailable source labels provide no competing semantic annotation and remain T004-excluded. Compare the remaining annotations: exclude affected aliases only for opposite binary targets or definite-versus-indeterminate/undefined eligibility disagreement. Do not copy labels, relabel sources or remove whole exams automatically. Keep same-target shared aliases without deduplication, in their common dependency group.
- **Within-exam finding codes:** ED2 z2/z3 source codes 2/3 and Pt160C z5/z6 source codes 1/2 differ at the finding-code level but all map to T004 binary `abnormal`. As each pair is within one exam, neither creates cross-partition leakage. Keep the source annotations untouched; no additional T006 exclusion or canonicalization applies.
- **Partition:** Use one fixed seed (42) and a deterministic approximately 70/15/15 train/validation/test allocation of complete known-family/shared-recording groups. Balance on sequence-level normal and abnormal counts, not a group-majority label. The same cohort and manifests serve every F002 architecture candidate. Report device distribution rather than optimizing the split against model results or device matching.
- **Measured local snapshot:** T004's 1,765 image-backed candidates yield 1,762 retained T006 sequences (1,225 normal, 537 abnormal). Three genuinely conflicting recording components remove three T004 candidates; Pt45 omits 98 alternate frames across four sequences without removing those sequences. Of 212 source dependency groups, 211 contain retained sequences; ED4 has none. The corrected seed-42 partition has 1,232/265/265 sequences (train/validation/test); 83 source groups change partition relative to the first PR snapshot, before any model training. These are measured outputs, not hardcoded cohort totals.
- **Reproducibility and limit:** Version the small T006 configuration, verified zone-edge table, sequence-level partition and exclusion manifests, group assignment table and summary; their generator and hashes remain the reproducible source. Raw images, caches and full per-frame inventories remain ignored. The split protects against **known** family and shared-recording dependence but cannot certify distinct biological patients. No architecture, frame count `K`, image preprocessing or model-specific split is selected. See `data/openpocus/README.md` for commands and diagnostics.

---

### D024 — Preregister F002 architecture qualification and frozen baseline

**Date:** 2026-10-02
**Status:** Accepted; independently reviewed and merged in PR #22 (T007 complete; issue #9 closed)

- **Decision:** Use the frozen T006 cohort/seed-42 manifests. Compare only MobileNetV2, ResNet-18 and DeiT-Tiny as pretrained per-frame encoders with mean feature pooling and one binary sequence head. Use common full-square 224×224 bilinear RGB preprocessing and ImageNet normalization; train initially at `K=16`, then evaluate at most two selected checkpoints without retraining at nested deterministic `K={4,8,16}`. Keep duplicate aliases as T006 rows, weight each retained shared recording to total training weight 1, and divide every optimizer-step weighted loss by the fixed effective batch size 8; report a separate global epoch weighted mean. Aggregate validation probabilities to one component vote. Predict abnormal iff probability is `>=0.5`; use a common bounded fine-tuning recipe, duplicate-aware checkpoint and runtime-aware reduction gates, and no T006 test inference during T008. Seed 43, if triggered, supplies side-by-side robustness evidence to the human one-backbone gate, without an averaged winner score. After one backbone is recorded, train a fresh seed-2026 F002b static reference under the same rescue/validity gate; stop for a project-level decision if it fails.
- **Measurement:** Count the complete sequence path with one strict fvcore convention including transformer attention, ReLU and all traced arithmetic/ignored-operator checks. Measure model-only sequence latency and throughput on one DAS-5 CUDA GPU model/node class under fixed optimized FP32 flags; practical JPEG pipeline timing and frame throughput remain required supporting evidence. DeiT uses unfused canonical execution for training/selection/FLOPs and equivalent fused attention as its primary runtime path after a `1e-5` maximum-logit-difference check, retaining unfused timing secondarily. No distillation, LoRA/adapters, NAS or adaptive controller enters F002 qualification.
- **Rationale and implementation:** This fixes scientific choices before F002 training while preserving T006 source semantics and allowing a bounded evidence-based backbone decision. Full values, stopping rules, artifact contract and T008 checklist are in [`f002_methodology.md`](f002_methodology.md) and `configs/f002_protocol.json`; small sampler, duplicate and FLOPs helpers have no training code. Course notes in `adaptivity_project_notes.md` are advisory. Independent review and PR #22 merge complete the T007 methodology gate; T008 can implement the accepted protocol.

---

### D025 — Select MobileNetV2 as the F002b static backbone

**Date:** 2026-10-03
**Status:** Accepted by Salo and Charalampos on 2026-10-03 (§6 human evidence gate); the team records acceptance before F002b training, subject to the provenance limit below

**Decision-recording provenance:** The first T008 commit (`ee4aaa4`) has the
merged T007/D024 commit (`32bb08d`) as its direct parent, even though the
working branch may have been created earlier. It was therefore synchronized
or rebased onto the accepted methodology before the first T008 commit; branch
creation time does not establish the experimental base. The message says
F002a was complete, D025 was proposed pending agreement, and F002b had not
trained. The team records that Charalampos accepted MobileNetV2 before F002b
execution, but that acceptance entered Git only in the later commit that also
records F002b. The available PR #23 and issue #10 timelines contain no
independent timestamped acceptance before the F002b job; Git history alone
therefore cannot timestamp the human decision relative to that job. This is a
decision-recording provenance limit, not evidence that F002b used an unaccepted
D024 implementation. The measured seed-2026 F002b run remains the frozen
reference; rerunning after observing its result would not restore prospective
decision timing.

- **Charalampos's reasoning on acceptance:** The deciding evidence is not only the higher K16 F1. MobileNetV2's K4/K8 operating points already beat ResNet-18 K16 on predictive quality with much less compute, and beat its model-only latency across K. ResNet-18's same-K GTX 1080 Ti speed advantage is real and stays documented, but it does not outweigh the predictive, FLOPs, size and K-frontier evidence.

- **Decision:** Train the F002b static reference (seed 2026, K16) on MobileNetV2. ResNet-18 and DeiT-Tiny are not carried forward.
- **Evidence (F002a, seed 42, duplicate-aware validation; `experiments/F002/README.md`):** The final §6 gate kept MobileNetV2 and ResNet-18 with no automatic winner. MobileNetV2 has the higher macro-F1 and lower FLOPs at every K (K4/8/16: 0.813/0.858/0.867 vs 0.750/0.775/0.782; 1.28/2.55/5.11 vs 7.29/14.59/29.17 GFLOPs). ResNet-18 has the >10% model-only runtime advantage on the GTX 1080 Ti at every K (K16: 7.53 vs 11.35 ms). Seed 43 was not triggered (K16 gap 0.086).
- **Why MobileNetV2:** Better on every predictive dimension the gate lists (macro-F1, accuracy and both recalls at K16), better on FLOPs, the project's primary cost axis, and 5× smaller (2.23 M parameters, 9.2 MB FP32). Across K, MobileNetV2 at K4 (0.813, 1.28 GFLOPs, 6.24 ms) and at K8 (0.858, 2.55 GFLOPs, 6.35 ms) both beat ResNet-18 at K16 (0.782, 29.17 GFLOPs, 7.53 ms) on F1, FLOPs and model-only latency at once. At matched or better quality MobileNetV2 is also faster; the same-K runtime rule does not look at this. ResNet-18's same-K advantage is model latency on the benchmark GPU. In the practical JPEG pipeline it is 3 to 5 ms per request against 21 to 87 ms totals, though at K4 that is still 15% (20.99 vs 24.80 ms). The protocol does not apply the runtime rule to the pipeline path, and the GPU is a comparison platform, not the edge device.
- **Throughput (D020):** Sudaksh asked for throughput alongside FLOPs for this variable-frame design. At matched or better quality MobileNetV2 still has the higher model-only throughput (K8: 157 seq/s at 0.858 vs ResNet-18 K16: 133 seq/s at 0.782). Against it: on the 1080 Ti, MobileNetV2's model-only throughput barely changes from K4 to K8 (160 vs 157 seq/s), while ResNet-18's scales with K (357/232/133). In the practical JPEG pipeline both scale with K (MobileNetV2 40/23/12, ResNet-18 48/24/12 seq/s), because decoding dominates. A model-only throughput-vs-K curve on this GPU would look flatter for MobileNetV2.
- **Training stability:** Neither run needed a rescue, but both were unstable on validation. MobileNetV2's predicted-abnormal rate swung 0.14 → 0.41 → 0.61 → 0.17 over epochs 1–4 before settling (epochs 5–7 macro-F1 0.845–0.867). ResNet-18 peaked at epoch 1 and fell to 0.677 by epoch 4.
- **Rejected:** DeiT-Tiny, eliminated by the deterministic K16 gate. ResNet-18, for lower predictive quality at 5.7× the FLOPs. Training several F002b references was also considered and rejected, because D024 fixes exactly one fresh static reference.
- **Post-hoc robustness check (outside the protocol, not a decision rule):** On the same 254 validation units, the MobileNetV2 minus ResNet-18 K16 macro-F1 gap of 0.086 has a unit-bootstrap 95% CI of [0.021, 0.151] and a group-bootstrap CI of [0.008, 0.171]. Exact McNemar gives 34 vs 14 discordant units, p = 0.0055. Within device 0, the only device with many abnormal units (60/132), MobileNetV2 leads by more (macro-F1 0.884 vs 0.727), so its advantage is not a device artefact.
- **Caveat:** Single-seed validation evidence that also drove selection may be optimistic. MobileNetV2's latency on this GPU barely changes from K4 to K8 (6.24 vs 6.35 ms), which T009 must keep in mind when reporting runtime next to FLOPs.

---

### D026 — Select TinyViT as the adaptive-system backbone

**Date:** 2026-10-05
**Status:** Accepted by Charalampos and Salo

- **Decision:** Carry the F002c timm
  `tiny_vit_5m_224.dist_in22k_ft_in1k` epoch-8 checkpoint forward as the
  adaptive-system backbone for T009 onward. Its canonical shared frozen copy is
  [`checkpoints/f002c_tinyvit5m_epoch8.pt`](../checkpoints/f002c_tinyvit5m_epoch8.pt),
  with SHA-256
  `30b70734426e84f429b83011abca04511540149d1fc3e416274c6fbcb14853b2`.
- **Evidence retention:** The ignored source
  `artifacts/F002c/tinyvit5m_seed2026/checkpoint.pt` remains part of the raw
  experiment history. The complete external artifact archive remains archival
  evidence outside Git.
- **Static-reference boundary:** D025 continues to govern the frozen F002b
  MobileNetV2 seed-2026 K16 static reference. D026 does not retrain, replace or
  modify that checkpoint, invalidate F002a/F002b, or rewrite the historical F002
  architecture-selection result.
- **K16 validation evidence:** On the same duplicate-aware validation units at
  threshold 0.5, TinyViT versus frozen MobileNetV2 produced macro-F1
  **0.8737 vs 0.859**, accuracy **0.8937 vs 0.886**, normal recall
  **0.9213 vs 0.944**, and abnormal recall **0.8289 vs 0.750**.
- **Same-checkpoint K response:** Without K-specific retraining, TinyViT reached
  macro-F1 **0.8602 / 0.8676 / 0.8737** at K4/K8/K16. This provides a useful
  initial frame-count response while retaining a transformer architecture that
  exposes additional bounded adaptive dimensions for T009.
- **Rationale:** This is a project engineering and research choice for the
  adaptive phase. TinyViT is predictively competitive with the static reference,
  improves abnormal recall in this run, and provides a richer architecture for
  studying shared-model adaptive inference. MobileNetV2 remains the required
  static comparison.
- **Supersession scope:** D026 supersedes only the earlier consequence in D022/D025
  that downstream adaptive work necessarily uses the backbone selected for F002b.
  All other accepted data, task, F002 protocol and static-reference decisions
  remain in force.
- **Limits:** This does not establish statistical or universal superiority. The
  models used different training recipes, validation selected both checkpoints,
  only one TinyViT seed was run, and TinyViT still needs consistent F002-style
  FLOPs, latency and throughput characterization. The held-out T006 test partition
  remains untouched during T009–T012 development.

---

### D027 — Use frame count as the primary adaptive axis

**Date:** 2026-10-06
**Status:** Proposed by Salo from T009/F003; needs Charalampos's agreement

- **Decision:** Use frame count K as the primary adaptive axis for T010, with the frozen D026 TinyViT checkpoint and nested spread frames. Carry the measured K1/K2/K4/K8/K12/K16 candidates into T010; T010 selects the final operating points from their measured quality-cost trade-offs. The core T012 policy switches between the selected points by resource budget. Input-driven escalation stays an open T012 question.
- **Evidence (F003, validation):** macro-F1 0.816 / 0.860 / 0.868 / 0.874 and AUROC 0.888 / 0.937 / 0.950 / 0.952 at K1 / K4 / K8 / K16 for 1.29 / 5.15 / 10.30 / 20.60 GFLOPs. K8 and K12 are practically equal to K16, K4 slightly lower, K1/K2 clearly lower. Threshold-free, the K curve lies above every alternative tested; the trend holds within device 0.
- **Rejected:** frozen-trunk early exits and lower resolution (lower AUROC than the K curve at similar cost) and consecutive-frame sampling (worse at K4, p = 0.04).
- **Not resolved:** input-driven K → K16 escalation (cross-validated reruns scatter around plain K4; only 7 units fixable from K4) and the short-scan idea (no detectable effect of scan share).
- **Static baseline:** at its own cost (5.1 GFLOPs), TinyViT K4 ties MobileNetV2 F002b on macro-F1 at 0.5 with different errors. Run at the same nested K, MobileNetV2's frame-count curve is level with or below TinyViT's at every matched cost, most clearly at the cheap end (macro-F1 0.761 vs 0.816 at about 1.3 GFLOPs), though no single matched comparison is resolved. MobileNetV2 has higher device-0 normal recall.
- **Shared weights:** all K run the same weights, so sharing holds by construction; nested frames allow anytime upgrades (K4 → K16 reuses the computed frames). A stronger shared-weight mechanism (trained exits or training across K) changes the weights and should be decided before T010.
- **Limits:** one checkpoint and seed, 76 abnormal validation units, unknown frame rate; MobileNetV2's low-K response differs between checkpoints (F002a vs F002b). Details: [`experiments/F003/README.md`](../experiments/F003/README.md).

---

### D028 — T010 operating points: K1/K2/K4/K8/K12/K16 on the D026 weights

**Date:** 2026-10-07
**Status:** Proposed by Salo from T010/F004; needs Charalampos's agreement (builds on D027, also still proposed)

- **Decision:** The adaptive family is the unchanged D026 TinyViT checkpoint served at K = 1, 2, 4, 8, 12 and 16 nested spread frames through `experiments/F004/family.py` (`AdaptiveTinyViT`, `AnytimeClip`). K adapts the input's temporal window and reuses cached frame features (lecture 4, approach families 4 and 8): each point's computation is contained in every larger point's, so a clip can continue to a larger K (anytime inference). There are no exit heads; a budget policy chooses K. The weights stay D026's; the multi-K fine-tune (F004b) is not adopted. If D029 is accepted, the deployable weights become its pruned seed-2026 file (fine-tuned across K with distillation), and this decision's family is the unpruned reference.
- **Evidence (F004, validation):** macro-F1 0.816 / 0.829 / 0.860 / 0.868 / 0.870 / 0.874 and accuracy 0.846 / 0.854 / 0.882 / 0.890 / 0.890 / 0.894 at 1.29 / 2.57 / 5.15 / 10.30 / 15.45 / 20.60 GFLOPs, reproducing F003. Every point reads all 215 parameter tensors of one 20.4 MB file, equal to the checkpoint. K4 → K16 costs 15.45 instead of 20.60 GFLOPs; moving down encodes nothing. Warm switching between K changes per-K latency by at most 0.13 ms (three GTX 1080 Ti hosts); the first request at a new K costs up to about 150 ms (cuDNN tuning), so deployments warm up every K.
- **Why K12 is in:** it is on the quality/FLOPs front and costs 0.75 × K16, which may match a T012 budget state; measuring it costs no training. Only K1 and K2 are resolved below K16 on validation.
- **Rejected:** adopting the F004b multi-K fine-tune. Its pre-specified seed-2026 test failed (+0.003 mean macro-F1 over D026 against a 0.01 bar; K16 0.874 → 0.842). Over three seeds it averages +0.008 over D026 (also below the 0.01 bar) and +0.010 over a K16-only control, mostly at K1/K2, with K16 varying 0.842–0.896: a small low-K gain is possible but unproven, and the epochs were selected on validation. Revisit only with epochs selected on a split held out from validation.
- **Consequences:** T011 compresses the one D026 weight set, which applies to every point. T012 maps budget states to these measured points and needs to define what b is a fraction of.
- **Limits:** one checkpoint, validation reused for the checkpoint, the axis, the points and F004b's epochs; desktop GPU/CPU timing, not an edge device. Details: [`experiments/F004/README.md`](../experiments/F004/README.md).

---

### D029 — T011 compression: nested structured pruning with distillation

**Date:** 2026-10-07
**Status:** Proposed by Salo from T011/F005; needs Charalampos's agreement (builds on D027 and D028)

- **Decision:** Compress the adaptive family by nested structured pruning of TinyViT's hidden channels (two MBConv blocks, ten transformer MLPs), ordered by first-order Taylor importance and fine-tuned from D026 with the sandwich rule and offline distillation from the frozen D026 model (F005b). Width w ∈ {1.0, 0.75, 0.5, 0.25} keeps the first w of every hidden group, so one weight file serves 4 widths × 6 K: the widths are the slimmable mechanism, K the input-adaptation mechanism. The deployable checkpoint is seed 2026, chosen after the results by the convention of taking the first seed, not by its results (it is the weakest seed on the narrow widths). Dynamic INT8 (F005a) is reported as a size option, not part of the family.
- **Evidence (F005, validation, three seeds):** mean macro-F1 over the five trained K: D026 0.849; widths 100/75/50/25%: 0.853 / 0.857 / 0.856 / 0.846, at 1.29 / 1.10 / 0.91 / 0.72 GFLOPs per frame. The pre-specified rule (width 1.0 ≥ D026 − 0.01, width 0.5 ≥ D026 − 0.02) holds. The 50% width is within 0.011 of D026 at every K; the 25% width within 0.033 (seed 2026: 0.04 lower at K12/K16). AUROC is at or above the unpruned family at every cost. A model trained only at 50% scores like the nested 50% width (0.854 vs 0.856), at about a quarter of the training compute. Switching width and K per request adds no measurable latency (GPU ≤ 0.6%, CPU within repeat noise), and `NestedAnytimeClip` serves a clip across width changes exactly.
- **Rejected:** L1 importance (train BCE 0.615 vs 0.315 for Taylor at 50% width without retraining) and a random channel order (after the same training, −0.03 macro-F1 at 75/50% width and −0.10 at 25%, three seeds each); static INT8 in ONNX Runtime (AUROC −0.01 to −0.02 and 1.5–1.8× slower on the CPU; a `reduce_range` rerun was worse, so weight saturation is not the cause); dynamic INT8 as the compression step (no FLOP change, not faster on this CPU, 3.2× smaller file).
- **Limits:** the deployable weights are fine-tuned across K with distillation, i.e. they include the recipe F004b did not adopt on its own, accepted here under a margin rule on point estimates (no gain required, at most 0.01–0.02 loss) because pruning forces the weights to change, unlike F004b, which had to justify changing them (+0.01 required); neither margin is resolvable with paired intervals of about ±0.03–0.05; epochs selected on the same validation units the grid is reported on, and the front's best points are local peaks; one of 72 seed × cell McNemar tests is below 0.05 (p = 0.031), about what chance gives; part of the gain over D026 comes from the fine-tuning with distillation, not the pruning; only the MBConv conv3 BNs get per-width statistics (recalibrating all 19 downstream BNs after training lowered the narrow widths by 0.008 and was not adopted); narrow widths are noisier across K and seeds; desktop timing. Details: [`experiments/F005/README.md`](../experiments/F005/README.md).

---

# Open Decisions

The following are intentionally **not decided yet**:

- How well the validation-supported model and adaptive behavior generalize; OpenPOCUS remains a technical proxy rather than clinical deployment evidence.
- Any additional biological-patient identity mapping if new provenance becomes available; D023 fixes the current T006 quality exclusions and known-group split protocol.
- Final acceptance of D027 (frame count as the primary axis) and D028 (the T010 operating points).
- Mapping from active resource states to the measured F004 operating points (T012), including what b = 0.25/0.50/0.75/1.00 is a fraction of.
- Final acceptance of D029 (nested structured pruning as T011's compression).
- Whether resource conditions beyond the accepted simulated budget states will also be measured.
- Whether input uncertainty becomes part of the runtime policy.
- Whether an optimization-based controller is worth implementing.
- Final scenario-specific utility / global performance measure.
- Exact latency, battery, thermal, and network assumptions.
- Final held-out test protocol.
- Whether restricted FAST/pneumothorax data become available.
- Any later teaching-team clarification of throughput presentation; D024 specifies reporting both frames/s and completed zone-sequence predictions/s.
