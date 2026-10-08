# Lecture 05 — Project Design Notes

**Course:** Big Data at the Edge  
**Lecture:** *Efficient Model Design, Transfer, and Adaptation at the Edge* — Dolly Sapra, 28 September 2026  
**Source:** `efficient_model_design(2).pdf` (70 PDF pages)  
**Purpose:** extract decisions, constraints, implementation options, and evaluation rules relevant to the **Big Data at the Edge project**. This is **not** a complete study summary of the lecture.

> **Reading convention:** `PDF p. N` refers to the PDF's physical page number, not the smaller slide number printed on the slide. **Lecture concept** identifies material taught in the slides; **Project implication** is an engineering recommendation derived from that material. Project applications are proposals, not claims that the lecture prescribes a particular architecture.

---

## 0. The key design principle

The lecture broadens the question from **“How do we run inference cheaply?”** to **“How do we build and maintain a useful edge model across devices, domains, and time?”** (PDF pp. 2–3, 68–70).

| Problem we actually have | Technique to consider | Caution |
|---|---|---|
| Trained model too large / expensive for target | Distillation; pruning; quantization | Choose based on **measured** target-device constraints. |
| Cheap mode loses too much accuracy | Teacher–student / self-distillation | A strong teacher must also work on the target domain. |
| New task/site/domain requires efficient **training updates** | LoRA, adapters, selective fine-tuning | **PEFT does not shrink the frozen base model.** |
| Need a model architecture tailored to unusual hardware | Hardware-aware NAS | Search can cost far more than the course project can justify. |
| Performance degrades after deployment | Drift monitoring; recalibration; continual adaptation | Updating can damage older performance. |
| Private data live on multiple devices | Federated learning, possibly with LoRA | No raw-data transfer ≠ automatic privacy/security. |
| Need to use cloud/server compute | Split inference or edge–cloud coordination | Intermediate features can exceed raw-image transmission cost. |

**Decision rule:** select a technique because it solves a demonstrated bottleneck. The techniques are complementary, not obligatory steps in one pipeline (PDF pp. 66–69).

---

## 1. Knowledge distillation: transfer quality into a cheaper runtime model

**Lecture basis:** PDF pp. 5–26.

### L05-01 — Keep teacher cost offline when possible

**Lecture concept:** A large, accurate, expensive **teacher** can supervise a small **student**. The teacher is pretrained/frozen, gradients update the student, and the cheap student is deployed at the edge (pp. 7, 13).

**Project implication:** Distillation is a candidate when we can train/profile a stronger reference model centrally but cannot afford its inference cost on the edge. Compare the student with **the same architecture trained from hard labels alone** to show whether distillation adds value.

### L05-02 — Distill soft information, not just class labels

**Lecture concept:** Teacher soft targets expose relative class similarities (“dark knowledge”) that one-hot targets do not (p. 8). A temperature `τ > 1` softens logits:

```text
p_i(τ) = exp(z_i / τ) / Σ_j exp(z_j / τ)
```

- `τ = 1`: ordinary softmax.
- `τ > 1`: reveals more relative structure in non-top classes.
- Very large `τ`: approaches uniform probabilities and loses discrimination.
- The lecture suggests typical `τ` values around 2–10 **as a starting range**, not an optimal setting (pp. 9–10).

**Project implication:** Tune the distillation temperature on a **validation set representative of the deployment domain**. Do not assume softer predictions are always better.

### L05-03 — Combine ground truth with teacher guidance, when labels exist

The lecture uses (p. 11):

```text
L = α · CE(y, p_student(τ=1))
  + (1−α) · τ² · KL(p_teacher(τ) || p_student(τ))
```

- `CE`: anchors the student to real labels; can correct teacher errors.
- `KL`: transfers softened teacher behavior.
- `τ²`: compensates for loss-gradient scaling with temperature.
- `α`: controls the balance; it is a hyperparameter.

**Lecture concept:** Excessively strong teacher supervision can transfer teacher mistakes; excessive hard-label emphasis discards soft-target information (p. 14). With **no ground-truth labels**, the lecture sets `α = 0`, yielding teacher-only training (p. 12).

**Project implication:** Record `τ`, `α`, student-only baseline, teacher accuracy, student accuracy, and calibration. In teacher-only training, separately evaluate against **genuine held-out labels** when available; teacher agreement alone does not establish task accuracy.

### L05-04 — Choose how much knowledge to transfer

| Transfer family | What is matched | Main practical consideration |
|---|---|---|
| **Response-based** | Logits / soft predictions | Simple, often architecture-agnostic. |
| **Feature-based** | Hidden feature maps / attention maps | Richer signal; may require matching layers and a projection. |
| **Relation-based** | Sample-to-sample distances/similarities | Can preserve relative structure; expensive relation computation. |

Lecture: pp. 15–17. A feature-distillation regressor (e.g. a `1×1` convolution) can map student features to the teacher's dimensions **during training only** and be discarded for deployment (p. 17).

**Project implication:** Start with **response-based KD**. Add feature loss only if small-model accuracy is inadequate and the extra training complexity is justified. Distinguish training-only adapters from modules that add inference latency.

### L05-05 — Distillation is relevant to elastic modes, not only separate students

**Lecture concept:** Distillation can be offline, online, or **self-distillation**. Deeper exits can teach earlier exits; the lecture explicitly relates self-distillation to SHARD (p. 18). A teacher cohort can also cover different domains, but disagreement makes aggregation a design problem (p. 19).

**Project implication:** For a nested/early-exit adaptive model, consider **full-to-small / deep-to-shallow supervision** to preserve the useful accuracy–cost frontier discovered in Lecture 04. This can be more relevant than training a completely separate student.

### L05-06 — Unlabeled sensor archives help only if the teacher is trustworthy

**Lecture concept:** Teacher outputs can pseudo-label unlabeled data; the lecture also discusses data-free distillation and cross-modal teacher–student transfer (pp. 20–22). Unlabeled-data KD explicitly warns that value depends on teacher reliability and domain match (p. 22).

**Project implication:** If using unlabeled frames/clips, first establish teacher performance on the **actual target data distribution**. For cross-modal KD, paired/time-synchronized data are needed; do not introduce this merely because the idea is attractive.

### L05-07 — Decide whether to compute teacher outputs online or store them

**Lecture concept:** Two KD pipelines are presented (p. 23):

- **On-the-fly:** run teacher every training pass; discard outputs; no large logits store; higher teacher compute cost.
- **Precomputed:** run teacher once; store soft targets; cheaper repeated student training; higher storage cost.

The lecture gives a large-scale illustration where complete logits become several TB and mentions storing only **top-k** outputs (p. 21).

**Project implication:** If many student configurations reuse the same data, precomputed teacher outputs may be worthwhile. Benchmark storage against teacher recomputation time; if caching labels, record teacher version, preprocessing, temperature, and sample IDs for reproducibility.

### L05-08 — Validate the limits of distillation

**Lecture concept:** Students can sometimes outperform teachers through regularization and extra data, but capacity limits, teacher–student gaps, poor calibration, and domain mismatch can cause KD failure (pp. 24–25).

**Project implication:** Do **not** assume KD guarantees improved student accuracy. A meaningful experiment compares: teacher, small student without KD, small student with KD, and their resource costs.

---

## 2. Parameter-efficient fine-tuning: cheap updates ≠ cheap inference

**Lecture basis:** PDF pp. 27–45.

### L05-09 — Distinguish *inference adaptation* from *learning new parameters*

**Lecture concept:** Lecture 04 primarily asks how much computation to run **per inference**. Lecture 05 introduces **parameter-efficient fine-tuning (PEFT)**: freeze most base weights and train small additional updates for a task, site, user, or domain (pp. 28–29).

**Project implication:** PEFT should only enter the project if **post-training specialization/update cost** is part of the research problem. A controller that switches between pre-trained inference modes is not itself PEFT.

### L05-10 — Know the PEFT choices before choosing LoRA

The lecture groups PEFT methods (pp. 30–32):

- **Additive:** adapters, prompt/prefix methods — add small trainable components.
- **Selective:** update only chosen existing parameters (e.g. head, bias, BN, selected layers).
- **Reparameterized:** LoRA and related methods — learn compact weight updates.

**Project implication:** Test a simple **head-only / BN-only** adaptation baseline before adding LoRA, unless the research question explicitly concerns adapters. Small modules can still add latency if executed as separate layers (p. 32).

### L05-11 — LoRA changes update size, not the base-model footprint

**Lecture concept:** LoRA keeps `W₀` frozen and learns a rank-limited update (pp. 33–37):

```text
W = W₀ + ΔW
ΔW = B A
W₀ ∈ R^(d×k), B ∈ R^(d×r), A ∈ R^(r×k)
rank(ΔW) ≤ r
```

Trainable parameters are about `r(d+k)` instead of `dk` for that matrix. The lecture example uses a 4096×4096 matrix: at `r=8`, LoRA has **65,536 trainable parameters vs 16,777,216**, about **0.39%** (p. 36).

**Critical caution from p. 39:** the entire **base model still has to be stored and executed**. LoRA does **not** replace pruning, quantization, or distillation when the base itself is too expensive for edge inference.

### L05-12 — LoRA rank is a resource/expressivity knob

**Lecture concept:** Lower rank means fewer update parameters and smaller transferable updates; higher rank adds expressivity and cost. Large domain changes may exceed low-rank capacity (pp. 35, 37).

**Project implication:** If site-specific adaptation is included, measure **accuracy gain vs trainable parameters/update bytes and training memory**, not just adapted accuracy. Use a small rank sweep instead of assuming a default rank is optimal.

### L05-13 — Decide between merged and swappable updates

**Lecture concept** (p. 38):

- **Merge `W₀ + BA`:** no extra inference path/overhead from LoRA itself, but changing specialization requires different merged weights or re-merging.
- **Keep separate adapters:** share one base, hot-swap task/site updates, but incur some per-layer compute and management complexity.

The lecture also describes context-based adapter routing and warns that a wrong adapter may be worse than a generic model (p. 41).

**Project implication:** If testing multiple acquisition environments/sites, compare one generic model against environment-specific updates **and** measure selection errors and adapter-switch overhead. Never count update-file size as total on-device model size.

### L05-14 — Compression and PEFT can combine

**Lecture concept:** QLoRA keeps a quantized (4-bit) frozen base and trains higher-precision low-rank adapters (p. 40). The lecture summarizes the roles:

```text
pruning       → fewer weights/structural units
quantization  → fewer bits per weight
distillation  → a new, potentially smaller student
LoRA/adapters → fewer trainable parameters / smaller updates
```

**Project implication:** Combine methods only to solve distinct bottlenecks. E.g. compression for inference feasibility, then adapters for domain-specific updates. QLoRA is not, by itself, evidence of low-latency edge inference.

### L05-15 — Where learning occurs is a design choice

**Lecture concept:** fine-tuning can be **cloud**, **on-device**, or **hybrid** (collect local statistics, train centrally, redistribute adapters). Choice depends on privacy, connectivity, latency, and update frequency (pp. 43–45).

**Project implication:** State explicitly whether **training** is expected on the edge. If not, avoid claiming on-device learning simply because LoRA is involved. If updates are deployed, plan versioning, compatibility, signing, and rollback.

---

## 3. Hardware-aware neural architecture search (NAS)

**Lecture basis:** PDF pp. 46–50.

### L05-16 — Search only if architecture selection is the actual bottleneck

**Lecture concept:** NAS searches choices like depth, channels, kernels, block types, attention heads, input resolution, and skip connections; the combinatorial space becomes enormous (p. 47).

> Pruning asks **what to remove** from an existing model; NAS asks **what architecture should have been built**.

**Project implication:** For a constrained project, first test a small set of reasonable existing architecture/configuration choices. A full NAS project is a separate scope commitment.

### L05-17 — NAS has three components

1. **Search space:** which architectures/configurations are legal?
2. **Search strategy:** evolutionary, RL, differentiable, or weight-sharing/one-shot search.
3. **Objective/evaluation:** accuracy proxy **plus target-device latency, memory, and energy** (p. 48).

**Project implication:** If NAS is used, specify all three before coding. Make the objective match the actual system constraint; do not optimize accuracy alone and claim hardware awareness.

### L05-18 — Measure the hardware, not just mathematical FLOPs

**Lecture concept:** Architecture rankings can change by hardware, and efficient-looking tensor shapes may not run efficiently in practice (pp. 47–50; final takeaway p. 68).

**Project implication:** Benchmark candidate subnets on the *intended available device/runtime*. If only a remote GPU is accessible, label results as **proxy measurements** and do not generalize them to a different edge processor without evidence.

### L05-19 — Search cost and baselines matter

**Lecture concept:** RL/evolutionary NAS can be very expensive; weight-sharing and differentiable search reduce some costs. The lecturer explicitly calls **random search a strong baseline** (p. 49).

**Project implication:** Compare under equal search/training budgets. For a small project, a controlled manual sweep or random search may deliver more defensible results than a complex but under-resourced NAS implementation.

### L05-20 — NAS is often not justified

**Lecture concept:** NAS is more attractive for large-scale, long-lived deployment, unusually tight constraints, or unusual target hardware. It is less appealing when standard efficient architectures meet requirements, the dataset is small, or target-hardware measurements are unavailable (p. 50).

**Project implication:** Treat NAS as **optional** until experiments show that existing model families or elastic operating points cannot meet requirements.

---

## 4. After deployment: drift, cheap fixes, and continual learning

**Lecture basis:** PDF pp. 51–60.

### L05-21 — Plan for distribution shift, not a one-time deployment

**Lecture concept:** Inputs, environments, classes, users, and sensors change over time. The training and deployment distributions can diverge (pp. 52–54).

| Shift | Lecture example | Conceptual distinction |
|---|---|---|
| **Covariate/data shift** | camera summer → winter | input distribution changes |
| **Label/prior shift** | many more boats on King's Day | class frequencies change |
| **Concept drift** | previously normal vibration becomes abnormal | relationship between input and correct label changes |

The slide emphasizes that **concept drift is particularly dangerous** because similar inputs can require different answers (p. 53).

**Project implication:** Document which shifts are plausible in the scenario and how they would be detected/evaluated. Do not call every performance drop “concept drift.”

### L05-22 — Use the cheapest response that fits the failure

The lecture offers a progression (pp. 55–56):

```text
1. Recalibrate BN statistics / thresholds         tiny
2. Test-time adaptation on unlabeled stream      small
3. Update head / adapter with local labels       moderate
4. Federated update from many devices            fleet-wide
5. Cloud retrain + over-the-air redeployment      large/slow
```

**Project implication:** Do not jump directly to online fine-tuning. First ask whether better calibration or an updated threshold solves the measured degradation.

### L05-23 — Unlabeled adaptation can help, but can collapse

**Lecture concept:** Under covariate shift, BN running statistics can be re-estimated from recent unlabeled inputs without gradient training. The TENT-style alternative uses **entropy minimization** with gradient updates during deployment (p. 56).

**Main risk:** repeated adaptation on a long or non-stationary stream can accumulate errors or collapse into overconfident incorrect predictions.

**Project implication:** For any test-time update method, report both **short-term recovered performance** and **performance after repeated updates**. Include a frozen/no-adaptation baseline.

### L05-24 — Avoid catastrophic forgetting

**Lecture concept:** Adapting to one environment can overwrite behavior needed for another (summer performance lost after winter training). This is the **stability–plasticity trade-off** (p. 57).

The lecture lists three mitigation families (p. 58):

| Family | Example | Cost/trade-off |
|---|---|---|
| **Regularization** | EWC, Learning without Forgetting | May weaken across long task sequences. |
| **Replay** | Store/recreate old examples | Memory and privacy cost. |
| **Parameter isolation** | Adapter/LoRA per task or season | Need to identify/select the correct module. |

**Project implication:** If implementing learning-after-deployment, evaluate on **both old and new conditions** after updating. Report accuracy retention, update size, and rollback ability.

### L05-25 — Update governance belongs in the architecture

**Lecture concept:** Model updates require privacy/security safeguards, version control, signing, rollback, and failure recovery. Keeping raw data local is not a complete privacy guarantee (p. 63).

**Project implication:** If deploying more than one checkpoint/adapter version, define a minimal **update protocol**: compatible base version, validation gate, signed artifact when applicable, deployment record, and rollback path. If no live update mechanism is in scope, record this as an explicit limitation.

---

## 5. Distributed learning, split inference, and edge–cloud coordination

**Lecture basis:** PDF pp. 59–64.

### L05-26 — Federated learning exchanges updates, not raw data

**Lecture concept:** Devices train locally, send model updates for aggregation, receive a new global model, and repeat (p. 59). LoRA can reduce communication by sending only small adapter updates; the lecture gives an **illustrative** 500 MB vs 2 MB example (p. 60).

**Project implication:** FL is most relevant for **multiple distributed devices with private heterogeneous data**. It is not needed for ordinary offline training of one model on a public dataset. Update information can still leak private information (p. 63).

### L05-27 — Split inference must be evaluated end-to-end

**Lecture concept:** Execute early layers on-device and later layers on an edge server/cloud. The split point trades local compute, data sent, privacy exposure, and network latency (p. 61).

**Key correction from p. 64:** intermediate features can be **much larger** than a compressed input image. Early split is not inherently bandwidth-saving.

Possible mitigations from the lecture:
- quantize features to INT8
- learn a compact bottleneck via teacher–student training
- split later, after a layer with smaller output

**Project implication:** For any split/offload claim, calculate and benchmark:

```text
local pre-split inference time
+ feature encoding time
+ transmission size / link throughput / network overhead
+ remote inference time
+ response transmission time
```

Compare this with **fully local inference** and **sending the compressed original input**. Include offline behavior and privacy limitations. The lecture notes that intermediate features may still leak private information.

### L05-28 — Separate fast local inference from occasional heavy retraining

**Lecture concept:** Edge nodes can run local inference while cloud resources handle heavy analysis/retraining. Updates can wait for favorable network/energy conditions; cloud teachers can periodically distill into edge students (p. 62).

**Project implication:** A realistic architecture need not learn online on every frame. It can use:

```text
FAST PATH: on-device sensing → adaptive inference → prediction
SLOW PATH: monitor → collect permitted statistics → central evaluation/update
                           → validate → deploy a new student/adapter
```

This separation is often simpler to implement and evaluate.

---

## 6. What Lecture 05 adds to our Lecture 04 adaptive-inference project

Lecture 04 gave the core structure:

```text
SENSE → ESTIMATE → CHOOSE → EXECUTE → UPDATE
```

Lecture 05 adds **model construction, specialization, and lifecycle maintenance** around that loop:

```text
OFFLINE DESIGN / TRAINING
  target-device requirements
      ↓
  select teacher / baseline architecture
      ↓
  train capable reference model
      ↓
  optionally distill smaller or nested configurations
      ↓
  optionally prune / quantize / select architecture
      ↓
  profile every configuration on available hardware
      ↓
  validate accuracy, confidence, and cost frontier

ONLINE INFERENCE
  sense input + system state
      ↓
  select operating mode (account for controller / switch cost)
      ↓
  execute on edge → result
      ↓
  log accuracy proxies, latency, resource state, and failures

SLOW MAINTENANCE LOOP (OPTIONAL)
  detect performance / distribution change
      ↓
  recalibrate first; adapt/retrain if needed
      ↓
  validate new + old conditions
      ↓
  version, deploy, monitor, rollback if necessary
```

**Important distinction:** changing depth/width/tokens **during inference** is not the same as changing learned weights using PEFT or continual learning. Both are called adaptation, but they solve different problems and operate on different time scales.

---

## 7. Project-specific interpretation: adaptive medical edge inference

> **Project application, not a prescribed lecture requirement.** These recommendations are tailored to the project's previously discussed medical ultrasound / transformer direction. Keep them conditional on the final task, dataset, edge-device access, and locked experimental design.

### Highest-value ideas to investigate first

**A. Can a stronger teacher improve a cheap operating mode?**  
Train/evaluate a capable reference model; then compare a cheaper/student mode **with vs without KD**. Use the actual clinical task labels where available and evaluate teacher correctness on held-out exams, not just teacher–student agreement. This is a direct application of L05-01 through L05-08.

**B. Are all proposed modes useful on the accuracy–latency frontier?**  
Profile all candidate configurations on the available runtime; compare accuracy, latency distributions, memory, and if feasible energy. Combine Lecture 05's hardware-aware design principle with Lecture 04's controller/switching-cost rules. KD only earns a place if it improves that frontier.

**C. How robust are predictions and confidence to deployment-like shifts?**  
Consider device/site variation, image-quality changes, acquisition settings, and temporal changes. Measure accuracy and calibration under available **real or well-justified simulated** shifts. This may be more valuable than another architecture variation.

### Potential second-stage ideas (only if the core works)

**D. Cheap domain specialization using adapters/LoRA.**  
Consider this only if a meaningful site/device/domain split exists and ordinary calibration is inadequate. Benchmark head-only adaptation against LoRA; report update size and retained performance on the original data.

**E. Teacher pseudo-labeling for extra unlabeled frames.**  
Only if unlabeled data are available and teacher errors/domain mismatch can be evaluated. Keep label provenance and avoid training/evaluation leakage.

**F. Limited architecture search.**  
Use a compact candidate sweep or random search under a fixed compute budget before attempting advanced NAS.

### Not automatically justified by this project

- Full continual-learning pipeline on-device without long-term deployment streams.
- Federated learning without multiple devices/private local datasets.
- Split inference without a reliable network and a real offload requirement.
- Cross-modal distillation without time-aligned multimodal data.
- LoRA just to claim a smaller inference model: the base remains.

### Concrete experiments suggested by this lecture

| Experiment | Comparison | Primary question | Minimum measurements |
|---|---|---|---|
| **KD ablation** | small model with CE vs CE+KD | Does the teacher help the cheaper mode? | accuracy, calibration, training cost, inference latency |
| **Mode frontier** | all static subnets/configs; adaptive policy | Are the operating points useful and hardware-efficient? | accuracy vs latency, memory, controller/switching overhead |
| **Shift stress test** | unchanged model vs calibration update | How much accuracy/confidence degrades under shift? | per-condition accuracy, confidence/reliability, old-condition retention |
| **PEFT pilot** *(optional)* | head-only vs LoRA/adapters | Is small-update specialization worth complexity? | new- and old-domain accuracy, update bytes, time/memory |
| **Search baseline** *(optional)* | manual/random candidates vs sophisticated search | Is architecture search earning its cost? | validation accuracy, hardware latency, total search budget |

**Evaluation hygiene (project recommendation):** use held-out **independent exams/subjects** when applicable, audit duplicates and near-duplicates across partitions, and separate model-quality failures from policy-selection failures. The lecture does not specify these dataset-splitting details; they are project-specific safeguards.

---

## 8. Practical decision gates

Before adding any Lecture 05 method, answer:

1. **What measured limitation does this method fix?** (accuracy, inference cost, update cost, drift, communications?)
2. **Is there a cheaper baseline?** (CE-only student, calibrated thresholds, head-only fine-tuning, random search, fully local inference?)
3. **Which resource is actually constrained?** (RAM, storage, training memory, latency, energy, bandwidth, privacy?)
4. **What changes at deployment?** (active compute graph, learned parameters, server split, model version?)
5. **Does it improve the real objective under a fair comparison?**
6. **What new risk is introduced?** (teacher errors, poor calibration, forgetting, wrong adapter, update leaks, switching overhead?)
7. **Can we evaluate that risk with available data and hardware?**

If no defensible answer exists, **defer the technique rather than expanding scope**.

---

## 9. Short checklist to carry into project planning

- [ ] Demonstrate that the cheap model/mode has a real bottleneck that KD could address.
- [ ] Choose teacher, student, targets, distillation scheme, and the `τ` / `α` validation plan.
- [ ] Test small mode **without** KD before attributing gains to KD.
- [ ] Measure accuracy **and actual hardware cost** for every operating point.
- [ ] Keep PEFT separate from compression: LoRA reduces trainable/update parameters, not base size.
- [ ] Use standard architectures/random search before expensive NAS.
- [ ] Define realistic distribution shifts and monitor confidence/performance under them.
- [ ] If models are updated, measure old-task retention and provide a rollback path.
- [ ] If data are moved to cloud/server, quantify actual bytes, timing, and privacy exposure.
- [ ] Keep fast inference, offline design/training, and slow maintenance as distinct workflows.

---

## 10. Lecture page index for follow-up

| Topic | PDF pages |
|---|---|
| Lecture storyline / four new questions | 2–3 |
| Teacher–student KD, dark knowledge, temperature, KD loss | 5–14 |
| Response / feature / relation transfer and teacher variants | 15–20 |
| KD data pipelines, unlabeled data, storage, failure modes | 21–26 |
| PEFT families, adapters, LoRA math and rank | 27–37 |
| LoRA merge vs swap, QLoRA, site routing, limitations | 38–45 |
| Hardware-aware NAS and when it is justified | 46–50 |
| Drift, recalibration, TENT, forgetting, mitigations | 51–58 |
| FL / LoRA updates, split inference, coordination, security, feature-size trap | 59–64 |
| Decision map, integrated edge design, key takeaways | 65–70 |

**One-sentence takeaway:** The best edge system is not merely a small neural network; it is a system with **an appropriate training/transfer strategy, an efficient hardware-validated runtime, and a credible plan for changing conditions over its lifetime**.
