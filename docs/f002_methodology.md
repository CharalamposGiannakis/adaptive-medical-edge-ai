# F002 methodology — T007 preregistration

**Status:** accepted after independent review and PR #22 merge (T007 complete;
issue #9 closed). This is the pre-registered experimental contract for F002a architecture qualification and
F002b frozen static baseline. `configs/f002_protocol.json` fixes the values;
`src/f002_protocol.py` and `src/f002_flops.py` implement the critical rules.
The merged T006 manifests are frozen inputs. F002 training must not silently
change the cohort, labels, split, sampling, or evaluation units. Once the first
F002 architecture training run starts, any methodology change that can affect
comparison requires documentation, a decision-log amendment, and rerunning
**every affected candidate** under the changed protocol. Never change a rule
after seeing which architecture it favors. Course-derived
`adaptivity_project_notes.md` is advisory; accepted decisions and this reviewed
protocol govern execution.

## 1. Input foundation and test protection

Use only the T006 train/validation/test manifests and verified shared-recording
edge table. `configs/f002_protocol.json` pins SHA-256 fingerprints for all
tracked T006 files (the split-groups hash uses T006's canonical LF convention)
and the T004 manifest provenance recorded by T006. Verify them before each run.
The retained **1,762** sequences are 1,225 normal and 537 abnormal; the fixed
seed-42 split has **1,232/265/265** train/validation/test rows across 211
retained dependency groups. Every row has at least 17 retained frames. These
are T006 snapshot measurements, not a new data selection. Known groups are
disjoint, but biological patient independence remains unverified.

Train gradients on **train** only; choose checkpoints and compare architectures
on **validation** only. The test manifest may be fingerprinted or inspected as
metadata to check provenance, but no F002a or F002b model inference, threshold
fit, architecture comparison, or checkpoint choice uses test during T008. No
new split, model-specific exclusion, or source-label/raw-data change is allowed.
Final held-out evaluation is a later preregistered project step.

## 2. F002a candidates and common sequence model

Use exactly three ImageNet-pretrained frame encoders, with the installed and
preregistered versions below. `timm` is already used by F001 and installed in
the existing local environment; no weight is downloaded in T007.

| Candidate | Constructor and weight identifier | Feature before original classifier |
| --- | --- | --- |
| MobileNetV2 | `torchvision.models.mobilenet_v2`, `MobileNet_V2_Weights.IMAGENET1K_V2` | 1,280-dimensional pooled vector; replace `classifier` with identity |
| ResNet-18 | `torchvision.models.resnet18`, `ResNet18_Weights.IMAGENET1K_V1` | 512-dimensional pooled vector; replace `fc` with identity |
| DeiT-Tiny | `timm.create_model('deit_tiny_patch16_224.fb_in1k', pretrained=True, num_classes=0)` | 192-dimensional final normalized CLS vector |

Pinned API/runtime versions: Python 3.14.3; PyTorch 2.14.0+cu126;
torchvision 0.29.0+cu126; timm 1.0.30; fvcore
0.1.5.post20221221; Pillow 12.3.0; NumPy 2.5.3; scikit-learn 1.9.1.
Record the exact wheel/build, platform, CUDA driver and hardware for T008 runs.
An equivalent platform wheel must preserve these model/weight identities and
pass the protocol checks. DeiT uses `fused_attn=False` for canonical training,
validation/selection predictions and strict FLOPs tracing, exposing QK,
softmax and attention-value matmuls. The same checkpoint uses the supported
optimized `fused_attn=True` path for **primary runtime timing only after** the
equivalence gate in §9; unfused timing remains a secondary diagnostic. If a
pinned implementation/weight/preprocessing contract is
unavailable, stop before training and amend the method.

For each K-frame zone sequence, apply **the same pretrained frame encoder** to
each frame, take its vector before the original ImageNet classifier, arithmetic
mean the K feature vectors, and apply one new `Linear(feature_dim, 1)` head
with bias. Its output is one binary logit; sigmoid yields the sequence
probability. Fine-tune encoder and head together. No temporal transformer,
LSTM, attention/gated pooling, early exit, token pruning, or model-specific
adaptive module enters F002. Replacing MobileNetV2's entire original
`classifier` also removes its `Dropout(0.2)`; all three use only the common new
linear binary sequence head.

## 3. Input processing and deterministic frame selection

Read the exact ordered retained JPEG paths through `cohort_loader.py`, which
checks the T006 path hash. Decode using Pillow, convert to RGB, resize the
entire 512×512 square to 224×224 with Pillow bilinear interpolation, then use
`torchvision.transforms.functional.to_tensor` (float32 in [0,1]) and normalize
all candidates with ImageNet mean **[0.485, 0.456, 0.406]** and standard
deviation **[0.229, 0.224, 0.225]**. No crop, padding, stochastic augmentation,
color jitter, MixUp, CutMix, or model-specific transform. If a selected
pretrained implementation cannot accept this common contract, stop.

The initial common screen trains/evaluates at **K=16**. The later small response
screen evaluates selected **same K16 checkpoints** at **K=4, 8, 16**; it does
not train K-specific models. `nested_frame_indices(N, K)` is zero-based and
depends only on the ordered sequence length. Begin with four sorted inclusive
anchors `floor(j*(N-1)/3)` for `j=0,1,2,3`. Until K positions exist, find the
widest gap between selected indices (choose the leftmost on ties) and add its
lower integer midpoint `floor((left+right)/2)`. Return sorted positions. Thus
K4 is a strict subset of K8, K8 of K16; all are unique and span the first to
last retained frame. Integer arithmetic is stable across operating systems.
No padding, frame repetition, stochastic draw, architecture or label input is
used. Save both selected zero-based indices and retained frame identities.

## 4. Same-recording components, loss and metrics

Build connected components from the **verified** T006
`shared_recording_edges.csv`, including excluded endpoints for provenance.
Intersect each component with retained rows. Assert retained aliases share
binary target, dependency group and partition, and that no T006 conflict
component enters retention. Missing-label aliases add no target, no vote, and
no recovered label. If graph semantics are ambiguous, stop. The tested mapping
currently yields 1,208 train, 254 validation and 265 test evaluation units;
the latter count is metadata only, with no test predictions.

Keep every retained row for training. For each retained component of `n`
aliases in train, weight each row `1/n`; singleton rows weigh 1. Use
`BCEWithLogitsLoss(reduction='none')`. For **each optimizer step**, including
the final partial accumulation window, backpropagate
`sum_effective_batch(weight_i * BCE_i) / 8` with the fixed effective-batch
denominator **8**. With microbatch 2 / accumulation 4, backpropagate
`sum_micro(weight_i * BCE_i) / 8` from each microbatch, then step after the
window. Never divide a step by that batch's weight sum: doing so changes a
row's gradient coefficient according to which other aliases were shuffled
beside it. The component weights still sum to 1 over the epoch. Separately,
report the interpretable **global epoch weighted mean**
`sum_epoch(weight_i * BCE_i) / sum_epoch(weight_i)`; the rescue check uses this
reported epoch loss, never the fixed-denominator optimization scalar. Keep
all retained aliases and use no class weights by default. The tested helper
`optimization_loss` fixes the denominator; `reported_weighted_train_loss`
implements the separate epoch statistic.

On validation, save every row's logit/probability and metadata. Average alias
**probabilities** within each retained same-recording component; each such
component contributes one vote. Assert target agreement. Compute duplicate
aware accuracy and macro-F1 (primary), normal and abnormal recall, confusion
counts, and mean binary cross-entropy of the aggregated probabilities. Also
report equivalent raw row-level metrics as diagnostics. The fixed probability
threshold is **0.5: predict abnormal iff probability >= 0.5** for all
validation metrics, checkpoint selection, architecture screens and F002b.
There is no model-specific threshold tuning.
These metrics measure OpenPOCUS annotation agreement, not clinical diagnostic
performance. The same validation partition is reused for checkpoint
selection, architecture qualification and F002b checkpoint selection. Its
performance is therefore development/selection evidence and may be optimistic;
final generalization claims require later evaluation on the untouched test
partition under a separate approved protocol.

Report an all-normal validation reference at the same component units. Also
fit one cheap metadata-only shortcut diagnostic on train: `OneHotEncoder`
fitted only on training device code and zone ID (`handle_unknown='ignore'`),
then `LogisticRegression(C=1, max_iter=1000, solver='lbfgs', penalty='l2',
fit_intercept=True, class_weight=None, random_state=42)` with the same
duplicate training sample weights. Evaluate it on validation at threshold 0.5 with the same
component aggregation. It is a diagnostic of non-image signal, not an
architecture candidate; it receives no test predictions in T008.

## 5. Common bounded training and checkpoint rule

Initialize the three specified pretrained encoders and fresh binary heads.
Use full fine-tuning, AdamW over all trainable parameters (betas 0.9/0.999,
epsilon 1e-8), learning rate
**1e-4**, weight decay **1e-4**, no scheduler, at most **12** completed epochs,
sequence microbatch **2**, gradient accumulation **4**, effective batch **8**.
Flatten `B×K` frames internally for the encoder, then restore B/K for mean
pooling. The same physical/effective batch applies to all candidates, avoiding
architecture-specific BatchNorm exposure. During full fine-tuning, BatchNorm
layers stay in normal train mode and update running statistics; validation and
inference use eval mode. Use **no gradient clipping**. Shuffle train
deterministically
using one generator seeded by run seed; use zero data-loader workers. Seed
Python, NumPy, PyTorch and CUDA; set cuDNN deterministic true, benchmark false,
`torch.use_deterministic_algorithms(True)`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Disable TF32
for CUDA matmul and cuDNN. If a deterministic
operation is unavailable, stop and document it. On CUDA use float16 autocast
and GradScaler for **all** candidates; on CPU use FP32. Validation and reported
static model semantics are FP32. Save actual environment and determinism flags.

After each epoch compute duplicate-aware validation macro-F1 and duplicate
aware validation loss. Select the checkpoint by highest macro-F1, then lower
validation loss on a tie, then earlier epoch on an exact tie. A tie in F1 is an
exact equality of integer confusion-derived scores; loss is compared at its
recorded full precision. Early stop after **3 consecutive completed epochs**
without a checkpoint improvement by this ordered rule. Keep all epoch history.
No architecture-specific sweep is permitted.

One optimization-failure rescue is allowed: if any nonfinite loss occurs, or
by completion of epoch 4 the best **reported global weighted epoch train loss**
in epochs 2–4 has not fallen at least 1% below the reported epoch-1 loss,
rerun that candidate once at **1e-5** with all other settings unchanged.
Record trigger and rescue. Poor validation
performance alone never triggers rescue. If the rescue also fails, mark the
candidate invalid; no second rescue. If memory truly makes the common batch
impossible, stop before continuing comparisons and amend the protocol for
**all** candidates, including reruns as required by the freeze rule.

## 6. F002a reduction and one-backbone decision

At K16 a candidate is valid only if its duplicate-aware validation macro-F1
is **strictly above** the all-normal reference and both class recalls are
positive. If all three fail, stop F002 for review; do not add a fourth model.
Remove invalid candidates. The initial reduction uses the **primary model-only
CUDA** K16 benchmark from §9. Candidate A may eliminate B only when A's K16
macro-F1 is at least B's, A's sequence FLOPs are at most B's, at least one of
those comparisons is strict, **and B has no material runtime advantage over
A**. B has such an advantage if its model-only median latency is
`< 0.90 × A latency` **or** its model-only sequence throughput is
`> 1.10 × A throughput`. Practical JPEG pipeline measurements support
interpretation but do not trigger automatic elimination.

If more than two valid, non-eliminated candidates remain, fill two slots
deterministically. First choose highest macro-F1, breaking ties by lower FLOPs,
then lower model-only median latency, then candidate ID. Among remaining
candidates, provisionally choose lowest FLOPs, breaking ties by higher F1,
then lower latency, then ID. If any other remaining candidate has the defined
`>10%` runtime advantage over that provisional second choice, use a runtime
challenger instead. With multiple challengers choose lower model-only median
latency, then higher model-only sequence throughput, lower FLOPs, higher F1,
then ID. `reduce_k16_candidates` implements this exact cap of two. Save the
complete gate table, including size and training stability; its helper also
rejects CPU, mismatched GPU models/node classes or benchmark flags. Use no weighted
score.

Evaluate the at-most-two retained selected **K16 checkpoints** on validation
at K4/8/16 with the nested sampler. Preserve row and component predictions,
quality, complete-sequence FLOPs, runtime, throughput and size for each K.
If only one candidate is valid after the gate, it is selected after its K
response is recorded. For two candidates, select one automatically only when
it has macro-F1 no worse and FLOPs no higher at **each corresponding K**, at
least one strict advantage, and no material runtime contradiction. A runtime
contradiction means the other candidate has the same defined >10% model-only
runtime advantage at any K under the same benchmark platform/protocol.
Otherwise present the complete prespecified evidence table to Charalampos
and Salom: K16 macro-F1/accuracy and both recalls; K4→8→16 change; FLOPs;
model-only median/p95 latency; model-only sequence/frame throughput;
checkpoint size; training stability; and implementation/deployment complexity.
They record one-backbone
choice and reasoning in the decision log before F002b. No score composed from
new weights/metrics may be invented after results.

If exactly two finalists remain ambiguous after the K response, their K16
macro-F1 differs by **<0.02**, and neither has clear dominance under the rule
above, run exactly one confirmatory seed **43** for **both** finalists under
the complete same K16 training, rescue, validity and selected-checkpoint
K4/8/16 response, FLOPs/runtime and artifact protocol. Seed 43 is **robustness
confirmation**: retain
seed-42 and seed-43 evidence separately and present both side-by-side at the
existing Charalampos-and-Salom human evidence gate. Do **not** average seeds
into a new winner score or run another automatic tournament. The human gate
may use only the prespecified dimensions listed above: macro-F1/accuracy,
both recalls, K response, FLOPs, model-only latency, both throughput units,
size, training stability and implementation/deployment complexity. Record the
one-backbone decision and rationale before F002b. If either seed-43 run fails
validity even after its single allowed rescue, record the failure and send the
complete evidence to the human gate; never substitute another seed. Do not
spend seed 43 when a backbone is already clear. No extra seeds or K-specific
retraining.

## 7. F002b frozen static reference

After the one backbone is recorded, train **one fresh instance** with seed
**2026** at K16 under the same preprocessing, sampler, model, weighting,
training and checkpoint protocol. Train rows supply gradients; validation
rows select the checkpoint by duplicate-aware macro-F1. Threshold stays 0.5.
The exploratory F002a checkpoint never becomes the static reference by
convenience. Save immutable checkpoint, weights/config/protocol hashes and
validation evidence under ignored `artifacts/F002/`; T009 consumes this
reference unchanged. T008 generates **no test-set model inference**.
Apply the same optimization-failure rescue and basic all-normal/both-recalls
validity gate to seed 2026. If optimization fails, permit only the one
preregistered 1e-5 rescue. If it still fails, or its selected checkpoint fails
the validity gate, **stop F002b** and document the failure. Do not substitute
the seed-42 exploratory checkpoint, invent seed 2027/another seed, or silently
choose a second architecture. A project-level methodology decision is needed
before proceeding.

## 8. Complete-sequence FLOPs convention and coverage gate

Use **only** `fvcore.nn.FlopCountAnalysis` on the actual batch-one
`[1,K,3,224,224]` sequence wrapper: K frame encodes + mean feature pool +
binary head. Following fvcore, one fused multiply-add is **one reported
FLOP** (not two). The custom handlers count elementwise add/multiply/HardTanh/
GELU/**ReLU/clamp** as one per output element, softmax as five per output
element, mean as
one per input element, and max pooling as output elements × (kernel area−1)
comparisons. These are fixed analytical conventions, not measured energy or
latency. Shape/identity operations (reshape, view, transpose) are bookkeeping.
Use the same handlers across all three models and every K; never mix published
per-image counts or a separate manual estimator.

`src/f002_flops.py` fails if **any** operator is unsupported or if fvcore would
silently ignore an operator outside the explicit reviewed bookkeeping
allowlist. The exact qualified names currently allowed are `aten::Int`,
`aten::cat`, `aten::dropout` (identity in eval), `aten::expand`,
`aten::flatten`, `aten::permute`, `aten::reshape`, `aten::select`,
`aten::size`, `aten::slice`, `aten::squeeze`, `aten::transpose`,
`aten::unbind`, `prim::Constant`, `prim::ListConstruct`,
`prim::ListUnpack`, and `prim::NumToTensor`. A new traced ignored operation
must be reviewed and counted if arithmetic; it cannot silently inherit
fvcore's broad ignored set. ReLU, ReLU_, clamp and clamp_ have explicit
elementwise handlers. DeiT fused attention is disabled so attention QK and
attention-value matmuls are visible;
the checker additionally requires positive matmul and softmax counts for the
transformer, plus mean pool and a **module-specific positive binary-head
count** for every candidate. The no-training test traces all three at K4,
K8 and K16 with zero unsupported
and zero unapproved ignored operators, positive ResNet ReLU, and the expected
head contribution in the pinned local environment. T008 reruns this gate in
its actual
environment **before training** and saves total and by-operator counts. If
coverage becomes incomplete, stop rather than reporting understated FLOPs.
Older F001 fvcore transformer numbers are historical and not substituted into
this F002 comparison. FLOPs are an explicit analytical arithmetic convention;
runtime is measured with a supported optimized backend. The two quantities
must not be conflated.

## 9. Runtime, throughput, footprint

The **primary comparison platform is a CUDA GPU on DAS-5**. Every candidate/K
point in one comparison gate/table must run on the same GPU model and node
class. The first accepted runtime measurement fixes that GPU model for the
gate; record exact hostname/node, GPU model, driver, CUDA/PyTorch versions,
and relevant CPU threads. Do not fall back to CPU without a methodology
amendment. Use the same software, precision and flags across candidates.

The **primary automatic runtime gate** is model-only CUDA inference, one
zone-sequence request at a time with K frames batched inside the encoder.
Use `model.eval()`, `torch.inference_mode()`, FP32, optimized normal
PyTorch/cuDNN kernels and CUDA synchronization around each request. Set
`torch.use_deterministic_algorithms(False)`,
`torch.backends.cudnn.deterministic=False`,
`torch.backends.cudnn.benchmark=True`,
`torch.backends.cuda.matmul.allow_tf32=False`, and
`torch.backends.cudnn.allow_tf32=False` **for benchmarking only**. Training
and validation retain §5's deterministic flags. Warm-up precedes timing so
cuDNN/kernel-selection work is excluded. Do not force CNNs away from normal
optimized kernels.

Fix request IDs from the T006 validation manifest: the **first 200 retained
validation rows in manifest order** form the timed list; its first 50 rows
form the warm-up list. Use the same IDs and order for every candidate, repeat
and equivalent K comparison. `benchmark_request_ids` implements this. For
each candidate/K/path, run those **50 warm-up requests**, then the **200
timed requests**, for **5 repeats**.
Preserve every request latency and each repeat wall-clock total. Report median
of the five repeat medians and median of the five repeat p95 values, with raw
timings retained. Compute sequence predictions/s as 200 divided by the
repeat's actual synchronized wall-clock seconds; frames/s = 200×K divided by
the same seconds. Report the median of five repeat rates, not 1/median
latency. If these counts prove infeasible, amend the protocol once for all
candidates before comparison and record why.

Separate (a) model-only inference on already prepared tensors from (b)
**required supporting** practical JPEG decode + common preprocessing +
transfer + model path. Before the practical benchmark warm-up, make one
complete **untimed** read/decode pass over the 200 request list to avoid
measuring mainly cold filesystem cache. This is a warm-cache pipeline result,
not storage cold-start performance. Practical pipeline measurements, p95
latency and frames/s support interpretation but do not independently trigger
automatic architecture elimination. Record CPU/GPU model, execution device,
memory, OS, Python/PyTorch/CUDA/driver versions, hostname/node class,
benchmark flags, threads, precision, batch semantics, transfer boundary and
timing API. This machine is a comparison platform, not a claimed ambulance
device. Report parameter
count and actual serialized **FP32** checkpoint/model size. Peak inference
memory is optional if reliable. FLOPs do not imply latency or energy.

For DeiT, keep canonical `fused_attn=False` for training, selection predictions
and FLOPs. The **primary DeiT runtime** uses normal supported timm
`fused_attn=True` on the **identical checkpoint weights**. Before timing, on
all 200 fixed requests **separately at each benchmarked K4/K8/K16**, compare
fused and unfused FP32 logits; require
`max_abs_logit_difference <= 1e-5`. Preserve the result and both paths'
summaries, with unfused timing secondary. If equivalence fails, mark fused
runtime invalid, report the deviation, and use unfused timing only after
documenting its methodology consequence. Never silently accept fused results.

## 10. T008 artifact contract and stop conditions

For every candidate/run retain under ignored `artifacts/F002/`: architecture,
exact model/weight IDs and package/build versions; protocol SHA-256 and T006
manifest hashes; run seed and selected checkpoint SHA-256; epoch-by-epoch
train loss, duplicate-aware validation loss/metrics and raw row diagnostics;
final row logits/probabilities and aggregated component probabilities;
exam/zone, target, dependency group, component ID, device, K and sampled
indices/path identities; optimizer/rescue status; complete FLOPs totals and
operator breakdown; raw latency/throughput and both pipeline summaries;
parameter count/FP32 serialized size; all-normal and metadata-only reference
outputs; fixed benchmark request IDs/order; per-K DeiT fused/unfused
equivalence maxima and primary fused plus secondary unfused runtime summaries;
benchmark flags and DAS-5 node/GPU/driver/software provenance. Keep
checkpoint and result files ignored; track experiment README/config and a
measured-versus-interpretation account in the experiment log. Do not infer
clinical effectiveness from annotation agreement.

Stop before training if any T006 hash or row invariant fails, K16 cannot be
sampled without repetition, the shared-recording graph is ambiguous, common
224-square preprocessing fails a candidate, exact DeiT implementation/weights
cannot be reproduced, the common training recipe cannot run, or strict
complete-sequence FLOPs coverage fails. Stop F002 if all candidates fail the
validity gate. Document exact failure rather than silently changing the cohort
or methodology.

F002 excludes knowledge/self distillation, LoRA, adapters, hardware-aware NAS,
online/test-time or federated adaptation, split/cloud inference and
multi-teacher methods. T007/T008 implement no normalized budget→K mapping,
confidence routing, learned controller, early exit, stay/switch logic or
resource optimizer. Operating modes are measured before a later controller.

## 11. Ordered T008 handoff

1. Verify every T006 fingerprint and retained-frame invariant; load the
   reviewed `configs/f002_protocol.json` and record its SHA-256.
2. Reuse the tested nested sampler and duplicate component/weighting helpers;
   assert no conflicting T006 component enters retained units.
3. Build the common preprocessing and sequence wrapper with exact pretrained
   identifiers. Run strict FLOPs coverage for all candidates/K **without
   training**; stop on any unsupported or unapproved ignored operation.
4. Run the all-normal and train-fitted metadata-only validation diagnostics.
5. Train MobileNetV2, ResNet-18 and DeiT-Tiny at K16/seed42 under the same
   recipe; save every run artifact, including any bounded rescue.
6. Benchmark the fixed 200 validation requests on the locked DAS-5 CUDA
   platform, with the required DeiT equivalence gate and practical path.
   Apply the validity and deterministic runtime-aware reduction gates; stop
   if all fail.
7. Evaluate at most two selected checkpoints at K4/8/16 without retraining;
   save component and row predictions plus cost/runtime evidence.
8. Apply the final dominance/runtime gate. If its exact ambiguity trigger
   holds, run both seed43 confirmations and K responses; take **both seeds**
   to the human evidence gate without averaging or automatic reselection.
9. Present the prespecified table to Charalampos and Salom when needed; record
   the selected backbone and why alternatives were rejected before F002b.
10. Train a fresh seed2026 K16 F002b instance with the same rescue/validity
    rules. Stop for a project-level decision if it fails. Otherwise freeze its
    checkpoint/config/hashes and hand it to T009. Generate no test-set
    predictions in T008.
