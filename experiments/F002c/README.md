# F002c — One-evening TinyViT exploration

**Question:** can one pretrained transformer approximately match the frozen
OpenPOCUS MobileNetV2 reference while offering useful later adaptive dimensions?
User-authorized exploratory experiment, 2026-10-04; branch
`experiment/F002c-transformer`. This is separate from the accepted F002 protocol.
F002a/F002b, D024/D025, the dataset, T006 labels/splits and evaluation units remain
unchanged. No formal architecture-fairness or statistical-winner claim is intended.

## Design chosen before training

One model: timm `tiny_vit_5m_224.dist_in22k_ft_in1k`, ImageNet-22K distilled then
ImageNet-1K fine-tuned weights. Its convolutional stem/first stage, local attention
and small hierarchical representation are a plausible transfer starting point on
small ultrasound data. This is a hypothesis, not evidence that it transfers better.
The pretrained classifier is replaced with mean pooling over K normalized 320-D
frame embeddings, dropout 0.1 and one binary linear head. Keep the pretrained
spatial pooling and final feature normalization. Mean pooling avoids another
small-data temporal learner and permits the same weights at K4/K8/K16.

The [timm model card](https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k)
reports 5.4M parameters including its ImageNet classifier and about 1.2 GMACs per
image. Our binary sequence model has 5,072,085 parameters. The published MAC count
is **not** a measured F002-style sequence FLOPs result and cannot be directly
compared with the baseline's 5.11 GFLOPs. TinyViT likely costs more arithmetic;
stage/depth exits and local-token processing are future possibilities, not adaptive
mechanisms implemented or validated here. The [original implementation](https://github.com/microsoft/Cream/tree/main/TinyViT)
and [timm source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/tiny_vit.py)
describe the hybrid architecture. No second architecture is included.

Weights are pinned to Hugging Face revision
`b46989c2a2c95e7612b919a6f5e2cd7dcf4f1271`, the snapshot downloaded for these checks.

Recipe (exact values also recorded in `run_f002c.py:CONFIG`):

- Seed 2026; full configuration K16; frozen deterministic nested frame selection.
- Full-square RGB 224 bicubic resize and ImageNet mean/std. Keep the ultrasound
  field of view instead of the pretrained natural-image center crop. Train-only
  horizontal flip probability 0.5, one shared decision across a sequence. No crop,
  rotation, mixup or colour jitter.
- AdamW, encoder LR 5e-5 and new head LR 5e-4, betas (0.9, 0.999), eps 1e-8;
  weight decay 0.01 excluding biases, normalization parameters and attention biases.
- Two-epoch linear warm-up, then cosine to 5% of each peak LR. At most 20 epochs;
  stop after 5 epochs without a better checkpoint, not before epoch 6.
- FP32, unfused attention, strict deterministic training flags. Microbatch 2
  sequences, accumulate 4 microbatches to 8. Gradient norm clipping at 1.
- Fine-tune all parameters, but freeze BatchNorm running statistics because nearby
  frames are correlated and each forward contains only two sequence rows. Affine
  parameters still learn. Encoder drop path 0.1 and head dropout 0.1.
- Unweighted binary cross-entropy by class. Reuse duplicate-recording row weights
  and the fixed optimizer-step denominator 8. Report global weighted epoch loss.
- Checkpoint by duplicate-aware validation macro-F1 at threshold **0.5**, then
  lower aggregated BCE, then earlier epoch. No threshold tuning tonight.
- Two data-loader workers by default; all train rows each epoch. No test images
  are inventoried, decoded or inferred. The imported frozen verifier hashes and
  validates test **manifest metadata** as part of verifying the shared foundation.

The image loader checks complete row identity against the frozen train/validation
manifests before resolving paths; changing a test row's partition string fails.
F002 data inventory, sampling, weighting, aggregation and metrics are imported
without modification. Outputs are confined to ignored `artifacts/F002c/`.

## Run

Use the existing project environment. On this Windows checkout the user explicitly
authorized `.venv` (Python 3.14.3, exact existing F002 package versions) after no
environment named `bde-project` was found. No new environment or dependency is needed.

```powershell
$env:HF_HOME = "$PWD/artifacts/F002c/cache/huggingface"
$env:TORCH_HOME = "$PWD/artifacts/F002c/cache/torch"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_f002*.py'
.\.venv\Scripts\python.exe experiments/F002c/run_f002c.py prepare-weights --device cpu
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe experiments/F002c/run_f002c.py smoke --run local_preflight --device cuda
```

The smoke verifies all train/validation retained inventories, performs two real
K16 forward/backward/Adam steps (including memory with optimizer state allocated),
checks 265 validation rows aggregate into 254 units and predicts two validation
rows. It discards its changed weights. Training always starts afresh from the
pretrained encoder with seed 2026. The same-device preflight runs automatically
before training. Local GPU evidence does not verify GTX 1080 Ti memory/runtime.

On DAS-5, use a separate F002c checkout/copy with the unchanged T006/raw data and
stable helpers available. The following defaults match the documented existing
F002 DAS environment; check the actual account and allocation before submitting.

```bash
export F002C_PYTHON=/var/scratch/$USER/miniconda3/envs/myenv/bin/python
export TORCH_HOME=/var/scratch/$USER/cache/f002c/torch
export HF_HOME=/var/scratch/$USER/cache/f002c/huggingface
# CPU-only weight preparation, before the compute job uses offline mode:
"$F002C_PYTHON" experiments/F002c/run_f002c.py prepare-weights --device cpu
mkdir -p artifacts/F002c/logs
sbatch experiments/F002c/das5_f002c.sbatch train
```

The wrapper requests one GTX 1080 Ti, four CPUs, 32 GB host RAM and two hours;
runs the relevant tests on the allocated node; and runs CUDA preflight before
training. Estimate 1–2 hours before measuring actual epoch time, plus queue delay.
It reuses an existing environment and never installs packages. A finished or
partial run directory cannot be overwritten or implicitly resumed. No automatic
rescue is attempted: at most one optimization-motivated change can be documented
after inspecting a clear failure; low validation F1 alone is not a rescue trigger.

If selected K16 macro-F1 is at least 0.84 and abnormal recall at least 0.65, the
runner automatically evaluates that same checkpoint at K4/K8/K16. This is a
practical trigger to collect cheap descriptive evidence, not a new winner rule;
recall 0.65 is below the reference and must be reported as such. Manual replay:

```bash
"$F002C_PYTHON" experiments/F002c/run_f002c.py evaluate --run tinyvit5m_seed2026
```

Evaluation verifies checkpoint, configuration, frozen-input and execution-source
hashes before loading. No K-specific retraining is available. Strict FLOPs and
runtime comparison remain later work if predictive evidence is promising.

## Evidence and interpretation

Frozen reference (F002b, not changed): macro-F1 **0.859**, accuracy **0.886**,
normal/abnormal recall **0.944/0.750**, K16 **5.11 GFLOPs**.

Local verification on 2026-10-04: 36 existing F002 tests and 12 F002c tests passed.
The pretrained K16 CUDA smoke passed on the RTX 4060 Laptop GPU (8 GiB): two
real forward/backward/Adam steps used 3,122 MiB peak allocated and 3,390 MiB peak
reserved memory. All 1,232 train and 265 validation inventories verified; validation
aggregation retained 254 units. No test-image inference occurred. Source/data
hashes confirmed the frozen inputs were unchanged. These are engineering checks,
not predictive results or GTX 1080 Ti measurements.

**DAS status:** not submitted. This machine has no configured DAS SSH alias or key;
host/account/authentication instructions are pending from the user. Local F002b
checkpoint is absent, so only its recorded reference metrics are used here.

No full F002c training result exists yet. Approximately 0.859 with healthy abnormal
recall would be strong descriptive evidence; 0.84–0.859 may justify an adaptive
follow-up; clearly below 0.81 suggests closing this direction. Validation also
selects the checkpoint, so observed scores are development evidence only.

Raw evidence stays under `artifacts/F002c/`: `smoke.json`, test logs,
`tinyvit5m_seed2026/run.json` (configuration, environment/source hashes, epoch
history, selection, memory and elapsed time), the exact `initial_state.pt`,
selected `checkpoint.pt`, per-epoch/selected row predictions and aggregated-unit
JSON, optional `k_response.json`, and scheduler logs. Frozen inputs are hashed
before and after execution. The frozen F002b checkpoint is checked against its
known hash if present; its absence is not represented as an independent recheck.
