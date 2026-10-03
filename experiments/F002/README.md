# F002 — OpenPOCUS architecture qualification and static reference

**Status:** T008 in progress. F002a ran on DAS-5 (2026-10-03) through §11 step 8
of [`docs/f002_methodology.md`](../../docs/f002_methodology.md). The
deterministic gates left two finalists with no automatic winner and no seed-43
trigger, so the one-backbone choice now goes to the human evidence gate
(Charalampos and Salo, §6). F002b (seed 2026) has not been trained. No model
has seen the T006 test partition.

The scientific contract is D024, `docs/f002_methodology.md` and
[`configs/f002_protocol.json`](../../configs/f002_protocol.json). This folder
holds the run commands; `src/f002_pipeline.py` holds the training, evaluation
and timing code on top of the reviewed `src/f002_protocol.py` and
`src/f002_flops.py` helpers.

## Commands

All commands run from the repository root and write to ignored
`artifacts/F002/`. Locally:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_f002_*.py'
.venv/bin/python experiments/F002/run_f002.py preflight     # §11 steps 1-3, no training
.venv/bin/python experiments/F002/run_f002.py references    # §11 step 4
```

On DAS-5, `das5_f002.sbatch` wraps any `run_f002.py` command on a GTX 1080 Ti
node (CUDA 12.6 module, the existing `myenv` conda environment, weight caches
in `/var/scratch/$USER/cache`). The F002a run used:

```bash
sbatch experiments/F002/das5_f002.sbatch preflight
sbatch experiments/F002/das5_f002.sbatch references
for c in mobilenet_v2 resnet18 deit_tiny; do
  sbatch experiments/F002/das5_f002.sbatch train --candidate $c --seed 42
done
for c in mobilenet_v2 resnet18 deit_tiny; do   # one node for every timing
  sbatch -w node201 --exclusive experiments/F002/das5_f002.sbatch benchmark --candidate $c --seed 42 --k 16
done
python experiments/F002/run_f002.py gate-k16 --seed 42
for c in mobilenet_v2 resnet18; do             # the two K16 finalists
  sbatch experiments/F002/das5_f002.sbatch evaluate --candidate $c --seed 42
  sbatch -w node201 --exclusive experiments/F002/das5_f002.sbatch benchmark --candidate $c --seed 42 --k 4 8
done
python experiments/F002/run_f002.py gate-final --seed 42
```

After the backbone decision is in the decision log, F002b is
`train --candidate <choice> --seed 2026 --stage f002b --decision D0xx`, then
`benchmark --stage f002b ... --k 16`. `train` refuses to overwrite a finished
run.

## Environment and implementation choices

- **Platform.** DAS-5 (UvA), `gpunode,GTX1080Ti` node class, NVIDIA GeForce GTX
  1080 Ti (compute 6.1), driver 580.95.05, CUDA runtime 12.6, cuDNN 9.10.2.
  Training ran on node201 and node205; every benchmark ran on node201 with
  `--exclusive`. Each artifact JSON records hostname, job ID, GPU, driver,
  package versions, protocol SHA-256 and source-file hashes.
- **Who ran it, and how long it took.** Salo ran F002a on DAS-5 under his
  account on 2026-10-03. SLURM accounting is disabled on DAS-5, so job
  durations come from the per-epoch timings in `attempt.json` and the SLURM log
  modification times. The three trainings ran in parallel on separate GPUs:
  MobileNetV2 took 27.7 min (8 epochs of about 207 s), DeiT-Tiny 13.7 min
  (5 × about 165 s) and ResNet-18 10.6 min (4 × about 160 s). Each epoch is
  about 19,700 decoded JPEG frames, and JPEG decoding with zero data-loader
  workers, as the protocol requires, sets the pace. Benchmarks ran one after
  another on node201: about 3 min per K16 benchmark, and 2 to 4.5 min for each
  finalist's K4 + K8 benchmark. The K-response evaluations took about 2 min
  each, and preflight + references about 2 min. The whole sequence, from the
  successful preflight to the last benchmark, took about 47 min of wall-clock
  time and about 1.3 GPU-hours. Setup before that was copying the 6.1 GB of
  OpenPOCUS frames to `/var/scratch`, adding timm, fvcore and scikit-learn to
  the existing `myenv`, and pre-downloading the pretrained weights.
- **Python deviation.** The protocol pins Python 3.14.3. The reused `myenv`
  has Python 3.12.14. torch 2.14.0+cu126, torchvision 0.29.0+cu126, timm
  1.0.30, fvcore 0.1.5.post20221221, Pillow 12.3.0, NumPy 2.5.3 and
  scikit-learn 1.9.1 match the pins exactly, and the strict FLOPs gate gave
  identical counts on DAS-5 and locally (macOS, Python 3.13.5, torch 2.14.1).
  The local checks don't replace the DAS-5 run.
- **Determinism probe.** Before training, all three candidates ran a fp16-AMP
  backward pass on a 1080 Ti under the §5 deterministic flags. None raised a
  missing-deterministic-kernel error, and two repeats gave bitwise-identical
  gradients. This was an interactive check; its output is not saved as an
  artifact.
- **Warm-up.** The 50 warm-up requests run before each of the 5 timed
  repeats, not once per path.
- **Validation inference.** Validation runs one sequence per forward in FP32
  eval mode.
- **Peak memory.** `peak_memory_mb` includes the 200 request tensors that the
  model-only path preloads to the GPU (about 1.9 GB at K16). It is not a
  per-model inference memory figure.

## Measured results (F002a, seed 42, validation partition)

Validation is 265 rows, which collapse to 254 duplicate-aware units (178
normal, 76 abnormal). All metrics below are duplicate-aware at threshold 0.5.
The validation partition also selected checkpoints and architectures, so
these numbers are development evidence and may be optimistic (§4).

References: all-normal macro-F1 0.412; the train-fitted device + zone
metadata model reaches macro-F1 0.482, accuracy 0.709, abnormal recall 0.079.

**K16 qualification.** All three runs were valid on the first attempt
(lr 1e-4, no rescue).

| Candidate | Best epoch / ran | Macro-F1 | Acc. | Normal rec. | Abnormal rec. | GFLOPs | Model-only median / p95 ms | Seq/s | Frames/s | Params | FP32 ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 5 / 8 | 0.867 | 0.894 | 0.955 | 0.750 | 5.11 | 11.35 / 11.36 | 88.1 | 1409 | 2.23 M | 9.2 MB |
| ResNet-18 | 1 / 4 | 0.782 | 0.815 | 0.860 | 0.711 | 29.17 | 7.53 / 7.57 | 132.8 | 2124 | 11.18 M | 44.8 MB |
| DeiT-Tiny | 2 / 5 | 0.710 | 0.799 | 0.966 | 0.408 | 20.29 | 12.74 / 12.77 (fused) | 78.5 | 1256 | 5.52 M | 22.2 MB |

DeiT fused versus unfused attention on the 200 requests had a maximum logit
difference of 9.5e-7 (limit 1e-5), so fused is its primary timing. Its unfused
timing was 13.52 ms (73.9 seq/s).

The K16 gate (`reduce_k16_candidates`) retained MobileNetV2 and ResNet-18.
MobileNetV2 eliminated DeiT-Tiny on F1 and FLOPs, and DeiT had no runtime
advantage. ResNet-18 survived because its model-only latency is below
0.90 × MobileNetV2's.

**K response.** These are the same K16 checkpoints evaluated at nested K, with
no retraining.

| Candidate | K | Macro-F1 | Acc. | Normal rec. | Abnormal rec. | GFLOPs | Model-only median ms | Seq/s | JPEG pipeline median ms |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 4 | 0.813 | 0.854 | 0.944 | 0.645 | 1.28 | 6.24 | 159.8 | 24.8 |
| MobileNetV2 | 8 | 0.858 | 0.886 | 0.949 | 0.737 | 2.55 | 6.35 | 157.4 | 44.4 |
| MobileNetV2 | 16 | 0.867 | 0.894 | 0.955 | 0.750 | 5.11 | 11.35 | 88.1 | 87.4 |
| ResNet-18 | 4 | 0.750 | 0.795 | 0.871 | 0.618 | 7.29 | 2.80 | 356.9 | 21.0 |
| ResNet-18 | 8 | 0.775 | 0.811 | 0.865 | 0.684 | 14.59 | 4.29 | 232.3 | 41.3 |
| ResNet-18 | 16 | 0.782 | 0.815 | 0.860 | 0.711 | 29.17 | 7.53 | 132.8 | 82.8 |

Final gate (`gate-final`): MobileNetV2 has the higher F1 and the lower FLOPs at
every K. ResNet-18 has a >10% model-only runtime advantage at every K, which
§6 counts as a runtime contradiction. Result: no automatic backbone, seed 43
not triggered (K16 F1 gap 0.086 ≥ 0.02), and the human gate is required.

## Interpretation (not measured claims)

- MobileNetV2 leads on every predictive metric and needs 5.7× fewer FLOPs than
  ResNet-18, yet runs slower on this GPU. MobileNetV2 at K4 and K8 costs about
  the same 6.3 ms, so its latency here tracks per-layer overhead, not FLOPs. I
  expect that's a property of depthwise convolutions on the 1080 Ti and would
  look different on edge hardware, but F002 didn't test other hardware.
- In the practical JPEG path, CPU decoding dominates (about 5 ms per frame).
  ResNet-18 is 3 to 5 ms faster per request at every K, against totals of
  21 to 87 ms.
- Both finalists improve monotonically from K4 to K16, with most of the gain
  between K4 and K8. That is a first, single-seed hint that variable K is a
  usable adaptive axis for T009. It is not a T009 result.
- Best epochs came early and validation F1 moved by up to 0.22 between
  consecutive epochs (DeiT: 0.71 → 0.49 → 0.68; MobileNetV2 +0.12 from epoch 3
  to 4). With 76 abnormal units, one unit is about 1.3
  points of abnormal recall, so small F1 differences are noise.
- These are annotation-agreement scores on a technical proxy dataset. They are
  not clinical diagnostic performance.

## Artifacts

`artifacts/F002/` (ignored; a copy also stays on DAS-5 at
`/var/scratch/bde2621/amea/artifacts/F002/`):

- `preflight/`: strict FLOPs totals and by-operator counts for every
  candidate and K, evaluation units, benchmark request IDs and environment.
- `references/`: all-normal and metadata-only validation references with row
  predictions and units.
- `f002a/<candidate>_seed42/`: `run.json` (status, selection, checkpoint hash,
  environment) and `lr1e-4/` with `checkpoint.pt`, `attempt.json` (epoch
  history and flags), per-epoch and selected validation predictions (row CSV
  with logits, sampled frame indices and files, plus aggregated units JSON).
  The two finalists add `k_response/` and `k_response.json`. Every timed K has
  a `benchmark_k*.json` with all raw request latencies, repeat wall-clock
  times, both paths and the DeiT equivalence result.
- `gates/`: `k16_seed42.json` (gate table) and `final_seed42.json`.
- `logs/`: SLURM output for every job.
