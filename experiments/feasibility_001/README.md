# F001 - Feasibility of depth-adaptive shared-weight DeiT

This experiment tests whether one ImageNet-pretrained **DeiT-Tiny** can expose useful accuracy/compute operating points on **DermaMNIST at 224×224 input resolution**.

The operating points use the **same shared backbone**, truncated after transformer blocks **4, 8, and 12**. The full 12-block model is fine-tuned first. Its backbone is then frozen, and only the small depth-4 and depth-8 linear heads are trained. The depth-12 classifier is the classifier learned during full fine-tuning.

The official test split is deliberately not loaded. Train and validation are used for fitting and model selection; the test split remains untouched for a later pre-registered evaluation.

## Research question

Can a single shared DeiT-Tiny provide intermediate inference depths that meaningfully reduce compute and latency while retaining useful predictive performance?

## Setup

From the repository root:

```powershell
python -m pip install -r experiments/feasibility_001/requirements.txt
```

The first run downloads DermaMNIST-224 through the MedMNIST API and downloads the pretrained DeiT-Tiny weights through `timm`.

Neither dataset files nor model outputs are stored in Git.

The runner creates `--data-root` automatically before constructing either MedMNIST split.

## Run

The real F001 experiment used:

```powershell
python experiments/feasibility_001/run_experiment.py `
  --epochs 5 `
  --head-epochs 5 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda
```

Useful options include `--device cpu`, `--device cuda`, `--num-workers`, `--data-root`, `--checkpoint-root`, `--artifact-root`, and `--force-retrain`.

Existing checkpoints are reused unless `--force-retrain` is supplied.

The default output locations are:

```text
data/medmnist/                  downloaded dataset
checkpoints/feasibility_001/    full model and early-head checkpoints
artifacts/feasibility_001/      JSON/CSV results and Pareto plot
```

After both training phases, `depth_adaptive_model.pt` contains the shared backbone and both early heads as one deployable state dictionary. The phase checkpoints are kept separately so a failed or interrupted run can be resumed.

## Results

The completed 5+5 epoch feasibility run produced:

| Depth | Validation accuracy | Macro-F1 | fvcore FLOPs | Mean latency |
|---:|---:|---:|---:|---:|
| 4 | 74.48% | 0.3389 | 0.3792 G | 1.67 ms |
| 8 | 84.25% | 0.6655 | 0.7293 G | 3.25 ms |
| 12 | 84.45% | 0.6955 | 1.0794 G | 4.27 ms |

The main feasibility observation is that **depth 8 nearly matches the validation accuracy of the full depth-12 model while reducing measured latency and fvcore-reported computation**.

Depth 4 provides a much cheaper operating point, but with a substantially larger predictive-performance drop, especially in macro-F1.

These results are not treated as final model claims. F001 is a feasibility experiment used to decide whether depth-adaptive inference is worth developing further.

## Training observations

The full model reached its best validation accuracy before the final epoch:

```text
epoch 1: 0.7827
epoch 2: 0.8205
epoch 3: 0.8195
epoch 4: 0.8445
epoch 5: 0.7827
```

The runner retained the best validation checkpoint rather than blindly using the final epoch. The same model-selection logic applies to the early heads.

## Recorded outputs

`results.json` and `operating_points.csv` contain, for every depth:

- validation top-1 accuracy and macro-F1;
- fvcore-reported FLOPs using one fixed `1x3x224x224` input;
- latency statistics for batch size 1;
- the FLOP-counting convention;
- unsupported operators;
- uncalled modules.

`accuracy_vs_flops.png` is a validation-only plot. The runner reports the 12-block model as the static full-depth baseline and the three depths as the shared-weight adaptive family.

No test metric is produced.

## Methodological notes

`fvcore.nn.FlopCountAnalysis.total()` is used consistently for all three depths.

The result is recorded as **fvcore FLOPs**, not relabeled as MACs. Unsupported operators may therefore be omitted from the total. They are recorded in the output, with no invented handlers, so that a later report can decide whether the count is sufficiently complete for the final experiment.

The current unsupported operators include:

```text
aten::add
aten::scaled_dot_product_attention
aten::gelu
```

Latency is measured separately on the selected local device and should not be compared across machines without recording the hardware/software environment.

The uncalled-module list is also preserved. Transformer blocks after the selected cutoff are expected to be uncalled at depths 4 and 8 because those operating points intentionally truncate the shared backbone. Other entries may reflect identity/helper modules that are not invoked by the traced path.

The runner does not call `torch.jit.script` or `torch.jit.script_method`. `fvcore` FLOP analysis uses JIT tracing internally, so a Python 3.14 TorchScript warning from PyTorch or another dependency is dependency-internal and does not change this experiment's eager execution or recorded FLOP methodology. It is left visible rather than suppressed or worked around.

## Interpretation

F001 supports continuing with the adaptive-inference direction.

The next stage should focus on the **decision mechanism** rather than simply adding more fixed exits: when should an input stop at an intermediate depth, and how much compute can be saved without unacceptable loss in predictive performance?

The final project design will be refined after discussion with the course staff.
