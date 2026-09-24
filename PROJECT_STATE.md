# Current Phase

F001b depth characterization complete; scenario and adaptive-policy selection remain open.

## Leading hypothesis

Resource-aware adaptive inference for medical image classification using a shared-weight Vision Transformer.

## Feasibility experiments

- **F001:** ImageNet-pretrained DeiT-Tiny on DermaMNIST-224, with shared-backbone operating points at depths 4, 8, and 12. The official test split was not loaded.
- **F001b:** Dense characterization of depths 4 through 12 using the same fine-tuned F001 backbone. The F001 heads at depths 4 and 8 and the full-depth classifier were reused; six intermediate heads were trained with the backbone frozen. Train and validation splits only; the official test split remains untouched.
- Measured fvcore FLOPs increase from 0.3792G at depth 4 to 1.0794G at depth 12. Validation accuracy is non-monotonic: it peaks at 85.14% at depth 9, then falls at depths 10 and 11, with a small rise at depth 12. Macro-F1 peaks at depth 10 (0.7061).
- F001b source and methodology: `experiments/F001b_depth_sweep/`. Results and plots: ignored `artifacts/F001b_depth_sweep/`.

## Not yet decided

- Final medical scenario
- Runtime adaptivity signal and policy/controller
- Global utility and deployment constraints
- Whether to use an input-dependent exit policy or externally selected depth

## Current objective

Interpret the measured depth sweep and decide, with course staff, whether depth should remain an adaptive dimension and how future scenario-specific evaluation should be defined. Keep the official test set reserved for a later, pre-specified final evaluation.

## Plan

Review F001b results
→ discuss the scenario and trade-offs with Dolly/TA
→ lock the project scenario and evaluation objective
→ design the adaptive controller and final evaluation
