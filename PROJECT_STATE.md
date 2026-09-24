# Current Phase

F001, F001b, and the F001c per-input characterization are complete. Scenario selection and deployable adaptive-policy design remain open.

## Leading hypothesis

Resource-aware adaptive inference for medical image classification using a shared-weight Vision Transformer; whether routing should depend on each input remains unproven.

## Feasibility experiments

- **F001:** ImageNet-pretrained DeiT-Tiny on DermaMNIST-224, with shared-backbone operating points at depths 4, 8, and 12. The official test split was not loaded.
- **F001b:** Dense characterization of depths 4–12 using the F001 backbone. The F001 depth-4/8 heads and depth-12 classifier were reused; six intermediate heads were trained with the backbone frozen. Train/validation only; test untouched. Validation accuracy peaks at depth 9 (85.14%), then varies non-monotonically; macro-F1 peaks at depth 10 (0.7061). Measured **fvcore FLOPs** range from 0.3792G (depth 4) to 1.0794G (depth 12).
- **F001c:** Analysis-only validation inference through all existing depth heads (1,003 examples × 9 depths); no weights trained and F001/F001b inputs verified unchanged by SHA-256. At 9→12, 28 errors are rescued and 35 correct predictions regress (net −7). Depth 12 is the first correct depth for 5 samples; the all-depth hindsight oracle covers 92.2% of samples, explicitly as an unreachable upper bound. Confidence/entropy patterns are descriptive and conditional bins are sparse/undefined at confident shallow predictions, so no routing signal is validated.
- F001c source: `experiments/F001c_per_input_depth_utility/`; generated outputs are ignored under `artifacts/F001c_per_input_depth_utility/`.

## Not yet decided

- Final medical scenario and scenario-specific utility/constraints
- Runtime adaptivity signal and deployable policy/controller
- Target device, latency/energy budgets, and evaluation protocol
- Whether input-dependent exits outperform a fixed externally selected depth under the chosen scenario
- Pre-specified use of the reserved official test set

## Current objective

Discuss F001b/F001c trade-offs and interpretation with Dolly/TA, then lock the scenario and evaluation objective before designing a controller. Keep the official test set reserved for later, pre-specified final evaluation.

## Plan

Review F001b operating points and F001c rescue/oracle/uncertainty evidence
→ discuss scenario and trade-offs with Dolly/TA
→ lock scenario, utility, constraints, and test protocol
→ design and evaluate a deployable policy against fixed-depth baselines
