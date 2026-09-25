# Current Phase

F001, F001b, and F001c are completed pilot feasibility studies. The leading scenario is now provisionally prehospital/ambulance POCUS; exact task, dataset fit, and architecture are not locked.

## Leading hypothesis

Explore resource-aware inference for a provisional prehospital/ambulance POCUS scenario. A public POCUS dataset may serve as a technical proxy and may not represent trauma or ambulance patients exactly. The prediction task, dataset suitability, and architecture remain open. ViTs and optimization-based controllers are research possibilities, not current commitments.

## Feasibility experiments

- **F001:** ImageNet-pretrained DeiT-Tiny on DermaMNIST-224, with shared-backbone operating points at depths 4, 8, and 12. The official test split was not loaded.
- **F001b:** Dense characterization of depths 4–12 using the F001 backbone. The F001 depth-4/8 heads and depth-12 classifier were reused; six intermediate heads were trained with the backbone frozen. Train/validation only; test untouched. Validation accuracy peaks at depth 9 (85.14%), then varies non-monotonically; macro-F1 peaks at depth 10 (0.7061). Measured **fvcore FLOPs** range from 0.3792G (depth 4) to 1.0794G (depth 12).
- **F001c:** Analysis-only validation inference through all existing depth heads (1,003 examples × 9 depths); no weights trained and F001/F001b inputs verified unchanged by SHA-256. At 9→12, 28 errors are rescued and 35 correct predictions regress (net −7). Depth 12 is the first correct depth for 5 samples; the all-depth hindsight oracle covers 92.2% of samples, explicitly as an unreachable upper bound. Confidence/entropy patterns are descriptive and conditional bins are sparse/undefined at confident shallow predictions, so no routing signal is validated.
- F001c source: `experiments/F001c_per_input_depth_utility/`; generated outputs are ignored under `artifacts/F001c_per_input_depth_utility/`.

These DermaMNIST/DeiT experiments remain historical feasibility pilots. They do not lock the POCUS scenario, prediction task, or final architecture.

## Not yet decided

- Exact POCUS prediction task supported by the selected public dataset
- Whether and how the public POCUS data can serve as a technical proxy for prehospital/ambulance use; it may not represent trauma/ambulance patients exactly
- Final medical scenario and scenario-specific utility/constraints
- Model architecture (a ViT is a research possibility, not a commitment)
- Runtime adaptivity signal and deployable policy/controller
- Whether to pursue an optimization-based controller (research possibility, not a commitment)
- Target device, latency/energy budgets, and evaluation protocol
- Whether input-dependent exits outperform a fixed externally selected depth under the chosen scenario
- Pre-specified use of the reserved official test set

## Current objective

Audit the public POCUS dataset, determine the dataset-supported prediction task, and record Dolly/TA feedback (currently pending) before locking the scenario and evaluation objective. Keep F001/F001b/F001c as pilot evidence and the DermaMNIST official test set untouched; define a pre-specified evaluation protocol for the selected POCUS task.

## Plan

Integrate and audit the POCUS dataset (T001–T002)
→ record feedback and lock the dataset-supported task (T004–T005)
→ specify and run the F002 static baseline (T006–T008)
→ characterize adaptive options and construct shared operating points (T009–T010)
→ consider extensions only after core course work
