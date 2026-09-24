# Open tasks

- Discuss F001b/F001c findings and project scope with Dolly/Sudaksh; select the final medical scenario before designing a policy.
- Specify scenario utility, operating constraints, target hardware, and latency/energy measurement protocol.
- Decide whether adaptive depth should be input-dependent or selected globally from external device state; F001c does not establish a deployable routing policy.
- Treat confidence/entropy as candidate signals only. If pursued, pre-specify a simple policy and compare it with fixed-depth baselines without tuning on the final test set.
- Define a test-set protocol only after scenario, model/policy, and metrics are locked; the official DermaMNIST test split has not been used in F001/F001b/F001c.
- Reassess whether late depths, particularly 12, justify their added measured compute in the chosen scenario. F001c found 28 9→12 rescues but 35 regressions, with 5 samples first correct at depth 12.
- Keep oracle results clearly labeled as unreachable upper bounds; do not present their coverage/compute as achievable controller performance.
