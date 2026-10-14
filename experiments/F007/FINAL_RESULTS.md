# F007 final results — frozen held-out evaluation

**Status:** The authorized final evaluation completed on 2026-10-14. This is a
publication of its frozen report, not a new evaluation or a change to the
[accepted methodology](README.md). Exact cell values, intervals and all 24
prespecified contrasts are in the [copied official report](results/final_test_report.json),
[quality–coverage–cost CSV](results/final_quality_coverage_cost.csv) and
[paired-contrasts CSV](results/final_paired_contrasts.csv). The two figures use
only that report: [quality versus FLOPs](results/heldout_quality_vs_flops.png)
([SVG](results/heldout_quality_vs_flops.svg)) and
[heterogeneous coverage](results/heterogeneous_coverage.png)
([SVG](results/heterogeneous_coverage.svg)).

## Frozen setup and execution

F007 compares one exam × lung-zone alias-row request at a time on the T006
held-out split. Known shared-recording aliases are averaged into one evaluation
unit before applying the fixed 0.5 threshold. This test split has 265 alias
rows, 265 evaluation units, 81 abnormal and 184 normal units, organized in 32
known dependency groups. The five fixed resource states are
`b = 0/0.25/0.50/0.75/1.00`, plus one label-independent heterogeneous schedule.
Every system uses the exact simulated per-request limit
`b × (11/10) × 20,596,068,563` FLOPs. A deferral reduces coverage; it is not
scored as a classification error. Served requests are charged full independent-
request FLOPs. The run contains 10,070 request and 10,070 unit-cell records,
38 report cells (30 factorial, six heterogeneous, two supplementary), and all
24 prespecified contrast entries. Its intervals use the frozen 5,000 shared,
paired PCG64 draws over sorted dependency groups (seed 13013).

The execution used clean commit
`d061ea4dec0e32783bed8f57cc032af08f077585`, the accepted protocol at
commit `25756de9540ecc5ef1379ab09912e3585bb4e7e9`, accepted D030, and the
committed approval receipt. `attempt.json` and `evidence/run.json` both say
`completed`; the latter records `test_inference_authorized=true`, the same
accepted protocol and frozen checkpoint identities as the report, and a
successful no-decode inventory preflight. The original ignored attempt is
preserved at `artifacts/F007/final_test_20261014T155259/`.

## Systems and factorial resource behavior

**B, the original F002c TinyViT K16, is the main architectural baseline.**
A, frozen F002b MobileNetV2 K16, is an external static reference. C is the
canonical nested F005b TinyViT at width 0.5/K8. D chooses the cheapest
validation-quality-qualified configuration; E maximizes validation macro-F1;
F applies the frozen quality-qualified abnormal-recall-priority rule. D/E/F
route within the same nested F005b checkpoint. The supplementary original
TinyViT K4 and nested full-width K16 points are prespecified comparisons,
not extra primary systems.

The table reports held-out completed-unit macro-F1 and, in parentheses,
full GFLOPs per **assigned alias-row request**. A dash means all 265 units
deferred and no predictive metric exists. Every non-dash factorial cell covers
265/265 units. Values shown here are rounded for reading; the copied CSV and
JSON retain exact estimates and intervals.

| Resource `b` | A: MobileNetV2 | B: original TinyViT | C: nested static | D: min-cost | E: macro-F1 | F: recall-priority |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 0.25 | .875 (5.108) | — | — | .844 (1.817) | .844 (1.817) | .844 (1.817) |
| 0.50 | .875 (5.108) | — | .846 (7.266) | .844 (1.817) | .871 (8.782) | .866 (10.298) |
| 0.75 | .875 (5.108) | — | .846 (7.266) | .844 (1.817) | .871 (8.782) | .874 (15.447) |
| 1.00 | .875 (5.108) | .880 (20.596) | .846 (7.266) | .844 (1.817) | .879 (20.596) | .879 (20.596) |

At full capacity, original TinyViT B has macro-F1 0.8804, accuracy 0.8981,
abnormal recall 0.8395 and normal recall 0.9239. Nested full-width K16 and
E/F share macro-F1 0.8787 and abnormal recall 0.8148 at the same 20.596
GFLOPs. C's fixed width-0.5/K8 point uses 7.266 GFLOPs with macro-F1 0.8456;
D's width-0.5/K2 uses 1.817 GFLOPs with macro-F1 0.8439. F moves through
K2, full-width K8, K12 and K16 as the budget rises. The first and last
F/E configurations are identical by construction; this is an exact prediction
identity for the frozen checkpoint and requests, not independent corroboration.

The external MobileNetV2 reference A has macro-F1 0.8750, abnormal recall
0.7654 and normal recall 0.9565 at 5.108 GFLOPs. The separate original
TinyViT K4 diagnostic has macro-F1 0.8645 and abnormal recall 0.8395 at
5.149 GFLOPs. Its FLOPs are within 1% of A's, but their training recipes and
quality trade-offs differ; this near-matched pair does not measure adaptive
value by itself.

## Heterogeneous schedule

The frozen mixed schedule assigns 53 units to each of the five states, with
the same assignment for all six systems. Coverage and mean full FLOPs below
use **all 265 assigned requests**; macro-F1 uses each system's completed
units only. Because those completed subsets differ, the raw macro-F1 column
must not be read as a matched ranking. Use the paired contrasts for that.

| System | Completed / 265 | Coverage | Completed-unit macro-F1 | GFLOPs / assigned row |
|---|---:|---:|---:|---:|
| A · MobileNetV2 external | 212 | 80% | .8742 | 4.086 |
| **B · original TinyViT main baseline** | **53** | **20%** | **.8140** | **4.119** |
| C · nested static | 159 | 60% | .8586 | 4.360 |
| D · min-cost policy | 212 | 80% | .8230 | 1.453 |
| E · macro-F1 policy | 212 | 80% | .8544 | 7.995 |
| F · recall-priority policy | 212 | 80% | .8475 | 9.632 |

All systems defer at `b=0`. B serves only the `b=1` group; C serves the
`b≥0.50` groups; A and D/E/F serve all four positive states. F changes its
nested configuration with the assigned state, demonstrating resource-adaptive
execution while preserving its frozen 80% coverage under this schedule.

## Prespecified paired comparisons

Contrasts are left minus right. Predictive differences use common completed
units; coverage and total work use all assigned requests. Intervals are
approximate, unadjusted, per-comparison 95% dependency-group bootstrap
intervals, not formal superiority, noninferiority or equivalence tests.

| Held-out contrast | Common units | Macro-F1 difference (95% interval) | Interpretation |
|---|---:|---:|---|
| F−B, `b=1` | 265 | −.0017 [−.0221, .0150] | Original TinyViT and the selected nested full-width point are close on this split; the interval does not establish equivalence. |
| F−A, `b=1` | 265 | +.0037 [−.0427, .0569] | No clear macro-F1 separation from the external static reference. |
| F−C, `b=.75` | 265 | +.0282 [.0051, .0554] | The point and its unadjusted interval favor F over fixed nested C at this state, at greater FLOPs. |
| D−A, `b=.25` | 265 | −.0311 [−.0928, .0225] | D uses 64.4% fewer full FLOPs; predictive equivalence is not established. |
| F−E, heterogeneous | 212 | −.0069 [−.0258, .0000] | F costs more on average without a measured macro-F1 gain over E on their common completed units. |

The full [24-contrast table](results/final_paired_contrasts.csv) includes
accuracy, both recalls, AUROC where defined, cost, coverage, denominators,
valid-draw counts and all protocol-defined intervals. For example, F−C
macro-F1 at `b=1` is +.0331 [.0041, .0687], while the heterogeneous F−B
contrast uses only 53 common units and F−C uses 159. These unadjusted
intervals are not a familywise-corrected winner selection. The equal-FLOP
B−nested-full-width comparison is +.0017 [−.0150, .0221] macro-F1; the
different training recipes prevent attributing that difference to pruning
alone.

F's validation-derived recall priority does **not** reliably improve held-out
abnormal recall over E. At `b=.50`, F's abnormal recall is .8025 versus E's
.8148, with 10.298 versus 8.782 GFLOPs/request. On the mixed schedule it is
.7679 versus .7857, with 9.632 versus 7.995 GFLOPs/request and equal 80%
coverage. These are observed policy trade-offs, not a reason to retune the
frozen rule using the test split.

## Comparable CPU context

The prior **validation-only** official CPU benchmark
`artifacts/F007/cpu_validation_20261014T004948183271Z/benchmark.json`
(SHA-256 `223fd2655acb54b6b0ea775d8f220b83d607ec864aaa5a014d9867fdaa0e1de8`)
measured all nine prespecified points in one process on an AMD Ryzen AI 9
HX 370 with four threads, 200 timed validation requests, 50 warm-ups and five
repeats. Selected model-only chunk-of-four median sequence latencies are:
original TinyViT K16 **360.62 ms**, MobileNetV2 K16 **123.62 ms**, nested
full-width K16 **225.58 ms**, and nested width-0.5/K2 **19.32 ms**. The saved
direct/chunk4 maximum logit difference is zero at every point. CPU latency,
throughput and the separate JPEG pipeline are measured quantities distinct
from the held-out report's full FLOP accounting and from incremental clip-cache
FLOPs. FLOPs are not energy or battery measurements.

## Provenance and reproduction of this publication

The following SHA-256 values were checked against local bytes before copying.
The report records the same four source-evidence hashes; the three published
machine-readable files are byte-for-byte copies of the official exports.

| Identity or preserved ignored file | SHA-256 |
|---|---|
| `configs/f007_protocol.json` (accepted; commit `25756de`) | `c69ebae9e7620b2e4df4528881b5d08d6158b6b38b72c6266c64d8cc50d5c358` |
| `configs/f007_approval.json` | `6d7e1b72cafb6d9c9230e9cb32b27009be5f7042df603edd565a1bc6f4a061a5` |
| `artifacts/F007/final_test_20261014T155259/attempt.json` | `7d396eecb8cf2f1d36ae14ab07374a4705e54a405f24922f3a37d663cba70888` |
| `evidence/run.json` | `e71a3ac99e5231287941997349954d4da55aca7b9b966ea316113009a4a0f5ca` |
| `evidence/requests.jsonl` | `ed2545802fc665e3806aa0dd7014074e0bd63e8986f616c701eeafaa5215ad07` |
| `evidence/units.jsonl` | `635fd017ef9ea9d079763838760f4fdbc4d3fabd1629ebe3620c4d57c792e3b2` |
| `evidence/summary.json` | `3f7dda9566242848554ab4b85db95345afa5afff293d0d189189e3779f160713` |
| `report/report.json` | `1e5c3846bd01b03f9c407702f5a3a116b6622f1a32e363fac10f1079364824a5` |
| `report/bootstrap_indices.npy` | `b0bc17e33cc144d1cdc45427fa4e2e367a1de8eff4d78e72cf47ff07fd4b5fb7` |

The accepted protocol contains the frozen T006 manifest fingerprints and three
checkpoint hashes. The run record verifies them and cites the D030 sign-off
and [independent PR review](https://github.com/CharalamposGiannakis/adaptive-medical-edge-ai/pull/29#pullrequestreview-5518707989).
The raw request probabilities, unit records, bootstrap draws and original
figures remain ignored in the complete attempt directory; none was moved or
overwritten. Regenerate only the two presentation figures from the copied
report, using the existing environment:

```powershell
& C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.results.plot_final_results
```

## Supported conclusions and limits

The frozen system routes nested TinyViT points across resource states and
serves 80% of units on the mixed schedule, whereas the full original TinyViT
K16 baseline serves 20% under the same budgets. This demonstrates the
specified **simulated-resource adaptive inference** and coverage behavior.
It does not establish a universal predictive winner: observed quality changes
with configuration, E and F trade quality and work, and F's recall-priority
selection did not consistently improve held-out abnormal recall. The original
TinyViT baseline remains the architectural reference; MobileNetV2 supplies a
separate external static comparison.

The study has one frozen held-out split and one selected seed per checkpoint.
Known dependency groups do not prove biological-patient independence;
OpenPOCUS device and label are confounded, and this dataset is a technical
proxy rather than a prehospital clinical cohort. Intervals cover known-group
resampling, not model-seed uncertainty, and the many prespecified comparisons
are unadjusted. Resource budgets are simulated FLOP limits; CPU timing is from
one validation host, and no edge battery, thermal or clinical outcomes were
measured. No test-derived model, threshold or controller retuning was done.
