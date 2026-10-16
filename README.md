# Adaptive Medical Edge AI

[Final presentation (PowerPoint)](presentation/Adaptive_Edge_AI_Final.pptx) · [Final presentation (PDF)](presentation/Adaptive_Edge_AI_Final.pdf)

## Project overview

We study resource-adaptive lung-ultrasound inference at the edge. OpenPOCUS is a technical proxy, not a clinical or ambulance validation cohort. The task classifies one lung-zone sequence as normal or definitely abnormal.

The deployable family uses **one nested TinyViT checkpoint**: four channel widths × six frame counts = **24 configurations in one weight file**. A frozen controller selects a configuration or defers under an exact simulated FLOP budget. The held-out comparison followed the accepted [F007 protocol](experiments/F007/README.md).

## Key results

- At full compute, original TinyViT K16 scored 0.880 held-out macro-F1; nested full-width K16 scored 0.879 at the same 20.60 GFLOPs. This does not establish equivalence.
- The nested checkpoint offers a quality–computation range, including 1.82 to 20.60 GFLOPs at the controller's served points.
- Under the frozen mixed-budget schedule, original TinyViT K16 completed 20% of units; adaptive controller F completed 80%. External static MobileNetV2 also completed 80%, with fewer modeled FLOPs than F.
- F never exceeded controller E's held-out abnormal recall at any positive state or on the mixed schedule, despite equal or greater computation. F had slightly higher macro-F1 at `b=0.75`.

See [final scientific results](experiments/F007/FINAL_RESULTS.md) for exact cells, paired intervals, denominators, and limitations.

## Reproduction

Use Python 3.14 with the [F002 dependency pins](experiments/F002/requirements.txt) plus Matplotlib. In an existing Python environment, download OpenPOCUS (~5.3 GB):

```sh
python -m pip install -r experiments/F002/requirements.txt matplotlib
bash data/openpocus/download_openpocus.sh
```

The images go under ignored `data/openpocus/raw/`; see the [dataset notes](data/openpocus/README.md) for labels and license. Canonical checkpoints and T006 manifests are tracked. Verify their identities without decoding images, then run the `unittest` suite:

```sh
python -c "from src.f007_protocol import load_protocol, verify_inputs; print(verify_inputs(load_protocol(), check_checkpoints=True))"
python -m unittest discover -s tests -v
```

**Saved-report figures (no model inference):** The plotter verifies the published report SHA-256 and writes both figures outside tracked source files:

```sh
python -m experiments.F007.results.plot_final_results --output-dir artifacts/F007/reproduced_figures
```

**Fresh validation inference:** Requires the downloaded images and new output directories; analysis uses the resulting prediction evidence:

```sh
python -m experiments.F007.validation_run --all-validation-units --output artifacts/F007/new_validation_run
python -m experiments.F007.analysis --evidence artifacts/F007/new_validation_run --output artifacts/F007/new_validation_report
```

**CPU benchmark:** The [original validation-only benchmark JSON](experiments/F007/results/cpu_benchmark.json) contains the slide's practical JPEG preprocessing-plus-inference medians: 38, 164, 199, 309, 416, and 419 ms (rounded). To measure again:

```sh
python -m experiments.F007.benchmark_cpu --official --output artifacts/F007/new_cpu_benchmark
```

Absolute CPU latency depends on hardware and software. The saved job used one AMD Ryzen AI 9 HX 370, four threads, 200 timed validation requests, 50 warm-ups, and five repeats; model-only timing is a separate field.
Its protocol SHA records the pre-acceptance JSON; the scientific settings and checkpoint identities match the accepted version.

**Fresh held-out inference:** The guarded runner requires the accepted committed protocol, approval receipt, matching frozen hashes, raw images, and a clean checkout. It refuses an existing output directory. The original authorized run is published; this command was **not** rerun for submission:

```sh
python -m experiments.F007.final_test --output artifacts/F007/new_final_test_run --device cpu
```

Full reproduction from a clean clone has not been demonstrated; ignored raw images and original request-level evidence must be supplied separately. Use fresh output paths for each run.

## Experiment navigation

| Experiment | Purpose | Documentation |
| --- | --- | --- |
| F001b/F001c | Early adaptive-depth feasibility | [F001b](experiments/F001b_depth_sweep/README.md), [F001c](experiments/F001c_per_input_depth_utility/README.md) |
| F002 | Static architecture qualification | [F002](experiments/F002/README.md) |
| F002c | Frozen original TinyViT backbone | [F002c](experiments/F002c/README.md) |
| F003 | Adaptive-axis tests | [F003](experiments/F003/README.md) |
| F004 | Shared-weight frame-count family | [F004](experiments/F004/README.md) |
| F005 | Nested width compression | [F005](experiments/F005/README.md) |
| F006a | Cost–quality profile | [F006a](experiments/F006a/README.md) |
| F006a_2 | Validation quality screen | [F006a_2](experiments/F006a_2/README.md) |
| F006b | Resource-aware controller | [F006b](experiments/F006b/README.md) |
| F007 | Frozen final evaluation | [F007](experiments/F007/README.md) |

## Additional documentation

[Final results](experiments/F007/FINAL_RESULTS.md) · [Submission reproducibility audit](docs/submission_reproducibility_phase_a.md) · [Decision log](docs/decision_log.md)
