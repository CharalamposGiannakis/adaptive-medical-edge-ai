# Submission reproducibility audit — Phase A

Checked on 2026-10-16 from main commit `351c8492bbd74b4a0a0621a69df8e3458cfbfb7c`
with `C:\venvs\bde-project\Scripts\python.exe` (Python 3.14.3, PyTorch
2.14.0+cu126). This phase changed tests, figure output placement, and this
record. It did not change the accepted F007 protocol, approval, published
results, checkpoints, or inference and statistical methods.

## Commands and outcomes

Run from the repository root in PowerShell:

| Command executed | Outcome |
| --- | --- |
| `C:\venvs\bde-project\Scripts\python.exe -m unittest discover -s tests -v` | Initial run discovered 190 tests; 11 errors came from the sandbox's unwritable default Windows Temp directory. No tracked scientific output changed. |
| `$env:TEMP='C:\uni\bde\project\artifacts\F007'; $env:TMP='C:\uni\bde\project\artifacts\F007'; & 'C:\venvs\bde-project\Scripts\python.exe' -m unittest discover -s tests -v *> 'artifacts\F007\submission_phase_a_unittest.log'; exit $LASTEXITCODE` | Complete applicable suite: **190 run, 186 passed, 0 failures, 0 errors, 4 skipped**. Includes the formerly excluded F007 final-runner denial test. Output was captured in the ignored log. |
| `C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.results.plot_final_results --output-dir artifacts/F007/submission_phase_a_figures_20261016` | Rendered both PNG/SVG pairs outside tracked source directories, from the published report after its pinned SHA-256 check. All four files matched their committed Git blobs byte for byte. |
| `C:\venvs\bde-project\Scripts\python.exe -c "from src.f007_protocol import load_protocol,verify_inputs; print(verify_inputs(load_protocol(),check_checkpoints=True))"` | Passed canonical T006 and all frozen checkpoint hash checks; partition counts were 1,232/265/265. This verifies identities and metadata, without image decoding or inference. |
| `git status --short` and `git diff --check` | Only the intended source/test/audit-note files changed; no whitespace errors. |

The four regenerated figure SHA-256 values matched `git show HEAD:experiments/F007/results/<filename>`: `heldout_quality_vs_flops.png` `45ff33d0538029d9eb2a22c4f8b20eb932776d03ccc0482fa6e4d5c9fc382ba2`, `.svg` `d3d7a5fa41414bb97017d7ace9d720d09092d719e7713be883f20e79aad33caa`, `heterogeneous_coverage.png` `d9ca13a33eb28fe2a75c6b64f21fbc58ddb457a7cb3e52cfd01391d178a14b0e`, and `.svg` `a99b03e3967e36849db32835dfdb910589a3ee0d1195392662bc20ba3eb98eb8`. On Windows, an SVG working-tree copy can use CRLF; compare generated LF bytes with the canonical Git blob.

Two skips compare canonical checkpoints with historical ignored raw F002b/F005b bundles, which are absent locally. Two F005 dynamic-INT8 tests skip because this PyTorch build exposes only `['onednn']`; F005's frozen implementation selects x86/fbgemm/qnnpack. The FP32 quantization-class test ran. No quantization engine or methodology was changed.

## What each reproduction path establishes

1. **Figures from a saved report:** The command above reads `experiments/F007/results/final_test_report.json` (pinned SHA-256 `1e5c3846bd01b03f9c407702f5a3a116b6622f1a32e363fac10f1079364824a5`) and redraws figures only. It does not reconstruct predictions, statistics, or confidence intervals.
2. **Statistics from raw predictions:** The preceding audit executed `C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.final_test --recover-evidence artifacts/F007/final_test_20261014T155259/evidence --output artifacts/F007/audit_recovery_20261015`. The recovered 38 cells and 24 contrasts equal the published report; recovered `report.json` is not byte-identical because it records new recovery provenance. This depends on the ignored original request/unit evidence, which is not in a clean clone. This phase only rechecked those saved report fields; it did not recompute estimates.
3. **Fresh validation inference:** The preceding audit ran `C:\venvs\bde-project\Scripts\python.exe -m experiments.F007.phase_a1 smoke-validation --device cpu --output artifacts/F007/audit_smoke_20261015` in a clean checkout linked to local raw data. It served/deferred 12/12 of 24 requests over two real validation rows. This is a bounded smoke test, not the complete validation experiment, and requires locally supplied raw images.
4. **Frozen held-out evaluation:** The authorized original attempt is preserved in ignored `artifacts/F007/final_test_20261014T155259/`. Held-out inference was **not rerun** in this audit or Phase A. A clean clone alone lacks raw images and the complete ignored original evidence; full clean-clone reproduction has not been demonstrated.

The original CPU benchmark JSON and raw prediction evidence remain local ignored
artifacts, not published source files. Their absence from a clean clone limits
independent artifact-level verification even though the published report and
figures are reproducible from tracked bytes.
