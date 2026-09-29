# OpenPOCUS lung ultrasound

The independently measured T002 audit is in [`docs/openpocus_dataset_audit.md`](../../docs/openpocus_dataset_audit.md). It records image-backed sample counts, missing zones, exact duplicate sequences, recordings shared across folders, patient-identity uncertainty and candidate tasks. The selected T004 task is defined below and in [decision D019](../../docs/decision_log.md#d019--define-the-openpocus-binary-zone-sequence-task). No split has been selected.

License is CC BY-NC-SA 4.0 (`LICENSE.md`): non-commercial, share-alike.
Background on the labeling protocol, devices and interrater agreement is in
`docs/OpenPOCUS_paper_Kumar2026.pdf` and `docs/OpenPOCUS_supplementary_S1.pdf`.

```bash
bash data/openpocus/download_openpocus.sh   # ~5.3 GB into raw/
python3 data/openpocus/build_labels.py      # optional, rebuilds and checks labels.csv
python data/openpocus/audit_openpocus.py    # read-only audit; outputs under ignored artifacts/
python data/openpocus/build_binary_task.py # T004 derivation; reads labels.csv and raw/ only
uv run --no-project --with pillow --with numpy python data/openpocus/find_shared_recordings.py
                                            # shared recordings across folders, split groups
```

## T004: binary normal versus definite abnormal

The prediction unit is **one exam × lung-zone sequence**: all available ordered
frame files for one exam folder and one zone. Frames inherit a zone/clip annotation;
they are not independently adjudicated examples. The supplied exam-level `abnormal`
flag and `covid` label are not used to derive this target.

| Supplied zone code | Meaning | Derived target | Label eligibility |
|---|---|---|---|
| 0 | Normal | 0 (`normal`) | Eligible |
| 1 | B-lines | 1 (`abnormal`) | Eligible |
| 2 | Consolidation | 1 (`abnormal`) | Eligible |
| 3 | B-lines and consolidation | 1 (`abnormal`) | Eligible |
| 4 | Indeterminate | None | Excluded: `indeterminate` |
| 5 | Undefined in the supplied zone key | None | Excluded: `undefined_code_5` |
| Blank/missing | Unavailable label | None | Excluded: `unavailable_label` |

**`abnormal` means B-lines and/or consolidation according to the supplied OpenPOCUS
zone-level annotation.** It does not mean trauma, pneumothorax, FAST/free fluid,
urgency, critical illness or need for intervention. This is an annotation-derived
lung-finding task in the audited ED/clinic dataset. TA Sudaksh considered the
prehospital/ambulance use-case design appropriate, with OpenPOCUS as a technical
experimental proxy; whether dataset bias and sample properties support useful
prediction/adaptivity remains to be tested (D020).

Primary classification metrics are **accuracy and macro-F1**, at the zone-sequence
prediction unit. The candidate classes are imbalanced (69.52% normal, 30.48%
abnormal), so accuracy alone is insufficient. T007 will define the final F002
evaluation protocol; there are no model measurements yet.

### Reproduce the task manifest

Use the existing project environment (this Windows checkout uses `.venv`):

```powershell
.\.venv\Scripts\python.exe data/openpocus/build_binary_task.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_openpocus_binary_task.py -v
```

The utility uses only the standard library and does not rebuild `labels.csv`.
It writes `zones.jsonl` and `summary.json` under ignored
`artifacts/T004_binary_task/`. `--labels` and `--raw` can select another source
snapshot; `--output` must remain under this repository's `artifacts/` and outside
the raw input directory. Generated manifests are not committed.

`zones.jsonl` contains every exam × zone slot, including exclusions and missing
frames. Fields preserve `exam_id` (the original `patient` folder string), `ai_id`,
`zone_id`, `source_label`, `binary_target`, `target_name`, `label_eligible`,
`label_exclusion_reason`, `data_availability`, `frame_count`, `frame_paths` and
`image_backed_candidate`. No patient IDs are merged or normalized. A source-eligible
zone without files retains its target but is not an image-backed candidate.
`image_backed_candidate` means only an eligible label plus at least one JPEG;
it does not assert final data-quality or split eligibility.

Records are sorted by the exact exam ID, then zones 1–12. JPEG paths are relative
to `raw/` and sorted by numeric filename frame index, then case-sensitive relative
POSIX path to break ties. All paths, including alternate Pt45 renditions and repeated
indices, remain present. This is a deterministic enumeration, not a reconstruction
of acquisition chronology. No preferred rendition or frame sampling is selected.
The utility inventories `.jpg`/`.jpeg` files without decoding them; T002 provides
image-readability evidence. Malformed/unmapped JPEG names and unexpected source
label codes fail explicitly. `summary.json` records the source CSV SHA-256,
manifest SHA-256, counts and eligible zones without frames. Outputs have no
timestamps or host-specific absolute paths, allowing byte-for-byte reruns.

### Reconciliation with the audited local snapshot (2026-09-26)

| Population | Normal | Definite abnormal | Total |
|---|---:|---:|---:|
| Eligible by source zone label | 1,233 | 541 | 1,774 |
| Eligible labels without frames | 6 | 3 | 9 |
| Image-backed candidates, before quality/split decisions | **1,227** | **538** | **1,765** |

All 226 exams × 12 zones produce **2,712 records**. Label exclusions comprise 97
indeterminate, 3 undefined-code and 838 unavailable-label records (938 total);
97, 3 and 6 of these respectively have frames. The manifest accounts for all
325,731 JPEG files; the 1,765 candidates contain 308,813 frame files, not that many
independent examples. The nine eligible zones without frames are `Pt09.19 z4`,
`Pt151B z2/z3/z4/z8/z9/z11`, `pt218 z1` and `ED3 z1`.

These counts reconcile exactly with T002. They are measurements of the current
snapshot, not constants used to force eligibility. The optional local integration
test compares every label and frame count against T002's `zones.csv`, verifies
its source CSV hash, and derives expected candidate counts from `audit.json`.
Without the local raw dataset and those audit outputs that check is explicitly
skipped; the synthetic mapping, unit, immutability and determinism tests still run.

### Deferred quality and methodology decisions

ED2/Pt160C inherited-label conflicts, Pt45 alternate renditions, A/B/C repeat-exam
families and unknown biological-patient identity remain unresolved. T002b found
five linked cross-ID sets sharing re-encoded recordings; see the audit and
`split_groups.csv` below. Shared recordings can have conflicting zone labels:
`Pt67`/`Pt81` z8 is code 4 versus code 0, which changes T004 binary eligibility.
T004 preserves source labels and folder/zone/frame identifiers; it applies no
deduplication or quality exclusions. T006 must decide how to handle these conflicts
and keep linked folders together when creating splits.

T006 will determine final group-aware split/exclusion rules. Architecture, model
input processing, frame counts/sampling, F002 methodology/training, adaptivity and
compression remain open. Later quality exclusions do not change this label mapping.

## Layout

```
download_openpocus.sh
build_labels.py
audit_openpocus.py
build_binary_task.py
find_shared_recordings.py
LICENSE.md
README.md
labels.csv           the file to use for labels, one row per exam folder
split_groups.csv     provisional split group per exam folder; keep each group on one side
raw/                 not in Git
  labels.xlsx        original spreadsheet, as shipped inside the Box zip
  <patient>/*.jpg
```

`build_labels.py` stops with an error unless `labels.csv` agrees with all 94 totals in the
spreadsheet's own summary tables.

## What a folder is

One folder in `raw/` represents an exam with varying zone coverage. There are 226 folders
and one row in `labels.csv` per folder. The paper reports 226 patients from emergency
departments and ED/clinic controls, scanned between March 2020 and April 2022. The local
identifiers do not independently establish 226 distinct people.

Inside a folder, each zone (`z01` to `z12`) is a putative clip's frame group. The paper reports native-rate extraction and 512×512 letterboxing, but the local audit finds overlapping sequences across some zones; these groups are not verified independent videos. Numerical FPS and durations are unavailable. The 12-zone protocol takes two front, two side and two
back views per side of the chest. Patients who couldn't turn got an 8-zone scan without the
back views (156 patients, 69%). Zones 5, 6, 11 and 12 are labeled for about 70 exams.
The exact zone-to-anatomy mapping is not established in the available metadata.

Probably not separate people: `Pt146A/B/C`, `Pt151A/B`, `Pt156A/B`, `Pt157A/B` and
`Pt160A/B/C`. The paper does not resolve these individual IDs. Every group shares device,
COVID status and abnormal flag, and `Pt157A` and `Pt157B` differ in only 2 of 12 zones. They
look like repeat exams of one person. Keep each group together across train/validation/test partitions.

Some folders contain the same recordings under another ID, re-encoded so exact hashes miss
them. The five linked sets are `{Pt67, Pt81}`, `{Pt109, Pt176}`, `{Pt135, Pt173}`,
`{Pt140, Pt169, Pt177}` and `{Pt147, Pt163, Pt178}`. Within these sets,
`Pt67`/`Pt81` and `Pt147`/`Pt163` are complete duplicate exams; other links share clips.
Their labels and device entries sometimes disagree. Keep each of these sets together as well,
which leaves 212 split groups, listed in `split_groups.csv` (rebuilt by
`find_shared_recordings.py`); details are in the audit.
Before writing either output CSV, the search checks that the raw exam folders match
`labels.csv` and that every folder contains JPEG frames. A missing, empty or
unlabelled exam folder stops regeneration rather than silently changing the groups.
This coverage check does not establish that every zone or frame is complete.
The dotted IDs (`Pt31.20`, `Pt38.4`) are unexplained. Distinct folder IDs do not prove distinct people; an authoritative patient mapping is unavailable.

## labels.csv

Labels are per exam per lung zone. A frame can inherit its zone label, but was not individually adjudicated. The audit found 515 exact duplicate-image groups with conflicting inherited labels across zones within ED2 and Pt160C.

| column | meaning |
|---|---|
| `patient` | folder name in `raw/`, e.g. `Pt31.20`, `pt210`, `ED4`, `Pt146A` |
| `ai_id` | ID as stored in the spreadsheet, e.g. `31.2`, `146 A` (see below) |
| `device` | 0 Butterfly, 1 EchoNous, 2 SonoSite, 3 Lumify, 4 Vave, 5 GE |
| `covid` | COVID positive: 0 no, 1 yes |
| `abnormal` | supplied exam flag: 0 normal, 1 abnormal; not always reconstructible from definite zone findings |
| `z1` ... `z12` | zone finding, codes below. Empty means an unavailable label; files may still exist |
| `frames` | number of frames for the patient, per the spreadsheet |

Zone codes:

| code | finding |
|---|---|
| 0 | normal |
| 1 | B-lines |
| 2 | consolidation |
| 3 | B-lines and consolidation |
| 4 | indeterminate |

Counts from the spreadsheet: 1233 normal, 338 B-lines, 84 consolidation, 119 both, 97
indeterminate, 1871 labeled zones in total. Zones 5, 6, 11 and 12 are labeled for only
about 70 patients. The rest are labeled for 186 to 219.
Only 1,862 of these defined zone labels have matching frames. Nine labeled zones lack files and six observed zones have blank labels; see the audit for IDs and counts.

## Things that got lost or are odd

- Device 5 is missing from the spreadsheet's header, which only lists 0 to 4. The
  spreadsheet's device tally lists GE with 14 patients, and exactly 14 rows have code 5
  (ED1 to ED10 and pt218 to pt221), so 5 = GE.
- Three zone cells hold code 5, which the key doesn't define: pt220 z4, pt220 z10, ED4 z10.
  The spreadsheet's own totals skip them. Treat them as unknown.
- The spreadsheet on GitHub (`Database for Github.xlsx`) is older than the one in the Box zip.
  It calls pt218 to pt221 "ED11" to "ED14", which matches no folder. `labels.csv` is built
  from the Box copy (`raw/labels.xlsx`).
- Excel stored IDs like `31.20` as the number 31.2, so `ai_id` has lost the trailing zero.
  With a decimal-comma locale it also shows up as `31,2`. Use `patient` to find the folder.
- The spreadsheet's per-frame tab ("Frame Level Data") repeats the per-patient table. There
  are no per-frame zone labels.
- The `frames` column doesn't match the jpg count for 8 folders: Pt45 (98 vs 787), Pt163
  (1140 vs 1440), pt189 (2083 vs 2803), and pt180, Pt182, pt193, pt194, pt195 (off by one).

## Frame files

The zone is the `z01` to `z12` token in the filename (case varies). Naming differs by folder:

- `image_001_Pt102_z01_frame_000000.jpg` in folders that came from a per-patient zip. These
  also have a `metadata.csv` with image size. It lists `.png` names but the files are `.jpg`.
- `image_002_z02_frame_000001.jpg` in pt218 to pt221, with no patient ID in the name.
- `Pt182_Z01_frame_000000.jpg` in folders that were already unpacked.
- 41 folders have a `<patient>annotations.csv`: 9,663 entries contain `0`, and 81,102 are blank. All referenced zones have normal labels. The zero entries use filenames preceding the `image_...` renaming. These files have no documented independent frame-label vocabulary; use the zone metadata with its limitations.
- Pt45 has two frame sets for the same zones (z01, z02, z07, z08): 98 files ending in
  ` 2.jpg`, which matches the spreadsheet's frame count, and 689 from its zip. It's unclear
  which set is intended.

## Using it for machine learning

- Split by the 212 groups above (A/B/C families plus folders that share recordings), never by frame. Neighbouring frames of one clip are
  nearly identical, so a random frame split inflates test scores.
- There are 1,862 image-backed defined-label zone groups across 225 exams, with unresolved patient identity and some overlapping sequences. All 226 exams have exam-level targets. These counts do not establish the number of independent clinical observations, and frame-sampling effects have not been measured.
- Frames by zone label: 218,698 normal, 56,200 B-lines, 20,027 both, 14,784 indeterminate,
  13,888 consolidation. There are also 1,119 frames in code-5 zones and 1,015 frames in zones with no label. T004 excludes indeterminate, code-5 and unavailable-label zones from its binary target while retaining their manifest records and source files.
- 67% of frames inherit normal labels and 59% of exams used a Butterfly probe. Class imbalance and strong device/COVID association must be considered in the later evaluation protocol.
