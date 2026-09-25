# OpenPOCUS lung ultrasound

License is CC BY-NC-SA 4.0 (`LICENSE.md`): non-commercial, share-alike.
Background on the labeling protocol, devices and interrater agreement is in
`docs/OpenPOCUS_paper_Kumar2026.pdf` and `docs/OpenPOCUS_supplementary_S1.pdf`.

```bash
bash data/openpocus/download_openpocus.sh   # ~5.3 GB into raw/
python3 data/openpocus/build_labels.py      # optional, rebuilds and checks labels.csv
```

## Layout

```
download_openpocus.sh
build_labels.py
LICENSE.md
README.md
labels.csv           the file to use for labels, one row per patient
raw/                 not in Git
  labels.xlsx        original spreadsheet, as shipped inside the Box zip
  <patient>/*.jpg
```

`build_labels.py` stops with an error unless `labels.csv` agrees with all 94 totals in the
spreadsheet's own summary tables.

## What a folder is

One folder in `raw/` is one lung ultrasound exam of one patient, covering both sides of the
chest. There are 226 folders and one row in `labels.csv` per folder. The paper counts these as
226 patients from emergency departments, scanned between March 2020 and April 2022.

Inside a folder, each zone (`z01` to `z12`) is one video clip, cut into frames at the native
frame rate and letterboxed to 512×512. The 12-zone protocol takes two front, two side and two
back views per side of the chest. Patients who couldn't turn got an 8-zone scan without the
back views (156 patients, 69%). Zones 5, 6, 11 and 12 are filled for only about 70 patients,
so those are most likely the back views. The paper doesn't give the exact zone numbering.

Probably not separate people: `Pt146A/B/C`, `Pt151A/B`, `Pt156A/B`, `Pt157A/B` and
`Pt160A/B/C`. The paper counts each as its own patient. But every group shares device,
COVID status and abnormal flag, and `Pt157A` and `Pt157B` differ in only 2 of 12 zones. They
look like repeat exams of one person. Keep each group on the same side of a train/test split.
The dotted IDs (`Pt31.20`, `Pt38.4`) are unexplained. Treat them as separate patients.

## labels.csv

Labels are per patient per lung zone. A frame gets the label of its patient's zone.

| column | meaning |
|---|---|
| `patient` | folder name in `raw/`, e.g. `Pt31.20`, `pt210`, `ED4`, `Pt146A` |
| `ai_id` | ID as stored in the spreadsheet, e.g. `31.2`, `146 A` (see below) |
| `device` | 0 Butterfly, 1 EchoNous, 2 SonoSite, 3 Lumify, 4 Vave, 5 GE |
| `covid` | COVID positive: 0 no, 1 yes |
| `abnormal` | 0 normal, 1 abnormal (any zone) |
| `z1` ... `z12` | zone finding, codes below. Empty means that zone was not scanned or labeled |
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

## Things that got lost or are odd

- Device 5 is missing from the spreadsheet's header, which only lists 0 to 4. The
  spreadsheet's device tally lists GE with 14 patients, and exactly 14 rows have code 5
  (ED1 to ED10 and pt218 to pt221), so 5 = GE.
- Three zone cells hold code 5, which the key doesn't define: pt220 z4, pt220 z10, ED4 z10.
  The spreadsheet's own totals skip them. Treat them as unknown.
- The spreadsheet on GitHub (`Database for Github.xlsx`) is older than the one in the Box zip.
  It calls pt218 to pt221 "ED11" to "ED14", which matches no folder. The script uses the Box
  copy and drops the GitHub one.
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
- 41 folders have a `<patient>annotations.csv` with one row per frame. Every annotation value
  is `0`, and none of those frames sit in a B-line or consolidation zone, so `0` may just
  mean normal. It's undocumented, so use `labels.csv` instead.
- Pt45 has two frame sets for the same zones (z01, z02, z07, z08): 98 files ending in
  ` 2.jpg`, which matches the spreadsheet's frame count, and 689 from its zip. It's unclear
  which set is intended.

## Using it for machine learning

- Split by patient (and by A/B/C group), never by frame. Neighbouring frames of one clip are
  nearly identical, so a random frame split inflates test scores.
- The real sample size is about 1,871 labeled clips from about 220 people, not 325,731
  frames. Taking a few frames per clip loses little and trains much faster.
- Frames by zone label: 218,698 normal, 56,200 B-lines, 20,027 both, 14,784 indeterminate,
  13,888 consolidation. Drop the 1,119 frames in code-5 zones and the 1,015 frames in zones
  with no label.
- 67% of frames are normal and 59% of patients were scanned with a Butterfly probe, so report
  macro-F1 or balanced accuracy rather than plain accuracy, and check results per device.
