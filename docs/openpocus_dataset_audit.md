# T002: OpenPOCUS dataset audit

**Audit date:** 2026-09-26. **Scope:** dataset characterization only, for [issue #4](https://github.com/CharalamposGiannakis/adaptive-medical-edge-ai/issues/4). No training, data cleaning, split generation, task selection, architecture choice, or F002 definition was performed.

## Summary

| Property | Measured result / qualification |
|---|---|
| Exam folders / CSV rows | 226 / 226, one-to-one coverage |
| Biological patients | Not independently verifiable; paper reports 226; suffix grouping gives 219 provisional groups |
| Frames | 325,731 JPEG files; all decode as 512×512 RGB |
| Raw storage | 5,842,855,030 logical bytes: 5.84 GB / 5.44 GiB |
| Observed exam × zone groups | 1,871; not a verified count of distinct original videos |
| Supplied defined zone labels | 1,871 (codes 0–4), of which **1,862 have frames**, across 225 exams |
| Other observed zones | 3 with undefined code 5; 6 with missing labels |
| Frame label coverage | 323,597 frames have codes 0–4; 1,119 have code 5; 1,015 have no label |
| Exact duplicate evidence | 1,846 duplicate-image groups within exams; none across exam folders; 515 groups carry conflicting inherited labels |
| FPS / clip duration | Unavailable from the local files/metadata; neither estimated |
| Split feasibility | Exam/group-based partitioning is implementable; true patient independence remains unverified |

**Reading rule:** “measured” refers to this local snapshot; “paper” refers to publication claims; interpretations and unresolved identity questions are identified explicitly. No files or labels were excluded or repaired.

## Provenance and evidence

**Source facts.** OpenPOCUS is distributed through [Andre Kumar's OpenPOCUS repository](https://github.com/kumarandre/OpenPOCUS), which points to image hosting. Our merged T001 pipeline downloads Stanford Medicine Box file `f_2083355562839`, named `Lung Database.zip`, using shared link `ajv4y3fv5i6mhs345mwhbvkg80cuvzcc`. The local inputs are `data/openpocus/raw/` and the committed `data/openpocus/labels.csv` (the explicit D017 metadata exception).

The associated paper is Kumar et al., **Creation of an Open-Access Lung POCUS Image Database for Deep Learning and Neural Network Applications**, *POCUS Journal* 11(1), April 2026, pp. 62–67, [DOI 10.24908/pocusj.v11i01.19439](https://doi.org/10.24908/pocusj.v11i01.19439). The stored sources are [the paper](OpenPOCUS_paper_Kumar2026.pdf) and [Supplementary Material S1](OpenPOCUS_supplementary_S1.pdf). PDF page numbers below count from 1.

**Version uncertainty.** The local distribution has no explicit release/version identifier. A Box file ID does not pin immutable bytes. This audit identifies the observed snapshot with source SHA-256 hashes and a sorted raw-file inventory containing the SHA-256 of every file. These identify this download; they are not an upstream authenticity guarantee. The original download archive is no longer present. The repository base at audit start was `794d56e8b31d033f8cc201f43e6a1de4b0b72f06`.

| Snapshot input | SHA-256 |
|---|---|
| Committed `labels.csv` | `697aec2ca75438b9352ebee5bbf36323886f577fa03773fa0698e795cece37d5` |
| Downloaded `raw/labels.xlsx` | `98f743b517290c9201e949db6daa4b637479d760c46f9d6b67520e4dcfd8fad0` |
| Sorted `raw_inventory.csv` | `348999da86609c90a365e898a24ab2ba3185ee431d16232099df82f0082566be` |

**License.** The dataset's repository [license notice](../data/openpocus/LICENSE.md) states **CC BY-NC-SA 4.0**, attributed to Andre Kumar (2026). The [official license summary](https://creativecommons.org/licenses/by-nc-sa/4.0/) permits copying and adaptation for noncommercial purposes with attribution, a license link, indication of changes, and the same license for distributed adaptations, without additional restrictions. Noncommercial academic coursework/research is within those stated permissions when those conditions are met; “academic” is not a blanket permission for commercial use. The paper separately states **CC BY 4.0** on PDF p. 1. These are distinct licenses. The raw tree itself contains no additional license file.

**Paper facts, not independently reconstructed clinical facts.** The study enrolled 226 adults in March 2020–April 2022, primarily emergency-department patients with respiratory symptoms concerning for COVID-19, plus controls from emergency departments or clinics without known pulmonary disease/symptoms. Three blinded experienced physicians reviewed clips; 1,445/1,871 initially agreed and 426 required adjudication. Unresolved disagreements became “indeterminate.” The paper describes 12-zone acquisition, modified to 8 zones where necessary, native-rate frame extraction, masking of identifiers/device annotations, and 512×512 letterboxing (PDF pp. 2–3). It explicitly cautions that interpretation was at clip level, not frame level (p. 5). The supplement is an aggregate pathology-by-region table, not individual reviewer annotations, despite the Methods referring to S1 as a sample spreadsheet.

## Measurements and label semantics

**Observed in the workbook and CSV:** `patient`/`ai_id` identify folders/source rows; `device` is acquisition metadata; `covid` and `abnormal` are exam-level targets; `z1`–`z12` are zone-level findings; `frames` is the spreadsheet's reported exam frame count, not a count recomputed from disk. There are no missing exam-level COVID, abnormal or device entries.

| Zone code | Supplied meaning | Supplied zone labels | % of 1,871 defined labels | Zones with frames |
|---|---|---:|---:|---:|
| 0 | Normal | 1,233 | 65.90% | 1,227 |
| 1 | B-lines | 338 | 18.07% | 337 |
| 2 | Consolidation | 84 | 4.49% | 84 |
| 3 | B-lines and consolidation | 119 | 6.36% | 117 |
| 4 | Indeterminate | 97 | 5.18% | 97 |

There are **2,712 possible exam × zone slots** (226 × 12): 1,871 defined labels, **3 undefined code-5 cells** (0.11% of slots), and **838 blank cells** (30.90%). A blank does not establish that a zone was never acquired; file evidence is checked separately. Code 4 is explicitly uncertain interpretation, not a fifth disease. Code 5 has no definition in the supplied key and must not be silently recoded. Normal labels outnumber consolidation-only labels by 14.68:1.

| Inherited zone label | Frames | % of all 325,731 frames |
|---|---:|---:|
| Normal | 218,698 | 67.14% |
| B-lines | 56,200 | 17.25% |
| Consolidation | 13,888 | 4.26% |
| Both | 20,027 | 6.15% |
| Indeterminate | 14,784 | 4.54% |
| Undefined code 5 | 1,119 | 0.34% |
| Missing | 1,015 | 0.31% |

These frame counts describe label association by folder/filename, not individual frame adjudication. Exact repeated frames and Pt45's overlapping renditions are included, so these are file counts rather than unique clinical observations.

| Exam-level field | Counts and percentages (denominator 226 exams) |
|---|---|
| COVID | Positive 114 (50.44%); negative 112 (49.56%) |
| Abnormal | 143 (63.27%); normal 83 (36.73%) |
| Device | Butterfly 134 (59.29%); EchoNous 21 (9.29%); SonoSite 1 (0.44%); Lumify 0; Vave 56 (24.78%); GE 14 (6.19%) |

The workbook device key lists codes 0–4, omitting 5. Its device summary explicitly names GE; the 14 code-5 exam rows match that tally and the paper's GE count. This is a documented mapping inference, separate from the undefined **zone** code 5.

**Observed distinction between targets:** `Pt111`, `Pt130`, `Pt174` and `ED4` have `abnormal=1` with no zone coded 1/2/3. The first three include indeterminate zones; ED4 has only an undefined zone label. No `abnormal=0` exam contains a zone coded 1/2/3. Therefore the provided exam flag must be preserved as a separate target, rather than regenerated using an assumed logical OR of definite zone findings.

**Observed device/COVID association:** Butterfly has 106 positive and 28 negative exams; EchoNous 1/20; SonoSite 1/0; Vave 6/50; GE 0/14. Thus 106/114 positive exams (92.98%) come from Butterfly, and COVID positivity is 79.10% for Butterfly versus 8.70% for all other devices combined. **Interpretation:** device characteristics could act as shortcuts for a COVID model. Balanced overall COVID counts do not remove that concern. This is not a measured model result.

## Patient identity and leakage

The `patient` column identifies an **exam folder**. It is a usable join key, but does not establish a globally unique biological patient. `ai_id` is the workbook ID, with Excel numeric formatting loss; it is not a second independently verified patient identifier. Normalizing a numeric ID for matching is distinct from establishing independence between people.

The known suffix families must be treated as potential shared subjects for leakage control: **Pt146A/B/C, Pt151A/B, Pt156A/B, Pt157A/B, and Pt160A/B/C**. Their device, COVID and exam-level abnormal flags agree within each family. This supports caution but does not prove their acquisition dates or patient identity. Removing suffixes only for these families yields 219 provisional groups from 226 exams, not 219 verified people. No split has been generated.

The 28 dotted folder IDs (for example `Pt31.20`, `Pt38.4`) have no documented interpretation. Numeric-stem inspection found no further collisions beyond the five suffix families, but neither dots nor different folder IDs prove distinct people. The released metadata provides no visit dates or authoritative cross-exam patient key.

**Measured duplicate search:** both encoded-file SHA-256 and decoded-pixel SHA-256 find 1,846 groups involving 5,381 files (3,535 copies beyond one per exact group). All are within the same exam; **zero exact hash groups cross folders**, including the A/B/C families. Of these groups, 898 cross zone boundaries. In **515 groups / 1,030 files**, identical images inherit different labels:

| Exam | Identical-image groups | Zone pair | Inherited codes |
|---|---:|---|---|
| ED2 | 365 | z2 / z3 | 2 / 3 (consolidation / both) |
| Pt160C | 150 | z5 / z6 | 1 / 2 (B-lines / consolidation) |

For example, `ED2/image_1332_ED02_z02_frame_000725.jpg` equals `ED2/image_1697_ED02_z03_frame_000000.jpg` byte-for-byte. The 150 Pt160C pairs include z5 frame 226 versus z6 frame 0. This establishes overlapping frame content across nominal zones. It does **not** prove which zone assignment is correct or invalidate a clip-level label merely because an individual frame is shared. It does demonstrate inconsistent inherited supervision if every frame is treated as an independently labeled image.

**Splitting conclusion:** partitioning by whole exams, with each named A/B/C family kept together across **all train/validation/test partitions**, is technically possible and avoids those known family/within-exam leaks. A verified biological-patient-disjoint split cannot be guaranteed from the supplied identifiers. No absence-of-duplication claim is made for near-duplicates or different exams of the same person. No split proportions, assignment, exclusion or final grouping policy was selected. Further provenance on subject identity is an unresolved input to the later split protocol.

## Media and data quality

### Inventory and temporal information

The raw tree has **325,964 files**: 325,731 `.jpg`, 232 `.csv`, and one `.xlsx`. There are no original video files, DICOM files or remaining ZIP archives. All JPEGs were fully decoded using Pillow, with **zero decode exceptions**, and every file is 512×512, RGB (three channels). This checks readability, not clinical quality or complete de-identification.

| Frame-count unit | Minimum | Q1 | Median | Q3 | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Exam folder (226) | 240 | 908.25 | 1,371.5 | 1,805.25 | 3,831 | 1,441.29 |
| Observed exam × zone (1,871) | 17 | 111 | 143 | 207.5 | 1,090 | 174.09 |

Quartiles use linear interpolation at `(n−1)q`. Logical file lengths give the disk-size figure; filesystem allocation overhead is not included. Images alone account for 5,816,650,175 bytes.

Every filename yields a zone z01–z12 and a frame index. There are **1,875 filename families**, because Pt45 has a second family in four zones. A family or zone is not necessarily a distinct original video: the cross-zone duplicate sequences prevent that assumption. The number of observed zones per exam is: 1 (1 exam), 2 (4), 3 (2), 4 (25), 5 (3), 6 (10), 7 (11), 8 (106), 10 (4), 11 (4), 12 (56). Thus the local file coverage is not simply 70 complete twelve-zone and 156 complete eight-zone exams, even though those are the paper's acquisition-protocol counts.

**Frame rate and duration are unknown.** No EXIF fields were found. Only JFIF-related image metadata was present where image metadata existed; this contains no acquisition timing. Metadata CSV timestamps range from 2025-10-16 to 2025-11-17, with only 1–12 distinct timestamps per CSV despite hundreds/thousands of frames: they describe later file processing, not reliable source frame times for 2020–2022 acquisitions. No trustworthy FPS, timebase or duration field is supplied. Native-rate extraction is a paper claim, not a numeric frame rate. Neither FPS nor duration was inferred from image counts, frame indices, device type or conversion timestamps.

### Missing, unknown and misaligned zones

**Nine labels have no matching frame files:** `Pt09.19 z4` (3); `Pt151B z2/z3/z4/z8/z11` (0), `Pt151B z9` (3); `pt218 z1` (1); `ED3 z1` (0).

**Six zones have frames but blank labels:** `Pt147 z11` (120 frames), `Pt147 z12` (120), `Pt146B z6` (90), `Pt146B z12` (91), `pt218 z2` (294), `ED3 z2` (300). The z1/z2 pattern in pt218 and ED3 is a possible naming/label alignment problem, not permission to shift their labels. This explains why 1,871 observed zones and 1,871 defined label cells coincide numerically while only **1,862 intersect**.

**Undefined code 5:** `pt220 z4` has 378 frames, `pt220 z10` 246, and `ED4 z10` 495. ED4 has no zone with a defined code 0–4, though it has exam-level COVID/abnormal labels.

Defined label counts by zone z1…z12 are **219, 198, 191, 197, 70, 70, 219, 195, 186, 189, 69, 68**. Observed zones with frames are **217, 199, 190, 196, 70, 71, 219, 194, 185, 191, 69, 70**. Individual assignments and all 2,712 slots are reproducible in `zones.csv`. Do not equate a missing label with an unscanned zone or infer the physical zone numbering from these counts alone.

### Frame counts and Pt45

The CSV/workbook sum is **324,027**, whereas disk contains **325,731**, a net difference of **+1,704**. The eight discrepancies are independently confirmed:

| Exam | Spreadsheet | Actual files | Difference |
|---|---:|---:|---:|
| Pt45 | 98 | 787 | +689 |
| Pt163 | 1,140 | 1,440 | +300 |
| pt189 | 2,083 | 2,803 | +720 |
| pt180 | 2,240 | 2,239 | −1 |
| Pt182 | 1,826 | 1,825 | −1 |
| pt193 | 2,488 | 2,487 | −1 |
| pt194 | 1,941 | 1,940 | −1 |
| pt195 | 2,147 | 2,146 | −1 |

Pt45 contains **689 `image_...` files** and **98 files ending ` 2.jpg`**. Both cover z1/z2/z7/z8: the respective counts are 155/217/191/126 versus 19/37/37/5. The second family contains sparse frame indices and matches the spreadsheet's total 98; its identity as the intended dataset rendition is unconfirmed. A visual inspection of the two z1/frame-2 files showed the same ultrasound content with device overlays retained in the ` 2.jpg` rendition and masked in the `image_...` rendition. They are not exact duplicates by file or pixel hashes. This is evidence of overlapping content and inconsistent preprocessing, not four additional independent clips. No variant was removed or preferred. Only those four sparse Pt45 families have gaps between their first and last frame indices; contiguous numbering elsewhere is not evidence of a recovered video timebase.

### Workbook, CSV metadata and naming

- Workbook **Dataset** and **Frame Level Data** have identical A1:Q227 data regions; the latter is not a frame-by-frame ground-truth table. **Video Clips** contains aggregate tallies. The committed CSV matches the workbook rows after the existing ID normalization, and all **94 cached summary totals** agree.
- **191 metadata CSVs** contain 244,531 entries. All report original/output sizes of 512×512 and `status=success`; these “original” sizes therefore do not recover pre-standardization video resolution. Replacing `.png` with `.jpg` matches every entry to an existing file; only 8,736 names match without extension normalization. Pt45 has 98 additional files absent from its metadata CSV. `image_number`, `file_size` and `timestamp` are processing metadata, not labels or a source-video clock.
- **41 annotation CSVs** contain 90,765 rows: **9,663 annotation values are `0`, and 81,102 are blank**, correcting the previous README's “all zero” claim. The zeros occur in six files (Pt5, Pt112, Pt114, Pt116, Pt118, Pt119) whose filenames predate the `image_...` renaming, so none is an exact current filename match. The 81,102 blank entries match current filenames. All referenced zones have normal labels. These files provide no documented extra target vocabulary or independent frame adjudication and cannot replace the zone labels.
- Names vary in `Pt`/`pt`, `z`/`Z`, zero padding, optional `image_...` prefixes and patient tokens. The 14,036 frame-name ID discrepancies flagged by the script are `ED01`–`ED09` versus folder IDs `ED1`–`ED9`, a zero-padding convention. pt218–pt221 omit the patient token. Folder joins and case-insensitive zone parsing are necessary; filenames are not a uniform source-video identifier.
- Excel removes decimal trailing zeros (e.g. Pt31.20 → 31.2) and stores some numeric IDs with floating-point artifacts (33.16 as `33.159999999999997`). The audit mirrors the builder's three-decimal normalization only for matching IDs and checks one-to-one coverage. The README reports the GitHub workbook as older than the Box copy; this audit uses and fingerprints the local Box workbook and does not substitute that other version.

### Publication discrepancies

- The paper's abstract, preprocessing section and Figure 1 report **324,027 frames**; its introduction reports **303,977**. These are conflicting source claims, not alternative local counts (PDF pp. 1–3).
- The prose and Table 2 report **1,871 clips**; Table 1 says **1,874**. The workbook has 1,871 zone labels in codes 0–4 plus three undefined code-5 cells, numerically matching that difference; this is a possible explanation, not an author-confirmed resolution.
- S1 gives normal regional counts 575 anterior + 505 lateral + 171 posterior = **1,251**, which differs from the paper's 1,233 normal clips. The S1 right/left normal totals instead sum to 1,233. Do not use S1 to repair individual labels or infer a zone-to-anatomy mapping.
- Table 1's list of “other diagnoses” sums to 18 while its “Other” row is 17. Diagnoses may overlap, but that is not resolved in the released row-level metadata.

The audit preserves the supplied labels and records these discrepancies rather than reconciling them through undocumented changes.

## Scenario fit

**What the evidence represents:** preprocessed adult lung-ultrasound clips/frames from an emergency-department/clinic study with a substantial COVID component and healthy controls, acquired with several handheld/cart devices and adjudicated lung-finding labels.

**Mismatch with the provisional scenario:** these files do not establish ambulance acquisition, prehospital trauma triage, field motion/noise, battery/network constraints, decision deadlines, or clinical utility under edge deployment. They also do not contain original continuous ultrasound streams with reliable timing. Resource-aware inference could be investigated using them as a technical proxy, but whether to do so remains a design decision.

**Absent row-level targets:** no pneumothorax, lung sliding/lung point, FAST/free-fluid, trauma severity, urgency, intervention or outcome labels are provided. Other diagnosis categories, age, sex and BMI appear only as aggregate paper facts, not as per-exam fields in the supplied workbook/CSV. The paper's Table 1 footnote explicitly mentions **four chest-trauma cases**; therefore “no trauma patients” would be incorrect. Their IDs are unavailable and there is no trauma target to train or evaluate. Likewise, mentioning pneumothorax in the paper's introduction does not create a pneumothorax label. The annotation CSVs do not supply segmentation masks.

## Candidate tasks for T004 (none selected)

Counts below are **available labeled exam/zone groups with files**, before any future exclusion, overlap resolution or clinical-unit validation. Binary mappings are candidate definitions, not changes to the data.

| Candidate | Prediction unit and available sample count | Advantages | Major limitations |
|---|---|---|---|
| Five-way zone finding | One zone/frame sequence: **1,862 groups across 225 exams**; classes 1,227 / 337 / 84 / 117 / 97. Frame inputs would inherit these labels across 323,597 files. | Directly follows the supplied key; includes several lung findings. | Strong imbalance; “indeterminate” represents uncertainty; clip labels do not identify finding presence in each frame; overlapping sequences and missing zones. |
| B-lines presence | Zone sequence, if defined as codes 1+3 vs 0+2: **1,765 groups**, 454 positive / 1,311 negative. | Explicit morphology-based contrast; mixed B-lines/consolidation can be represented as positive. | Counts condition on setting aside 97 indeterminate groups plus unknown/missing cells; this eligibility policy is not selected. Identical frames in Pt160C would receive opposite inherited targets. |
| Normal versus definite abnormal zone | Zone sequence, if codes 0 vs 1+2+3: **1,765 groups**, 1,227 normal / 538 abnormal. | Combines sparse finding classes. | Broad heterogeneous abnormal class; ambiguity handling unresolved; frame inheritance remains uncertain. |
| Provided exam normal/abnormal | Whole exam or aggregation of its zones: **226 exams**, 83 normal / 143 abnormal. | Uses an explicit supplied flag. | Incomplete zone coverage; four flags cannot be reconstructed from definite zone findings; aggregation/unit choice and identity remain unresolved. |
| COVID status | Whole exam or aggregation of its zones: **226 exams**, 114 positive / 112 negative. | Approximately balanced supplied binary target. | Strong device association; small number of potential subjects; COVID is not trauma triage and is not a frame-level finding. |

Every option needs an explicit later decision on prediction/evaluation unit, exclusions, label uncertainty, duplicate handling and subject grouping. Labels inherited from a clip are weak supervision for individual frames; the frame count is not an independent sample size. T004 must resolve the task, and T005 feedback and T007 methodology remain pending before F002.

## Reproduction and validation

Run from the repository root with the existing project environment and Pillow:

```powershell
# This Windows checkout already provides the project environment here:
.\.venv\Scripts\python.exe data/openpocus/audit_openpocus.py --workers 8
```

On an installation where the established environment is named `bde-project`, activate that existing environment and run `python data/openpocus/audit_openpocus.py --workers 8`. No new environment is needed. The named environment was not discoverable on this audit host; the repository's existing `.venv` was used, consistent with its previous experiment run instructions. The script requires Python 3.11+ and Pillow; exact observed versions are recorded in `audit.json` (Python 3.14.3, Pillow 12.3.0 on this host).

The script writes only under ignored `artifacts/T002_openpocus_audit/` (or an `--output` subdirectory under `artifacts/`):

- `audit.json`: measurements, source fingerprints, label controls, anomalies, metadata inventory and cross-exam exact-duplicate pairs.
- `raw_inventory.csv`: every raw file's relative path, logical bytes and SHA-256.
- `frame_manifest.csv`: each image's file/pixel hashes, format, dimensions and mode.
- `zones.csv`: every exam × zone slot with its supplied label and observed frame count.
- `series.csv`: filename families and observed frame indices, without assuming each family is a distinct clinical clip.
- `file_duplicate_groups.json` and `pixel_duplicate_groups.json`: exact duplicate paths for inspection.

It independently compares every data row to the workbook, checks the same 94 cached summary controls as `build_labels.py`, decodes every image with Pillow, hashes all raw files and all decoded image pixel arrays, and compares raw path/size/mtime inventories before and after. Input source hashes are also checked again. It never writes the raw files, workbook or committed CSV. It does not recalculate workbook formulas. File hashes test identical encoded bytes; pixel hashes include decoded mode and dimensions. Neither detects all near-duplicates, cropped/re-encoded variants with changed pixels, or different scans of the same person.

The paper text was read in full, and the paper's Tables 1–2 and the complete supplement were also rendered and visually checked. Temporary extraction/render outputs are ignored under the same artifact directory. The existing `build_labels.py` was additionally executed with its CSV write redirected to an in-memory buffer: its 94 checks passed and the generated CSV rows exactly equaled the committed CSV. The committed CSV was not rewritten.

Final validation: the full audit completed, frame totals reconcile across exam-zone and filename-family tables, candidate counts reconcile to the observed label coverage, and source/inventory fingerprints agree. `git diff --check` passed. Nothing is staged; no raw files or generated audit artifacts are tracked, and the repository ignore rules cover both directories. The pre-existing team-workflow file move was left as found.

## Completion and unresolved questions

**T002's audit acceptance criteria are satisfied locally, pending independent review:** contents, class balance, split feasibility/limits, media properties and scenario mismatch are documented, and T004 has concrete candidate units, counts and limitations. This does not certify the dataset as clinically appropriate or a patient split as validated. The live issue has not been closed by this audit.

Unresolved inputs for T004 and later methodology:

1. Authoritative patient/visit mapping for A/B/C and dotted IDs; exact-image nonoverlap is insufficient proof of patient independence.
2. Provenance of absent/misaligned zones, the meaning of code 5, overlapping zone sequences in ED2/Pt160C, and the intended Pt45 rendition.
3. Original source-video identifiers, numerical FPS/durations and an authoritative zone-to-anatomy mapping.
4. Treatment of indeterminate, unknown, missing and conflicting inherited labels after a task/unit is chosen.
5. Whether the selected task can support an explicitly limited technical-proxy scenario, and how to address device association and clinical-domain mismatch. No winner is proposed here.

No training or random frame split occurred. No raw file, label, scientific decision or downstream completion state was changed.
