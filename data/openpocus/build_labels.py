"""Rebuild labels.csv from raw/labels.xlsx and check it against the spreadsheet's own totals.

Usage: python3 data/openpocus/build_labels.py   (after download_openpocus.sh)
"""
import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
XLSX = os.path.join(RAW_DIR, "labels.xlsx")
OUT = os.path.join(HERE, "labels.csv")

N = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
z = zipfile.ZipFile(XLSX)
strings = [''.join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(N + "si")]
cells = {}
for c in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(N + "c"):
    v = c.find(N + "v")
    if v is None:
        continue
    col, row = re.match(r"([A-Z]+)(\d+)", c.get("r")).groups()
    cells[col, int(row)] = strings[int(v.text)] if c.get("t") == "s" else v.text


# Excel stored IDs like 31.20 as the number 31.2, so match numerically.
def key(s):
    s = re.sub(r"^pt", "", s.replace(" ", ""), flags=re.I)
    if s.startswith("ED"):
        return s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([A-C]?)", s)
    return (round(float(m[1]), 3), m[2]) if m else None


folders = {key(d): d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))}
zones = [f"z{i}" for i in range(1, 13)]
rows, missing = [], []
r = 2
while (cell := cells.get(("A", r))) and key(cell) is not None:
    folder = folders.get(key(cell))
    if folder is None:
        missing.append(cell)
    vals = [cells.get((c, r), "") for c in "BCDEFGHIJKLMNOPQ"]
    try:
        ai_id = f"{round(float(cell), 3):g}"
    except ValueError:
        ai_id = cell
    rows.append([folder, ai_id] + vals)
    r += 1
if missing:
    sys.exit(f"no raw/ folder for spreadsheet IDs: {missing}")
unlabeled = set(folders.values()) - {row[0] for row in rows}
if unlabeled:
    sys.exit(f"no spreadsheet row for raw/ folders: {sorted(unlabeled)}")

# Cross-check against the summary tables the spreadsheet keeps below the data.
bad = [v for row in rows for v in row[2:] if v and not v.isdigit()]
if bad:
    sys.exit(f"non-integer label values: {sorted(set(bad))}")
col = lambda i: [row[i] for row in rows]
checks = [("patients", len(rows), cells.get(("B", 251))),
          ("covid positive", col(3).count("1"), cells.get(("C", 233))),
          ("abnormal", col(4).count("1"), cells.get(("D", 233))),
          ("frames", sum(int(v) for v in col(-1) if v), cells.get(("Q", r)))]
for code, srow in enumerate(range(245, 251)):
    checks.append((f"device {code} ({cells.get(('A', srow))})", col(2).count(str(code)), cells.get(("B", srow))))
for zi, letter in enumerate("EFGHIJKLMNOP"):
    zone = col(5 + zi)
    checks.append((f"z{zi + 1} filled", sum(1 for v in zone if v), cells.get((letter, 233))))
    checks.append((f"z{zi + 1} codes 0-4", sum(1 for v in zone if v in "01234" and v), cells.get((letter, 241))))
    for code, srow in enumerate(range(236, 241)):
        checks.append((f"z{zi + 1} code {code}", zone.count(str(code)), cells.get((letter, srow))))
failed = [c for c in checks if str(c[1]) != c[2]]
if failed:
    sys.exit("labels.csv disagrees with the spreadsheet's totals:\n" +
             "\n".join(f"  {n}: csv {a}, spreadsheet {b}" for n, a, b in failed))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patient", "ai_id", "device", "covid", "abnormal", *zones, "frames"])
    w.writerows(rows)
print(f"{len(checks)} totals match the spreadsheet, wrote {len(rows)} patients to labels.csv")
