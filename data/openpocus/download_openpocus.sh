#!/usr/bin/env bash
# Downloads the OpenPOCUS lung-ultrasound dataset into raw/ next to this script.
#
# Source: https://github.com/kumarandre/OpenPOCUS (index only; the actual
# files are hosted on Stanford Medicine's Box). Dataset license is
# CC BY-NC-SA 4.0 (see LICENSE.md) - the paper itself is CC BY 4.0, the
# dataset files are not, they're non-commercial/share-alike.
#
# Result: raw/<patient folder>/*.jpg and raw/labels.xlsx (the spreadsheet
# shipped inside the Box zip). labels.csv is committed; rebuild it with
# build_labels.py.
#
# Safe to rerun: skips the download if raw/ exists, resumes a partial zip,
# and only tidies what is still untidy.
#
# Install aria2 (brew install aria2) for a faster multi-connection download.
#
# Usage: bash data/openpocus/download_openpocus.sh

set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$TARGET_DIR/raw"
SHARED_NAME="ajv4y3fv5i6mhs345mwhbvkg80cuvzcc"
BOX_BASE="https://stanfordmedicine.app.box.com/index.php?rm=box_download_shared_file&shared_name=${SHARED_NAME}"
ZIP="Lung Database.zip"

cd "$TARGET_DIR"

if [ ! -d "$RAW_DIR" ]; then
  echo "-> $ZIP (~5.3 GB, resumable)"
  if command -v aria2c >/dev/null; then
    aria2c -x 16 -s 16 -k 1M -c --file-allocation=none --console-log-level=warn \
      -o "$ZIP" "${BOX_BASE}&file_id=f_2083355562839"
  else
    curl -L -C - -o "$ZIP" "${BOX_BASE}&file_id=f_2083355562839"
  fi

  echo "-> extracting (unzip checks CRCs, the zip is only removed on success)"
  rm -rf .extract
  unzip -q "$ZIP" -d .extract -x '__MACOSX/*' '*.DS_Store'
  mv ".extract/Lung Database" "$RAW_DIR"
  rm -rf .extract "$ZIP"
fi

echo "-> tidying"
find "$RAW_DIR" -name .DS_Store -delete
if [ -f "$RAW_DIR/Database for Github copy.xlsx" ]; then
  mv "$RAW_DIR/Database for Github copy.xlsx" "$RAW_DIR/labels.xlsx"
fi

echo "-> unpacking per-patient zips"
find "$RAW_DIR" -mindepth 2 -maxdepth 2 -name '*.zip' -print0 |
  xargs -0 -P 8 -I{} sh -c 'unzip -q -n -j "$1" -d "$(dirname "$1")" && rm "$1"' _ {}

echo "Done. Frames in $RAW_DIR"
