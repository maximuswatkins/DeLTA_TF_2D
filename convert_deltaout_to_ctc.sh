#!/bin/bash
set -euo pipefail

for seq in /home/max/MaxWork/DeLTA_TF_New/data/My2D/CTC/test/[0-9][0-9]; do
    [ -d "$seq" ] || continue
    python delta_to_ctc.py "$seq" "${seq}_RES" \
    --prediction-threshold 0.25 \
    --min-target-coverage 0.05 \
    --division-coverage 0.6 \
    --min-overlap-pixels 1

python - "$seq" <<'PY'
from pathlib import Path
import csv
import re
import sys

import numpy as np
import tifffile as tiff

seq = Path(sys.argv[1])
res_dir = seq.parent / f"{seq.name}_RES"
gt_seg_dir = seq.parent / f"{seq.name}_GT" / "SEG"
csv_path = res_dir / "mask_label_counts.csv"

def count_labels(path: Path) -> tuple[int, int]:
    img = tiff.imread(str(path))
    labels = np.unique(img)
    n_labels = int(np.count_nonzero(labels > 0))
    max_label = int(img.max()) if img.size else 0
    return n_labels, max_label

def gt_mask_for_mask_file(mask_file: Path) -> Path | None:
    # mask004.tif -> 004
    m = re.search(r"(\d+)", mask_file.stem)
    if not m:
        return None
    fid = m.group(1)
    for prefix in ("man_seg", "man_track"):
        for suffix in (".tif", ".tiff", ".png"):
            candidate = gt_seg_dir / f"{prefix}{fid}{suffix}"
            if candidate.exists():
                return candidate
    return None

rows = []
for f in sorted(res_dir.glob("mask*.tif")):
    pred_n, pred_max = count_labels(f)

    gt_file = gt_mask_for_mask_file(f)
    if gt_file is None:
        gt_name = ""
        gt_n = -1
        gt_max = -1
    else:
        gt_name = gt_file.name
        gt_n, gt_max = count_labels(gt_file)

    rows.append((f.name, pred_n, pred_max, gt_name, gt_n, gt_max))

with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow([
        "mask_file",
        "pred_num_labels",
        "pred_max_label",
        "gt_mask_file",
        "gt_num_labels",
        "gt_max_label",
    ])
    writer.writerows(rows)

print(f"[{res_dir.name}] wrote {csv_path}")
print("first 5:", rows[:5])
PY

done