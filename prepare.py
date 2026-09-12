"""Unpack the dataset and build a patient-level index.

The split is by PATIENT, never by recording. Each patient contributes several
auscultation locations, and putting one of a patient's recordings in train and
another in test would quietly inflate every number we report.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ZIP = DATA / "circor.zip"
ROOT = DATA / "circor"


def unpack() -> Path:
    if ROOT.exists() and any(ROOT.rglob("*.wav")):
        return ROOT
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(ROOT)
    return ROOT


def find_training_dir(root: Path) -> Path:
    for p in root.rglob("training_data.csv"):
        return p.parent
    raise FileNotFoundError("training_data.csv not found in the archive")


def build_index() -> list[dict]:
    root = unpack()
    base = find_training_dir(root)
    wav_dir = base / "training_data"
    rows = list(csv.DictReader((base / "training_data.csv").open()))
    index = []
    for r in rows:
        pid = r["Patient ID"]
        wavs = sorted(str(p) for p in wav_dir.glob(f"{pid}_*.wav"))
        if not wavs:
            continue
        index.append({
            "patient_id": pid,
            "murmur": r.get("Murmur", "").strip(),      # Present / Absent / Unknown
            "outcome": r.get("Outcome", "").strip(),    # Abnormal / Normal
            "age": r.get("Age", "").strip(),
            "locations": r.get("Murmur locations", "").strip(),
            "wavs": wavs,
        })
    (DATA / "index.json").write_text(json.dumps(index, indent=1))
    return index


if __name__ == "__main__":
    idx = build_index()
    from collections import Counter
    print(f"patients indexed : {len(idx)}")
    print(f"recordings       : {sum(len(p['wavs']) for p in idx)}")
    print(f"murmur           : {dict(Counter(p['murmur'] for p in idx))}")
    print(f"outcome          : {dict(Counter(p['outcome'] for p in idx))}")
