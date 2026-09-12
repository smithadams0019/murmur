"""Assert the browser features equal the Python features.

The model's held-out numbers were measured on the Python pipeline. If these two
diverge, those numbers describe a system nobody is running.
"""
import json, sys
from pathlib import Path
import numpy as np
import features as F

wav, js_path = sys.argv[1], sys.argv[2]
js = json.loads(Path(js_path).read_text())
py = F.recording_features(wav)                     # (segs, 1, mels, frames)
jsarr = np.array(js["segments"], dtype=np.float32)
jsarr = jsarr.reshape(len(js["segments"]), js["nMels"], js["frames"])

print(f"python: {py.shape[0]} segments, {py.shape[2]}x{py.shape[3]}")
print(f"js    : {jsarr.shape[0]} segments, {jsarr.shape[1]}x{jsarr.shape[2]}")
assert py.shape[0] == jsarr.shape[0], "segment count differs"
w = min(py.shape[3], jsarr.shape[2])
a, b = py[:, 0, :, :w], jsarr[:, :, :w]
diff = np.abs(a - b)
corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
print(f"max |py - js| = {diff.max():.5f}   mean = {diff.mean():.6f}   corr = {corr:.6f}")
if diff.max() > 0.05:
    raise SystemExit(f"FEATURES DIVERGE (max {diff.max():.4f}) - the browser would see different data")
print("features agree")
