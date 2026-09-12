"""The check that actually matters: does the browser produce the same VERDICT?

Tiny per-cell differences in log-mel are expected, because log amplifies noise
in near-silent mel bins. What must not differ is the score the model returns,
because that is what a clinician acts on.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, onnxruntime as ort
import features as F

sess = ort.InferenceSession("web/murmur.onnx", providers=["CPUExecutionProvider"])
thr = json.loads(Path("metrics.json").read_text())["threshold"]
rows = []
for wav in sorted(Path("web/samples").glob("*.wav")):
    out = Path(tempfile.mktemp(suffix=".json"))
    subprocess.run(["node", "verify_dsp.mjs", str(wav), str(out)],
                   check=True, capture_output=True)
    js = json.loads(out.read_text())
    jsf = np.array(js["segments"], np.float32).reshape(-1, 1, js["nMels"], js["frames"])
    pyf = F.recording_features(str(wav))
    w = min(jsf.shape[3], pyf.shape[3])
    sp = 1 / (1 + np.exp(-sess.run(None, {"segment": pyf[..., :w]})[0]))
    sj = 1 / (1 + np.exp(-sess.run(None, {"segment": jsf[..., :w]})[0]))
    p, j = float(sp.max()), float(sj.max())
    rows.append((wav.name, p, j, abs(p - j), (p >= thr) == (j >= thr)))

print(f"{'recording':30} {'python':>8} {'browser':>8} {'delta':>8}  same verdict")
worst = 0.0
for name, p, j, d, same in rows:
    worst = max(worst, d)
    print(f"{name:30} {p:8.4f} {j:8.4f} {d:8.5f}  {'yes' if same else 'NO'}")
print(f"\nlargest score difference: {worst:.5f}")
if not all(r[4] for r in rows):
    raise SystemExit("browser and python disagree on a verdict")
if worst > 0.02:
    raise SystemExit(f"scores drift by {worst:.4f}; too much to ship")
print("browser and python agree on every sample")
