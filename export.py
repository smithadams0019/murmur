"""Export to ONNX and check the exported model agrees with PyTorch.

Inference runs in the browser, so the model that ships is this file's output,
not the checkpoint. If the two disagree, the thing users get is not the thing
that was measured -- so the agreement is asserted here rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import features as F
from model import MurmurNet

HERE = Path(__file__).resolve().parent
OUT = HERE / "web" / "murmur.onnx"


def main():
    meta = json.loads((HERE / "metrics.json").read_text())
    width = meta["width"]
    # The shipped threshold is the operating point chosen from the sweep, not
    # whatever train.py happened to land on. evaluate.py must have run first.
    ops = json.loads((HERE / "operating_points.json").read_text())
    chosen = next(o for o in ops if abs(o["target"] - 0.95) < 1e-9)
    net = MurmurNet(F.N_MELS)
    net.load_state_dict(torch.load(HERE / "murmur_net.pt", map_location="cpu"))
    net.eval()

    dummy = torch.randn(1, 1, F.N_MELS, width)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net, dummy, str(OUT),
        input_names=["segment"], output_names=["logit"],
        dynamic_axes={"segment": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,   # keep the weights inside one file for the browser
    )

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    probe = np.random.randn(5, 1, F.N_MELS, width).astype(np.float32)
    with torch.no_grad():
        want = net(torch.from_numpy(probe)).numpy()
    got = sess.run(None, {"segment": probe})[0]
    delta = float(np.max(np.abs(want - got)))
    print(f"exported {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"max |pytorch - onnx| on random input: {delta:.2e}")
    if delta > 1e-4:
        raise SystemExit("ONNX output disagrees with PyTorch; refusing to ship it")

    (HERE / "web" / "model_meta.json").write_text(json.dumps({
        "n_mels": F.N_MELS, "width": width, "sample_rate": F.SAMPLE_RATE,
        "seg_seconds": F.SEG_SECONDS, "hop_seconds": F.HOP_SECONDS,
        "n_fft": F.N_FFT, "hop": F.HOP, "fmin": F.FMIN, "fmax": F.FMAX,
        "threshold": chosen["threshold"],
        "chosen_operating_point": chosen, "metrics": {"val": meta["val"], "test": meta["test"]},
    }, indent=1))
    print("wrote web/model_meta.json")


if __name__ == "__main__":
    main()
