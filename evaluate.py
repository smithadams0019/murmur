"""Sweep the operating point on the held-out test set.

The threshold is a product decision, not a model property. A screening tool
that misses murmurs sends children home; one that over-refers wastes echo
slots. This prints the whole curve so the choice is made with the cost visible.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import features as F
from model import MurmurNet
from train import patient_split, load_split, patient_scores

HERE = Path(__file__).resolve().parent

def main():
    index = json.loads((HERE / "data" / "index.json").read_text())
    train_p, val_p, test_p = patient_split(index)
    cache: dict = {}
    Xva, _, ova = load_split(val_p, cache)
    Xte, _, ote = load_split(test_p, cache)
    width = json.loads((HERE / "metrics.json").read_text())["width"]
    Xva, Xte = Xva[..., :width], Xte[..., :width]

    net = MurmurNet(F.N_MELS)
    net.load_state_dict(torch.load(HERE / "murmur_net.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        vs, vl = patient_scores(torch.sigmoid(net(torch.from_numpy(Xva))).numpy(), ova, val_p)
        ts, tl = patient_scores(torch.sigmoid(net(torch.from_numpy(Xte))).numpy(), ote, test_p)

    print(f"val  AUC {roc_auc_score(vl, vs):.3f}   test AUC {roc_auc_score(tl, ts):.3f}")
    print(f"\ntest set: {int(tl.sum())} with murmur, {int((1-tl).sum())} without\n")
    print(f"{'target sens (val)':>18} {'thr':>7} | {'test sens':>9} {'test spec':>9} {'missed':>7} {'over-referred':>14}")
    rows = []
    for target in (0.85, 0.90, 0.925, 0.95, 0.975, 1.00):
        order = np.sort(np.unique(vs))[::-1]; thr = order[-1]
        for t in order:
            p = vs >= t
            tp = ((p == 1) & (vl == 1)).sum(); fn = ((p == 0) & (vl == 1)).sum()
            if tp / max(tp + fn, 1e-9) >= target: thr = t; break
        p = ts >= thr
        tp = float(((p==1)&(tl==1)).sum()); tn = float(((p==0)&(tl==0)).sum())
        fp = float(((p==1)&(tl==0)).sum()); fn = float(((p==0)&(tl==1)).sum())
        sens = tp/max(tp+fn,1e-9); spec = tn/max(tn+fp,1e-9)
        print(f"{target:>18.3f} {thr:>7.4f} | {sens:>9.3f} {spec:>9.3f} {int(fn):>7} {int(fp):>14}")
        rows.append({"target": target, "threshold": float(thr), "test_sensitivity": sens,
                     "test_specificity": spec, "missed": int(fn), "over_referred": int(fp)})
    (HERE / "operating_points.json").write_text(json.dumps(rows, indent=1))

if __name__ == "__main__":
    main()
