"""Train the murmur screener.

Three rules this file exists to enforce:

1. The split is by patient. A patient's several recordings never straddle it.
2. The threshold is chosen for screening, not for accuracy. Missing a murmur
   costs a child an echo they needed; a false alarm costs one they did not.
   So we pick the operating point by sensitivity and report what it costs.
3. Patients labelled Unknown are excluded from training. The annotators could
   not hear the answer, so neither can the network, and training on them would
   teach it to guess.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import features as F
from model import MurmurNet

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20260912
EPOCHS = 12
BATCH = 64
TARGET_SENSITIVITY = 0.90


def patient_split(index, seed=SEED):
    """Stratified by label, split on patient id."""
    rng = random.Random(seed)
    present = [p for p in index if p["murmur"] == "Present"]
    absent = [p for p in index if p["murmur"] == "Absent"]
    rng.shuffle(present); rng.shuffle(absent)

    def cut(xs):
        n_test = max(1, int(0.2 * len(xs)))
        n_val = max(1, int(0.1 * len(xs)))
        return xs[:n_test], xs[n_test:n_test + n_val], xs[n_test + n_val:]

    pt, pv, ptr = cut(present)
    at, av, atr = cut(absent)
    return ptr + atr, pv + av, pt + at


def load_split(patients, cache: dict):
    """Segment-level X and y, plus the patient id each segment came from."""
    X, y, owner = [], [], []
    for p in patients:
        label = 1.0 if p["murmur"] == "Present" else 0.0
        for w in p["wavs"]:
            if w not in cache:
                try:
                    cache[w] = F.recording_features(w)
                except Exception:
                    cache[w] = np.zeros((0, 1, F.N_MELS, 1), dtype=np.float32)
            feats = cache[w]
            if feats.shape[0] == 0:
                continue
            X.append(feats)
            y.append(np.full(feats.shape[0], label, dtype=np.float32))
            owner.extend([p["patient_id"]] * feats.shape[0])
    if not X:
        raise SystemExit("no features built; is the dataset unpacked?")
    return np.concatenate(X), np.concatenate(y), np.array(owner)


def patient_scores(seg_scores, owners, patients):
    """A murmur may be audible at one location only, so take the patient's max."""
    by = {}
    for s, o in zip(seg_scores, owners):
        by[o] = max(by.get(o, -1e9), float(s))
    ids = [p["patient_id"] for p in patients if p["patient_id"] in by]
    return np.array([by[i] for i in ids]), np.array(
        [1.0 if next(p for p in patients if p["patient_id"] == i)["murmur"] == "Present" else 0.0
         for i in ids]
    )


def threshold_for_sensitivity(scores, labels, target=TARGET_SENSITIVITY):
    """Lowest threshold that still reaches the target sensitivity."""
    order = np.sort(np.unique(scores))[::-1]
    best = order[-1]
    for t in order:
        pred = scores >= t
        tp = float(((pred == 1) & (labels == 1)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        sens = tp / max(tp + fn, 1e-9)
        if sens >= target:
            best = t
            break
    return float(best)


def report(name, scores, labels, thr):
    pred = scores >= thr
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1e-9)
    spec = tn / max(tn + fp, 1e-9)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    print(f"  {name:10} n={len(labels):4d}  AUC={auc:.3f}  "
          f"sensitivity={sens:.3f}  specificity={spec:.3f}  "
          f"(tp={tp:.0f} fn={fn:.0f} fp={fp:.0f} tn={tn:.0f})")
    return {"n": len(labels), "auc": float(auc), "sensitivity": float(sens),
            "specificity": float(spec), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    index = json.loads((DATA / "index.json").read_text())
    train_p, val_p, test_p = patient_split(index)
    print(f"patients  train={len(train_p)}  val={len(val_p)}  test={len(test_p)}")

    cache: dict = {}
    Xtr, ytr, otr = load_split(train_p, cache)
    Xva, yva, ova = load_split(val_p, cache)
    Xte, yte, ote = load_split(test_p, cache)
    print(f"segments  train={len(ytr)}  val={len(yva)}  test={len(yte)}")

    width = min(a.shape[-1] for a in (Xtr, Xva, Xte))
    Xtr, Xva, Xte = Xtr[..., :width], Xva[..., :width], Xte[..., :width]

    net = MurmurNet(F.N_MELS)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"class weight for Present: {float(pos_weight):.2f}")

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(ytr_t))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = net(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward(); opt.step()
            total += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            va = torch.sigmoid(net(torch.from_numpy(Xva))).numpy()
        ps, pl = patient_scores(va, ova, val_p)
        auc = roc_auc_score(pl, ps) if len(set(pl)) > 1 else 0.0
        print(f"  epoch {epoch:2d}  loss={total/len(perm):.4f}  val patient AUC={auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        va = torch.sigmoid(net(torch.from_numpy(Xva))).numpy()
        te = torch.sigmoid(net(torch.from_numpy(Xte))).numpy()
    vs, vl = patient_scores(va, ova, val_p)
    ts, tl = patient_scores(te, ote, test_p)

    thr = threshold_for_sensitivity(vs, vl)
    print(f"\noperating point chosen on VALIDATION for sensitivity >= {TARGET_SENSITIVITY}: {thr:.4f}")
    print("patient-level results")
    v = report("val", vs, vl, thr)
    t = report("test", ts, tl, thr)

    torch.save(net.state_dict(), HERE / "murmur_net.pt")
    (HERE / "metrics.json").write_text(json.dumps(
        {"threshold": thr, "target_sensitivity": TARGET_SENSITIVITY,
         "val": v, "test": t, "seed": SEED, "width": int(width)}, indent=1))
    print("\nsaved murmur_net.pt and metrics.json")


if __name__ == "__main__":
    main()
