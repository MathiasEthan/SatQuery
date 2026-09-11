# src/train.py
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score

from src.fusion import FusionClassifier
from src.cached_dataset import CachedFeatures, ShuffledFeatures
from src.device import get_device

CKPT_DIR = Path("checkpoints")

CONFIGS = {
    "s2_only": dict(use_s2=True,  use_s1=False),
    "s1_only": dict(use_s2=False, use_s1=True),
    "fused":   dict(use_s2=True,  use_s1=True),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inputs(batch, cfg, device):
    return dict(
        s2=batch["s2"].to(device) if cfg["use_s2"] else None,
        s1=batch["s1"].to(device) if cfg["use_s1"] else None,
    )


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    probs, trues = [], []
    for batch in loader:
        logits = model(**_inputs(batch, cfg, device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(batch["label"].numpy())
    P, Y = np.concatenate(probs), np.concatenate(trues)
    hard = (P > 0.5).astype(int)

    present = Y.sum(0) > 0
    ap = average_precision_score(Y[:, present], P[:, present], average="macro")

    return {
        "macro_f1": f1_score(Y, hard, average="macro", zero_division=0),
        "micro_f1": f1_score(Y, hard, average="micro", zero_division=0),
        "mAP": float(ap),
    }


def train_one(config_name, seed=0, epochs=40, lr=1e-3,
              batch_size=128, patience=8, verbose=True, shuffle_s1=False):
    cfg = CONFIGS[config_name]
    set_seed(seed)
    device = get_device()

    ds_cls = ShuffledFeatures if shuffle_s1 else CachedFeatures
    loaders = {
        s: DataLoader(ds_cls(s), batch_size=batch_size, shuffle=(s == "train"))
        for s in ["train", "val", "test"]
    }

    model = FusionClassifier(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loaders["train"]:
            logits = model(**_inputs(batch, cfg, device))
            loss = loss_fn(logits, batch["label"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch["label"])
        sched.step()

        val = evaluate(model, loaders["val"], cfg, device)
        if verbose:
            print(f"  ep{ep:02d} loss={total/len(loaders['train'].dataset):.4f} "
                  f"val_f1={val['macro_f1']:.4f} val_mAP={val['mAP']:.4f}")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], cfg, device)

    CKPT_DIR.mkdir(exist_ok=True)
    tag = f"{config_name}_seed{seed}" + ("_shuffled" if shuffle_s1 else "")
    torch.save({
        "state_dict": best_state,
        "config_name": config_name,
        "model_kwargs": cfg,
        "seed": seed,
        "shuffle_s1": shuffle_s1,
        "best_val_f1": best_f1,
        "test_metrics": test,
    }, CKPT_DIR / f"{tag}.pt")

    return {"config": config_name, "seed": seed, "shuffled": shuffle_s1,
            "best_val_f1": best_f1, **test}