"""
common.py - shared code for Assignment 3 (CIFAR10).
Everything that is reused across Q1/Q2/Q3 lives here (first introduced in this file;
cite "common.py" + the question where it was first *used* in your report).

Design decisions (justify these in the report):
  * Images are kept in [0,1] on the device; per-channel normalisation is the first layer of
    every classifier. This lets us apply transforms (permutation, grayscale, noise) in raw pixel space.
  * Same split, optimiser (AdamW), batch size, epochs, cosine schedule and LR grid for every
    classifier -> fairness. LR is chosen per model on the VALIDATION set only.
  * The best-validation-accuracy epoch is restored (model selection uses validation only).
  * Test set is touched only for final reporting.
"""
import os, json, random
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------ config
ROLL_NUMBER = "26003"   
EPOCHS, LRS, BS, WD = 10, (1e-3,), 128, 1e-2
CKPT = "ckpt"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")


def get_seed():
    digits = "".join(c for c in ROLL_NUMBER if c.isdigit())
    assert digits, "Edit ROLL_NUMBER in common.py"
    seed = int(digits)
    if seed >= 2**32:  # sklearn / numpy need a seed < 2**32
        print(f"[warn] roll-number seed {seed} >= 2**32, using seed mod (2**32-1); mention this in the report")
        seed %= (2**32 - 1)
    return seed


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ data
def load_data(root="./data"):
    """Stratified 45k/5k split of the 50k training images using the roll-number seed."""
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)

    def conv(ds):
        return (torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255.),
                torch.tensor(ds.targets))
    x, y = conv(tr)
    xte, yte = conv(te)
    tri, vai = train_test_split(np.arange(len(y)), test_size=5000, stratify=y.numpy(),
                                random_state=get_seed())
    d = dict(xtr=x[tri], ytr=y[tri], xva=x[vai], yva=y[vai], xte=xte, yte=yte)
    print({k: tuple(v.shape) for k, v in d.items()}, "device:", DEVICE)
    return {k: v.to(DEVICE) for k, v in d.items()}


def transform_data(data, fn):
    """Apply the SAME transform to train/val/test inputs (labels untouched)."""
    return {**data, **{k: fn(data[k]) for k in ("xtr", "xva", "xte")}}


# ------------------------------------------------------------------ transforms (operate on [0,1] images)
def make_pixel_perm(seed=123):
    return torch.randperm(32 * 32, generator=torch.Generator().manual_seed(seed))


def pixel_permute(x, perm):
    """Fixed permutation of the 1024 pixel POSITIONS, identical for all images and all 3 channels
    (each pixel keeps its RGB triple; only spatial arrangement is destroyed)."""
    n, c, h, w = x.shape
    return x.reshape(n, c, h * w)[:, :, perm.to(x.device)].reshape(n, c, h, w)


def grayscale(x):
    g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return g.expand(-1, 3, -1, -1).contiguous()


def patch_shuffle(x, grid, seed=0):
    """Split into grid x grid patches and apply one fixed permutation of patches to every image.
    Preserves the colour histogram and local texture exactly, destroys global layout."""
    n, c, h, w = x.shape
    ph, pw = h // grid, w // grid
    p = x.reshape(n, c, grid, ph, grid, pw).permute(0, 2, 4, 1, 3, 5).reshape(n, grid * grid, c, ph, pw)
    perm = torch.randperm(grid * grid, generator=torch.Generator().manual_seed(seed)).to(x.device)
    p = p[:, perm]
    return p.reshape(n, grid, grid, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)


def gaussian_noise(x, sigma, seed):
    """Fixed (seeded) additive Gaussian noise, clipped to [0,1]. Used for val/test corruption."""
    g = torch.Generator().manual_seed(seed)
    return (x + sigma * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1)


# ------------------------------------------------------------------ models
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


class MLP(nn.Module):
    def __init__(self, hidden, bn=True):
        super().__init__()
        layers, d = [Normalize(), nn.Flatten()], 3072
        for h in hidden:
            layers += [nn.Linear(d, h)] + ([nn.BatchNorm1d(h)] if bn else []) + [nn.ReLU(True)]
            d = h
        layers.append(nn.Linear(d, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """3 stages x (2 x [conv3x3-BN-ReLU]) + maxpool, then one linear layer. ~0.3M params."""
    def __init__(self, chans=(32, 64, 128), convs_per_stage=2):
        super().__init__()
        layers, cin = [Normalize()], 3
        for c in chans:
            for _ in range(convs_per_stage):
                layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)]
                cin = c
            layers.append(nn.MaxPool2d(2))
        layers += [nn.Flatten(), nn.Linear(cin * 4 * 4, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _mlp_params(h, depth, bn):
    p = 3072 * h + h + (depth - 1) * (h * h + h) + h * 10 + 10
    return p + (2 * h * depth if bn else 0)


def mlp_for_budget(target, depth=2, bn=True):
    """Return (constructor, width) of a `depth`-hidden-layer MLP whose #params is closest to target."""
    h = min(range(8, 8192), key=lambda h: abs(_mlp_params(h, depth, bn) - target))
    return (lambda: MLP([h] * depth, bn=bn)), h


# ------------------------------------------------------------------ train / eval
@torch.no_grad()
def predict(model, x, bs=2000):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])


def evaluate(model, x, y):
    lg = predict(model, x)
    return F.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item(), lg


def train_classifier(model, xtr, ytr, xva, yva, epochs=EPOCHS, lr=1e-3, wd=WD, bs=BS, seed=0,
                     keep_best=True, epoch_hook=None, verbose=True, tag=""):
    """Returns history dict. epoch_hook(model)->dict of extra per-epoch metrics."""
    set_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n, hist = len(xtr), defaultdict(list)
    best_acc, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=xtr.device)
        tl = torch.zeros((), device=xtr.device); tc = torch.zeros((), device=xtr.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += loss.detach() * len(idx); tc += (out.argmax(1) == yb).sum()
        sched.step()
        vl, va, _ = evaluate(model, xva, yva)
        hist["train_loss"].append(tl.item() / n); hist["train_acc"].append(tc.item() / n)
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if epoch_hook:
            for k, v in epoch_hook(model).items():
                hist[k].append(v)
        if va > best_acc:
            best_acc, best_ep = va, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"{tag} ep{ep:3d} train {hist['train_acc'][-1]:.3f} val {va:.3f}")
    if keep_best:
        model.load_state_dict(best_state)
    hist = dict(hist); hist["best_epoch"] = best_ep
    return hist


def fit_cached(name, model_fn, data, lrs=LRS, epochs=EPOCHS):
    """LR search on validation accuracy, keep best (lr, epoch). Cached in ckpt/<name>.pt so that
    Q2/Q3 can REUSE models trained in Q1/Q2 (delete ckpt/ to retrain from scratch)."""
    os.makedirs(CKPT, exist_ok=True)
    path = f"{CKPT}/{name}.pt"
    model = model_fn().to(DEVICE)
    if os.path.exists(path):
        s = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(s["state"])
        print(f"[cache] loaded {name} (lr={s['lr']})")
        return dict(model=model, hist=s["hist"], lr=s["lr"], search=s["search"])
    best, search = None, {}
    for lr in lrs:
        m = model_fn().to(DEVICE)
        h = train_classifier(m, data["xtr"], data["ytr"], data["xva"], data["yva"],
                             epochs=epochs, lr=lr, verbose=True)
        search[lr] = max(h["val_acc"])
        print(f"[{name}] lr={lr:g} best val acc={search[lr]:.4f}")
        if best is None or search[lr] > best[0]:
            best = (search[lr], lr, m, h)
    _, lr, m, h = best
    torch.save(dict(state=m.state_dict(), hist=h, lr=lr, search=search), path)
    return dict(model=m, hist=h, lr=lr, search=search)


# ------------------------------------------------------------------ analysis helpers
def confidence_stats(logits, y, bins=15):
    p = F.softmax(logits, 1)
    conf, pred = p.max(1)
    corr = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.float().mean() * (corr[m].mean() - conf[m].mean()).abs()).item()
    nan = float("nan")
    wrong = corr == 0
    return dict(acc=corr.mean().item(), mean_conf=conf.mean().item(),
                conf_correct=conf[corr == 1].mean().item() if (corr == 1).any() else nan,
                conf_wrong=conf[wrong].mean().item() if wrong.any() else nan,
                frac_errors_conf_gt_0p9=((conf > 0.9) & wrong).float().sum().item() / max(wrong.sum().item(), 1),
                ece=ece)


def per_class_acc(logits, y):
    pred = logits.argmax(1)
    return [(pred[y == c] == c).float().mean().item() for c in range(10)]


def plot_curves(hists, path, title=""):
    """solid = train, dashed = validation"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, h) in enumerate(hists.items()):
        c = f"C{i}"
        ax[0].plot(h["train_loss"], c=c, label=f"{name} train"); ax[0].plot(h["val_loss"], "--", c=c, label=f"{name} val")
        ax[1].plot(h["train_acc"], c=c, label=f"{name} train"); ax[1].plot(h["val_acc"], "--", c=c, label=f"{name} val")
    ax[0].set_title("loss"); ax[1].set_title("accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[1].legend(fontsize=7); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def print_table(rows):
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
