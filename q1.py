"""
Q1 - Why does an MLP do poorly on CIFAR10?   Run:  python q1.py

(a) CHOSEN REASON: an MLP lacks the spatial inductive bias of a CNN (locality + weight sharing).
    Rival explanations to separate it from:
      (R1) "not enough capacity"      -> give MLPs 1x, 4x, 16x the CNN's parameters.
      (R2) "not trained long enough"  -> identical epoch budget for everyone + look at train curves
                                         (if MLP train acc >> val acc it is NOT under-fitting).
    Fairness: CNN and MLP_1x have matched parameter counts (BN in both, same optimiser, epochs,
    batch size, cosine schedule, LR grid searched on validation for each). MLP_4x / MLP_16x show what
    extra parameters buy. If the CNN at 1x still beats the MLP at 16x on validation, capacity (R1)
    alone cannot explain the gap.

(b) Memorisation: large MLP, no regularisation (wd=0, no dropout/augmentation), trained on training
    labels of which a fraction `frac` is replaced by a different random class. We log per epoch:
      - acc on correctly-labelled train samples
      - acc on the corrupted samples w.r.t. the ASSIGNED (random) label   -> memorisation
      - acc on the corrupted samples w.r.t. their TRUE label              -> useful structure
      - validation acc (clean labels)
    frac=0 is a control, frac=1 (pure random labels) shows memorisation with no structure at all.
"""
from common import *

OUT = "results/q1"; os.makedirs(OUT, exist_ok=True)
data = load_data()

# ============================================================ (a)
cnn_fn = lambda: CNN()
P = count_params(cnn_fn())
mlp1_fn, h1 = mlp_for_budget(1 * P)
mlp4_fn, h4 = mlp_for_budget(4 * P)
mlp16_fn, h16 = mlp_for_budget(16 * P)
specs = {"CNN": ("cnn_clean", cnn_fn), "MLP_1x": ("mlp_1x", mlp1_fn),
         "MLP_4x": ("mlp_4x", mlp4_fn), "MLP_16x": ("mlp_16x", mlp16_fn)}   # cnn_clean & mlp_1x are reused in Q2/Q3

rows, hists = [], {}
for label, (name, fn) in specs.items():
    r = fit_cached(name, fn, data)
    m, h = r["model"], r["hist"]
    _, tr, _ = evaluate(m, data["xtr"], data["ytr"])
    _, va, _ = evaluate(m, data["xva"], data["yva"])
    _, te, _ = evaluate(m, data["xte"], data["yte"])     # reported only, never used for selection
    rows.append(dict(model=label, params=count_params(m), lr=r["lr"], best_epoch=h["best_epoch"],
                     train_acc=tr, val_acc=va, gap=tr - va, test_acc=te))
    hists[label] = h
print("\n=== Q1(a) comparison ==="); print_table(rows)
plot_curves(hists, f"{OUT}/q1a_curves.png", "Q1a: train (solid) / val (dashed)")
save_json(dict(rows=rows, widths=dict(mlp1=h1, mlp4=h4, mlp16=h16)), f"{OUT}/q1a.json")

# ============================================================ (b)
HID, MEM_EPOCHS, MEM_LR, FRACS = [512, 512], 40, 1e-3, [0.3, 1.0]


def run_memorisation(frac):
    ytr, n = data["ytr"], len(data["ytr"])
    g = torch.Generator().manual_seed(0)
    nidx = torch.randperm(n, generator=g)[: int(frac * n)].to(DEVICE)
    shift = torch.randint(1, 10, (len(nidx),), generator=g).to(DEVICE)
    y_noisy = ytr.clone(); y_noisy[nidx] = (ytr[nidx] + shift) % 10     # guaranteed to differ from the truth
    is_noisy = torch.zeros(n, dtype=torch.bool, device=DEVICE); is_noisy[nidx] = True
    nan = float("nan")

    def hook(model):
        pred = predict(model, data["xtr"]).argmax(1)
        c = ~is_noisy
        return dict(
            acc_clean_subset=(pred[c] == y_noisy[c]).float().mean().item() if c.any() else nan,
            noisy_vs_assigned=(pred[is_noisy] == y_noisy[is_noisy]).float().mean().item() if is_noisy.any() else nan,
            noisy_vs_true=(pred[is_noisy] == ytr[is_noisy]).float().mean().item() if is_noisy.any() else nan,
            train_acc_assigned_all=(pred == y_noisy).float().mean().item())

    model = MLP(HID).to(DEVICE)
    h = train_classifier(model, data["xtr"], y_noisy, data["xva"], data["yva"], epochs=MEM_EPOCHS,
                         lr=MEM_LR, wd=0.0, keep_best=False, epoch_hook=hook, tag=f"mem{frac}")
    te = evaluate(model, data["xte"], data["yte"])[1]     # final (memorised) model, reporting only
    va = np.array(h["val_acc"])
    summ = dict(frac_noisy=frac, params=count_params(model), peak_val_acc=va.max(), peak_val_epoch=int(va.argmax()),
                final_val_acc=va[-1], final_train_acc_assigned=h["train_acc_assigned_all"][-1], final_test_acc=te)
    if frac > 0:
        a = np.array(h["noisy_vs_assigned"])
        summ["first_epoch_noisy_memorised>50%"] = int(np.argmax(a > 0.5)) if (a > 0.5).any() else None
        summ["peak_noisy_vs_true"] = float(np.nanmax(h["noisy_vs_true"]))
        summ["peak_noisy_vs_true_epoch"] = int(np.nanargmax(h["noisy_vs_true"]))
    return h, summ


all_h, summ_rows = {}, []
for f in FRACS:
    all_h[f], s = run_memorisation(f); summ_rows.append(s)
print("\n=== Q1(b) memorisation summary ==="); print_table([{k: v for k, v in s.items()} for s in summ_rows])

fig, axs = plt.subplots(1, len(FRACS), figsize=(5.5 * len(FRACS), 4), squeeze=False)
for ax, (f, h) in zip(axs[0], all_h.items()):
    ax.plot(h["val_acc"], "k", lw=2, label="val acc (clean labels)")
    ax.plot(h["acc_clean_subset"], label="train acc, clean-label subset")
    ax.plot(h["noisy_vs_assigned"], label="corrupted subset vs ASSIGNED label (memorisation)")
    ax.plot(h["noisy_vs_true"], label="corrupted subset vs TRUE label (structure)")
    ax.axhline(0.1, c="gray", ls=":"); ax.set_title(f"{int(f*100)}% labels randomised"); ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
axs[0][0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/q1b_memorisation.png", dpi=150)
save_json(dict(summary=summ_rows, hist={str(k): v for k, v in all_h.items()}), f"{OUT}/q1b.json")
print("""
How to read it: structure is learned BEFORE memorisation if, for frac=0.3, val acc and the
'corrupted subset vs TRUE label' curve rise first (and peak) while 'vs ASSIGNED' stays near 10%, and only
later 'vs ASSIGNED' climbs to ~100% while val acc / 'vs TRUE' decay. frac=1 shows memorisation with val ~10%.""")
