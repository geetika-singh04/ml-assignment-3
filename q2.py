"""
Q2 - CNN vs MLP.   Run:  python q2.py      (reuses ckpt/cnn_clean.pt and ckpt/mlp_1x.pt from Q1 if present)

(a) FAIRNESS CRITERION (fixed before looking at results): the two models must have parameter counts
    within 5% of each other and receive identical data, optimiser, epochs, batch size, schedule and
    LR grid; each picks its LR on validation accuracy. Hypothesis H: the CNN's advantage comes from
    spatial inductive bias (locality + weight sharing), not from capacity or training budget.

(b) EXPECTATION (written before running): a fixed random permutation of pixel positions leaves the
    MLP essentially unchanged (it is permutation-agnostic up to optimisation noise, +-1-2 pts), while
    the CNN loses most of its advantage and drops to roughly MLP level or below. If so, H is supported.

(c) Two competing explanations of what the clean CNN relies on:
      A) colour statistics     B) spatial arrangement / shape of local parts
    Test-time dissociation on the same clean-trained model:
      grayscale   -> removes colour, keeps all spatial structure
      patch shuffle (2x2, 4x4, 8x8 grids) -> keeps colour histogram + local texture, breaks global layout
    A predicts: big drop under grayscale, small under shuffle.  B predicts the opposite.
    Because test-time-only changes are also a distribution shift, we ALSO retrain a CNN on grayscale
    images (same transform on train/val/test) to measure how much colour is needed when the model can adapt.
    Errors AND confidence (acc, mean confidence, ECE, confident errors, per-class acc) are analysed.
"""
from common import *

OUT = "results/q2"; os.makedirs(OUT, exist_ok=True)
data = load_data()

EXPECTATION = ("MLP unchanged by pixel permutation (+-2 pts); CNN drops sharply toward/below MLP level.")
cnn_fn = lambda: CNN()
P = count_params(cnn_fn())
mlp_fn, hw = mlp_for_budget(P)
assert abs(count_params(mlp_fn()) - P) / P < 0.05, "fairness criterion violated"


def test_acc(model, d):
    return evaluate(model, d["xte"], d["yte"])[1]


# ------------------------------------------------------------ (a) clean
cnn = fit_cached("cnn_clean", cnn_fn, data)      # first introduced in Q1
mlp = fit_cached("mlp_1x", mlp_fn, data)         # first introduced in Q1
res = {"clean": dict(CNN=test_acc(cnn["model"], data), MLP=test_acc(mlp["model"], data))}

# ------------------------------------------------------------ (b) pixel permutation
perm = make_pixel_perm(seed=123)
pdata = transform_data(data, lambda x: pixel_permute(x, perm))    # SAME permutation for train/val/test
cnn_p = fit_cached("cnn_perm", cnn_fn, pdata)
mlp_p = fit_cached("mlp_perm", mlp_fn, pdata)
res["permuted"] = dict(CNN=test_acc(cnn_p["model"], pdata), MLP=test_acc(mlp_p["model"], pdata))
plot_curves({"CNN clean": cnn["hist"], "CNN perm": cnn_p["hist"], "MLP clean": mlp["hist"], "MLP perm": mlp_p["hist"]},
            f"{OUT}/q2b_curves.png", "Q2b")

# save a visual of the permutation
fig, ax = plt.subplots(1, 2, figsize=(5, 2.6))
ax[0].imshow(data["xte"][0].permute(1, 2, 0).cpu()); ax[1].imshow(pdata["xte"][0].permute(1, 2, 0).cpu())
ax[0].set_title("clean"); ax[1].set_title("permuted")
for a in ax: a.axis("off")
fig.savefig(f"{OUT}/q2b_example.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ (c) colour vs spatial structure
conds = {"clean": lambda x: x, "grayscale": grayscale,
         "shuffle_2x2": lambda x: patch_shuffle(x, 2), "shuffle_4x4": lambda x: patch_shuffle(x, 4),
         "shuffle_8x8": lambda x: patch_shuffle(x, 8)}
analysis, logits_store = {}, {}
for mname, m in (("CNN", cnn["model"]), ("MLP", mlp["model"])):
    for cname, fn in conds.items():
        lg = predict(m, fn(data["xte"]))
        s = confidence_stats(lg, data["yte"]); s["per_class_acc"] = per_class_acc(lg, data["yte"])
        analysis[f"{mname}/{cname}"] = s; logits_store[(mname, cname)] = lg

# retrain on grayscale (transform applied identically to train/val/test)
gdata = transform_data(data, grayscale)
cnn_g = fit_cached("cnn_gray", cnn_fn, gdata)
lg = predict(cnn_g["model"], gdata["xte"])
s = confidence_stats(lg, gdata["yte"]); s["per_class_acc"] = per_class_acc(lg, gdata["yte"])
analysis["CNN_retrained_on_gray/grayscale"] = s

# ------------------------------------------------------------ report
print("\n=== Q2 table: models ===")
print_table([dict(model="CNN", params=count_params(cnn["model"]), lr=cnn["lr"]),
             dict(model=f"MLP (2x{hw})", params=count_params(mlp["model"]), lr=mlp["lr"])])
print("\n=== Q2 test accuracy: clean vs permuted ===")
print_table([dict(condition=k, CNN=v["CNN"], MLP=v["MLP"], gap_CNN_minus_MLP=v["CNN"] - v["MLP"]) for k, v in res.items()])
print("Expectation was:", EXPECTATION)
print(f"CNN change: {res['permuted']['CNN']-res['clean']['CNN']:+.4f} | MLP change: {res['permuted']['MLP']-res['clean']['MLP']:+.4f}")

print("\n=== Q2(c) test-time conditions (confidence / error analysis) ===")
print_table([dict(model_cond=k, **{kk: vv for kk, vv in v.items() if kk != "per_class_acc"}) for k, v in analysis.items()])

a = analysis
drop_gray = a["CNN/clean"]["acc"] - a["CNN/grayscale"]["acc"]
drop_shuf = a["CNN/clean"]["acc"] - a["CNN/shuffle_4x4"]["acc"]
drop_retrain_gray = a["CNN/clean"]["acc"] - a["CNN_retrained_on_gray/grayscale"]["acc"]
print(f"\nCNN drop: grayscale(test-time) {drop_gray:.3f} | shuffle 4x4 {drop_shuf:.3f} | retrained-on-gray {drop_retrain_gray:.3f}")
print("Larger drop under grayscale -> explanation A (colour); larger under shuffle -> B (spatial arrangement).\n"
      "Caveat for the report: test-time perturbations are also a distribution shift; the retrained-gray number\n"
      "controls for that for colour but there is no equally cheap control for shuffling (see Q2b for the retrained version).")

# confidence histograms for the CNN
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, c in zip(axs, ["clean", "grayscale", "shuffle_4x4"]):
    lg = logits_store[("CNN", c)]; conf, pred = F.softmax(lg, 1).max(1); ok = pred == data["yte"]
    ax.hist(conf[ok].cpu(), bins=20, range=(0, 1), alpha=.6, label="correct")
    ax.hist(conf[~ok].cpu(), bins=20, range=(0, 1), alpha=.6, label="wrong")
    ax.set_title(f"CNN, {c}: acc={ok.float().mean():.3f}"); ax.set_xlabel("max softmax prob")
axs[0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/q2c_confidence.png", dpi=150)

save_json(dict(results=res, analysis=analysis, params=dict(CNN=P, MLP=count_params(mlp["model"])),
               lrs=dict(cnn=cnn["lr"], mlp=mlp["lr"], cnn_perm=cnn_p["lr"], mlp_perm=mlp_p["lr"], cnn_gray=cnn_g["lr"])),
          f"{OUT}/q2.json")
