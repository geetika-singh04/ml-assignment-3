"""
Q3 - Convolutional autoencoders.   Run:  python q3.py   (reuses ckpt/cnn_clean.pt from Q1/Q2, trains it if absent)

(a) Three ConvAEs that differ ONLY in the bottleneck width (latent dim 16 / 128 / 1024; input has 3072 values).
    Same encoder/decoder conv stack, optimiser, epochs, schedule. Chosen by lowest validation MSE.
(b) Corruption: additive Gaussian noise (sigma=0.2, clipped to [0,1]) - a corruption that reconstruction
    can reasonably help with. DAE: same architecture as the selected clean AE, input = freshly corrupted
    train image every batch, target = clean image, NO labels. Val/test noise is fixed (seeded).
    Classifier: the fixed clean-trained CNN from Q1/Q2 (never fine-tuned).
(c) Frozen encoders -> standardised features -> linear probe (weight decay chosen on validation).
    Compare probe accuracy against reconstruction MSE. Extra references: untrained (random) encoder
    and a linear probe on raw pixels.
"""
from common import *
from scipy.stats import spearmanr

OUT = "results/q3"; os.makedirs(OUT, exist_ok=True)
data = load_data()
LATENTS, AE_EPOCHS, AE_LR, SIGMA = [16, 128, 1024], 30, 1e-3, 0.2


class ConvAE(nn.Module):
    def __init__(self, latent):
        super().__init__()
        down = lambda i, o: [nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        up = lambda i, o: [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.enc = nn.Sequential(*down(3, 32), *down(32, 64), *down(64, 128), nn.Flatten(), nn.Linear(128 * 16, latent))
        self.dec_fc = nn.Linear(latent, 128 * 16)
        self.dec = nn.Sequential(nn.ReLU(True), *up(128, 64), *up(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(self.dec_fc(z).view(-1, 128, 4, 4))
    def forward(self, x): return self.decode(self.encode(x))


def mse(model, x_in, x_target):
    return F.mse_loss(predict(model, x_in), x_target).item()


def get_ae(name, latent, sigma=None, xva_in=None, seed=0):
    """Train (or load) an AE. sigma=None -> clean AE; else denoising AE with fresh noise each batch.
    The best epoch is chosen on validation MSE only."""
    os.makedirs(CKPT, exist_ok=True); path = f"{CKPT}/{name}.pt"
    model = ConvAE(latent).to(DEVICE)
    if os.path.exists(path):
        s = torch.load(path, map_location=DEVICE, weights_only=False); model.load_state_dict(s["state"]); return model, s["hist"]
    set_seed(seed)
    xtr, xva = data["xtr"], data["xva"]
    xva_in = xva if xva_in is None else xva_in
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_EPOCHS)
    hist, best, best_state = defaultdict(list), 1e9, None
    for ep in range(AE_EPOCHS):
        model.train(); perm = torch.randperm(len(xtr), device=DEVICE); tl = 0.0
        for i in range(0, len(xtr), BS):
            xb = xtr[perm[i:i + BS]]
            xin = (xb + sigma * torch.randn_like(xb)).clamp(0, 1) if sigma else xb
            loss = F.mse_loss(model(xin), xb)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(xb)
        sched.step()
        v = mse(model, xva_in, xva)
        hist["train_mse"].append(tl / len(xtr)); hist["val_mse"].append(v)
        if v < best: best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 5 == 0 or ep == AE_EPOCHS - 1: print(f"[{name}] ep{ep} train {hist['train_mse'][-1]:.5f} val {v:.5f}")
    model.load_state_dict(best_state); hist = dict(hist)
    torch.save(dict(state=model.state_dict(), hist=hist), path)
    return model, hist


# ============================================================ (a)
aes, rows = {}, []
for d in LATENTS:
    m, h = get_ae(f"ae_{d}", d); aes[d] = m
    rows.append(dict(latent_dim=d, params=count_params(m), val_mse=min(h["val_mse"]), best_epoch=int(np.argmin(h["val_mse"]))))
print("\n=== Q3(a) ==="); print_table(rows)
sel = min(rows, key=lambda r: r["val_mse"])["latent_dim"]
sel_test_mse = mse(aes[sel], data["xte"], data["xte"])
print(f"Selected (lowest validation MSE): latent={sel}; test MSE = {sel_test_mse:.5f}")
save_json(dict(rows=rows, selected=sel, selected_test_mse=sel_test_mse), f"{OUT}/q3a.json")

# reconstruction visual
with torch.no_grad():
    xs = data["xte"][:8]
    fig, ax = plt.subplots(1 + len(LATENTS), 8, figsize=(12, 6))
    for j in range(8):
        ax[0][j].imshow(xs[j].permute(1, 2, 0).cpu()); ax[0][j].axis("off")
        for i, d in enumerate(LATENTS):
            aes[d].eval(); ax[i + 1][j].imshow(aes[d](xs)[j].permute(1, 2, 0).cpu()); ax[i + 1][j].axis("off")
    for i, t in enumerate(["input"] + [f"d={d}" for d in LATENTS]): ax[i][0].set_title(t, fontsize=8, loc="left")
    fig.savefig(f"{OUT}/q3a_recons.png", dpi=150); plt.close(fig)

# ============================================================ (b)
xva_n, xte_n = gaussian_noise(data["xva"], SIGMA, seed=1), gaussian_noise(data["xte"], SIGMA, seed=2)
dae, dae_hist = get_ae("dae", sel, sigma=SIGMA, xva_in=xva_n)
clf = fit_cached("cnn_clean", lambda: CNN(), data)["model"]       # reused (Q1); frozen, never fine-tuned
clean_ae = aes[sel]


def pipeline(x_clean, x_noisy, y):
    acc = lambda x: evaluate(clf, x, y)[1]
    with torch.no_grad():
        rec_ae_n, rec_dae_n = predict(clean_ae, x_noisy), predict(dae, x_noisy)
        rec_ae_c, rec_dae_c = predict(clean_ae, x_clean), predict(dae, x_clean)
    T = lambda r: F.mse_loss(r, x_clean).item()
    return [dict(input="clean image", clf_acc=acc(x_clean), mse_vs_clean=0.0),
            dict(input="noisy image", clf_acc=acc(x_noisy), mse_vs_clean=T(x_noisy)),
            dict(input="clean-AE recon(noisy)", clf_acc=acc(rec_ae_n), mse_vs_clean=T(rec_ae_n)),
            dict(input="DAE recon(noisy)", clf_acc=acc(rec_dae_n), mse_vs_clean=T(rec_dae_n)),
            dict(input="clean-AE recon(clean)", clf_acc=acc(rec_ae_c), mse_vs_clean=T(rec_ae_c)),   # info lost by the bottleneck alone
            dict(input="DAE recon(clean)", clf_acc=acc(rec_dae_c), mse_vs_clean=T(rec_dae_c))], rec_dae_n


print(f"\n=== Q3(b) VALIDATION sanity (sigma={SIGMA}) ===")
print_table(pipeline(data["xva"], xva_n, data["yva"])[0])
tab, rec_dae_n = pipeline(data["xte"], xte_n, data["yte"])
print("\n=== Q3(b) TEST ==="); print_table(tab)
a = {r["input"]: r["clf_acc"] for r in tab}
rec = (a["DAE recon(noisy)"] - a["noisy image"]) / max(a["clean image"] - a["noisy image"], 1e-9)
print(f"Fraction of the noise-induced accuracy loss recovered by the DAE: {rec:.2%} "
      f"(DAE recon(clean) acc {a['DAE recon(clean)']:.3f} shows the ceiling imposed by the bottleneck itself)")
save_json(dict(table=tab, recovery_fraction=rec, sigma=SIGMA, selected_latent=sel), f"{OUT}/q3b.json")

with torch.no_grad():
    fig, ax = plt.subplots(4, 8, figsize=(12, 6.2))
    rows_img = [data["xte"][:8], xte_n[:8], predict(clean_ae, xte_n[:8]), rec_dae_n[:8]]
    for i, (t, imgs) in enumerate(zip(["clean", "noisy", "clean-AE", "DAE"], rows_img)):
        for j in range(8): ax[i][j].imshow(imgs[j].permute(1, 2, 0).cpu()); ax[i][j].axis("off")
        ax[i][0].set_title(t, fontsize=8, loc="left")
    fig.savefig(f"{OUT}/q3b_denoising.png", dpi=150); plt.close(fig)

# ============================================================ (c)
@torch.no_grad()
def feats(m, x):
    m.eval(); return torch.cat([m.encode(x[i:i + 2000]) for i in range(0, len(x), 2000)])


def linear_probe(ftr, fva, fte):
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr, fva, fte = [(f - mu) / sd for f in (ftr, fva, fte)]
    best = None
    for wd in (0, 1e-4, 1e-3, 1e-2):
        set_seed(0); lin = nn.Linear(ftr.shape[1], 10).to(DEVICE)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=wd)
        for _ in range(40):
            perm = torch.randperm(len(ftr), device=DEVICE)
            for i in range(0, len(ftr), 512):
                idx = perm[i:i + 512]; loss = F.cross_entropy(lin(ftr[idx]), data["ytr"][idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): va = (lin(fva).argmax(1) == data["yva"]).float().mean().item()
        if best is None or va > best[0]: best = (va, wd, lin)
    va, wd, lin = best
    with torch.no_grad(): te = (lin(fte).argmax(1) == data["yte"]).float().mean().item()
    return dict(probe_val_acc=va, probe_wd=wd, probe_test_acc=te)


torch.manual_seed(0); rand_enc = ConvAE(sel).to(DEVICE)
models = {**{f"AE_d{d}": (aes[d], d) for d in LATENTS}, f"DAE_d{sel}": (dae, sel), f"random_enc_d{sel}": (rand_enc, sel)}
c_rows = []
for name, (m, d) in models.items():
    pr = linear_probe(*[feats(m, data[k]) for k in ("xtr", "xva", "xte")])
    c_rows.append(dict(model=name, latent_dim=d, test_mse_clean=(mse(m, data["xte"], data["xte"]) if "random" not in name else float("nan")), **pr))
raw = linear_probe(*[data[k].flatten(1) for k in ("xtr", "xva", "xte")])
c_rows.append(dict(model="raw_pixels", latent_dim=3072, test_mse_clean=float("nan"), **raw))
print("\n=== Q3(c) TEST: reconstruction vs class information (frozen encoders, linear probe) ==="); print_table(c_rows)

trained = [r for r in c_rows if not r["model"].startswith(("random", "raw"))]
rho = spearmanr([-r["test_mse_clean"] for r in trained], [r["probe_test_acc"] for r in trained])
print(f"Spearman(-MSE, probe acc) over the {len(trained)} trained models: {rho.correlation:.2f} "
      "(only 4 points - interpret as ordering agreement, not a statistic)")
fig, ax = plt.subplots(figsize=(5, 4))
for r in trained:
    ax.scatter(r["test_mse_clean"], r["probe_test_acc"]); ax.annotate(r["model"], (r["test_mse_clean"], r["probe_test_acc"]), fontsize=7)
ax.set_xlabel("test reconstruction MSE (lower=better)"); ax.set_ylabel("linear-probe test acc"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/q3c_mse_vs_probe.png", dpi=150)
save_json(dict(rows=c_rows, spearman=rho.correlation), f"{OUT}/q3c.json")
