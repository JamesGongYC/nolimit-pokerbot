"""Train a small transformer Q-model on real preflop decisions (kit venv: torch).

  starter-kit/.venv/bin/python tools/tftrain.py out.npz data1.npz [data2.npz ...]

Tokens (sequence of 1 + 3 + H):
  [CTX]  linear projection of the context features (also the read-out token)
  card   rank embedding + suit-pattern embedding (suits relabelled by first
         appearance, so the model is suit-symmetric) + slot embedding
  hist   linear projection of each prior preflop action (padding masked)
2 pre-norm encoder layers, d=32, 4 heads, FF 64. Output: Q for fold / call /
raise / all-in = expected change in our ICM points for the hand. Loss: MSE on
the action actually taken (other heads unsupervised). Exports numpy weights
plus a held-out report; tools/tfinfer.py reproduces the forward pass in numpy.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

D, NH, FF, L = 32, 4, 64, 2


def suit_pattern(cards):
    """cards: (N,3) int -> (N,3) suit ids relabelled by first appearance."""
    s = cards % 4
    out = np.zeros_like(s)
    for i in range(s.shape[0]):
        m = {}
        for j in range(3):
            out[i, j] = m.setdefault(int(s[i, j]), len(m))
    return out


class Model(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.rank = nn.Embedding(13, D)
        self.suit = nn.Embedding(3, D)
        self.slot = nn.Embedding(3, D)
        self.ctx = nn.Linear(C, D)
        self.hist = nn.Linear(4, D)
        self.hpos = nn.Parameter(torch.zeros(H, D))
        layer = nn.TransformerEncoderLayer(D, NH, FF, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 4)

    def forward(self, ranks, suits, ctx, hist, hmask):
        B = ranks.shape[0]
        slots = torch.arange(3).expand(B, 3)
        c = self.rank(ranks) + self.suit(suits) + self.slot(slots)
        x = torch.cat([self.ctx(ctx).unsqueeze(1), c, self.hist(hist) + self.hpos], 1)
        pad = torch.cat([torch.zeros(B, 4, dtype=torch.bool), hmask], 1)
        y = self.enc(x, src_key_padding_mask=pad)
        return self.head(self.norm(y[:, 0]))


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    # each argument is data.npz or data.npz@weight (per-row sample weight, e.g. recency)
    paths = [a.split("@")[0] for a in specs]
    wts = [float(a.split("@")[1]) if "@" in a else 1.0 for a in specs]
    data = [np.load(p) for p in paths]
    rw = np.concatenate([np.full(len(d["action"]), w, np.float32) for d, w in zip(data, wts)])
    newest = np.concatenate([np.full(len(d["action"]), i == len(data) - 1, bool) for i, d in enumerate(data)])
    cards = np.concatenate([d["cards"] for d in data]).astype(np.int64)
    ctx = np.concatenate([d["ctx"] for d in data]).astype(np.float32)
    hist = np.concatenate([d["hist"] for d in data]).astype(np.float32)
    act = np.concatenate([d["action"] for d in data]).astype(np.int64)
    rew = np.concatenate([d["reward"] for d in data]).astype(np.float32)
    # rows where the action is nothing but "check" in the big blind carry no decision information
    n = len(act)
    ranks = cards // 4
    suits = suit_pattern(cards)
    hmask = (np.abs(hist).sum(-1) == 0)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    va, tr = idx[:n // 20], idx[n // 20:]
    T = lambda a, i: torch.from_numpy(a[i])
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = Model(ctx.shape[1], hist.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 4096
    t0 = time.time()
    epochs = int(__import__("os").environ.get("TF_EPOCHS", "2"))
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    step = 0
    for ep in range(epochs):
        perm = tr[rng.permutation(len(tr))]
        for b in range(0, len(perm) - bs + 1, bs):
            i = perm[b:b + bs]
            q = model(T(ranks, i), T(suits, i), T(ctx, i), T(hist, i), T(hmask, i))
            pred = q.gather(1, T(act, i).unsqueeze(1)).squeeze(1)
            w = T(rw, i)
            loss = (w * (pred - T(rew, i)) ** 2).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print("ep %d step %d/%d loss %.3f %.0fs" % (ep, step, steps, loss.item(), time.time() - t0), flush=True)
    model.eval()
    with torch.no_grad():
        preds = []
        for b in range(0, len(va), 8192):
            i = va[b:b + 8192]
            preds.append(model(T(ranks, i), T(suits, i), T(ctx, i), T(hist, i), T(hmask, i)).numpy())
        Q = np.concatenate(preds)
    for a in range(4):
        m = (act[va] == a) & newest[va]
        if m.sum() >= 100:
            y, p = rew[va][m], Q[m, a]
            print("NEWEST-round held-out action %d: n=%d r2=%.4f mean_y %.3f mean_pred %.3f" % (
                a, m.sum(), 1 - ((y - p) ** 2).mean() / max(1e-9, y.var()), y.mean(), p.mean()))
    for a in range(4):
        m = act[va] == a
        if m.sum() < 100:
            continue
        y, p = rew[va][m], Q[m, a]
        r2 = 1 - ((y - p) ** 2).mean() / max(1e-9, y.var())
        print("held-out action %d: n=%d r2=%.4f mean_y %.3f mean_pred %.3f" % (a, m.sum(), r2, y.mean(), p.mean()))
    W = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(out, **W)
    torch.save(model.state_dict(), out + ".pt")
    print("saved", out, "params", sum(v.size for v in W.values()), "train %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
