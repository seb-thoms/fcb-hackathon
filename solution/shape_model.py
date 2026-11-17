"""Proposal B: ball-blind team-shape model, self-trained on algorithmic pseudo-labels.

The ball is visible on ~59% of frames; where one team is clearly in control of it (and Proposal A's
decoder agrees) we know who is in possession. A gradient-boosting model learns what a team in
possession *looks like* from player positions alone (no ball features), and is then applied on
frames where the ball is missing. Cross-half training: the model that labels half h never saw half
h's pseudo-labels. No external data, no hand labels, no public-GT use.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402

FPS = 25
WINDOWS = (25, 75, 200)  # +/- 1 s, 3 s, 8 s


def moving_mean(x, win):
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - win), np.minimum(len(x), i + win + 1)
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _team_stats(players, mask, sign):
    """Per-frame centroid / spread of one team's outfield players, attack-normalised (+x = towards
    the opponent goal). sign: (n,) array of +1 / -1 giving this team's attack direction."""
    x = (players[:, mask, 0] - atd.PITCH_L / 2) * sign[:, None]
    y = players[:, mask, 1] - atd.PITCH_W / 2
    cnt = np.sum(~np.isnan(x), axis=1)
    with np.errstate(all="ignore"):
        cx = np.nanmean(x, axis=1)
        sx = np.nanstd(x, axis=1)
        sy = np.nanstd(y, axis=1)
        cy = np.nanmean(y, axis=1)
        # depth of the back line / front line: robust to a stray player
        lo = np.nanpercentile(x, 20, axis=1)
        hi = np.nanpercentile(x, 80, axis=1)
    cnt_ok = cnt >= 5
    out = [np.where(cnt_ok, v, np.nan) for v in (cx, sx, sy, cy, lo, hi)]
    # forward velocity of the centroid over 1 s
    v = np.full(len(cx), np.nan)
    v[FPS // 2: FPS // 2 + len(cx) - FPS] = out[0][FPS:] - out[0][:-FPS]
    return out + [v]


def build_features(data, half_of_frame):
    """(n, F) feature matrix. half_of_frame: (n,) int (1 or 2; any value outside halves ok)."""
    players, team, gk = data["players"], data["team"], data["gk"]
    n = len(players)
    sign_h = np.where(half_of_frame == 1, 1.0, -1.0)   # H attacks +x in the 1st half
    sh = _team_stats(players, (team == 0) & ~gk, sign_h)
    sa = _team_stats(players, (team == 1) & ~gk, -sign_h)
    base = list(sh) + list(sa)
    # relative features (own-coordinates: positive = home further forward than away is deep, etc.)
    base += [sh[0] - sa[0], sh[1] - sa[1], sh[6] - sa[6], sh[4] - sa[4], sh[5] - sa[5]]
    base = [b.astype(np.float64) for b in base]
    feats = [np.stack(base, axis=1)]
    for w in WINDOWS:
        feats.append(np.stack([moving_mean(b, w) for b in base], axis=1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def pseudo_labels(ball_interp, players, team, bspeed, decoded, P):
    """+1 home / 0 away / -1 unlabelled, from confident ball-visible control that Proposal A's
    decoded path (0=H,1=A,2=D) agrees with and that is stable over +/- 2 s."""
    n = len(ball_interp)
    d = np.linalg.norm(players - ball_interp[:, None, :], axis=2)
    big = 1e3
    dh = np.min(np.where(np.isnan(d[:, team == 0]), big, d[:, team == 0]), axis=1)
    da = np.min(np.where(np.isnan(d[:, team == 1]), big, d[:, team == 1]), axis=1)
    dmin = np.minimum(dh, da)
    dother = np.maximum(dh, da)
    present = ~np.isnan(ball_interp[:, 0])
    slow = np.nan_to_num(bspeed, nan=99.0) < P["pass_speed"]
    ctrl = present & slow & (dmin < 1.5) & (dother > dmin + 2.5)
    c = np.full(n, -1, dtype=np.int8)
    c[ctrl & (dh < da)] = 1
    c[ctrl & (da < dh)] = 0
    # stability: every control frame within +/- 2 s agrees (and there are enough of them)
    w = 50
    ch = moving_mean((c == 1).astype(float) * 1.0 + np.where(c == -1, np.nan, 0.0), w)
    cnt = moving_mean((c != -1).astype(float), w) * (2 * w + 1)
    # ch is the share of control frames in the window that are home (NaN-aware)
    stable_h = (ch > 0.999) & (cnt >= 10)
    stable_a = (ch < 0.001) & (cnt >= 10)
    y = np.full(n, -1, dtype=np.int8)
    y[stable_h & (decoded == 0)] = 1
    y[stable_a & (decoded == 1)] = 0
    return y


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)


def cross_half_logit(X, y, blocks):
    """Per-frame logit P(H) for every graded frame, each half predicted by a model trained only
    on the *other* half's pseudo-labels. blocks: [(half, first, last)]. Also returns diagnostics."""
    n = len(X)
    logit = np.full(n, np.nan)
    diag = {}
    for half, first, last in blocks:
        other = [b for b in blocks if b[0] != half][0]
        sl_tr = slice(other[1], other[2] + 1)
        Xtr, ytr = X[sl_tr], y[sl_tr]
        keep = (ytr >= 0) & ~np.isnan(Xtr).any(axis=1)
        # NaN is allowed by HistGradientBoosting, so only the label filter is required
        keep = ytr >= 0
        model = make_model()
        model.fit(Xtr[keep], ytr[keep], sample_weight=_balanced(ytr[keep]))
        sl = slice(first, last + 1)
        p = model.predict_proba(X[sl])[:, 1]
        p = np.clip(p, 1e-3, 1 - 1e-3)
        logit[sl] = np.log(p / (1 - p))
        # held-out accuracy on this half's own pseudo-labels
        yt = y[sl]
        k = yt >= 0
        diag[half] = dict(train=int(keep.sum()), heldout=int(k.sum()),
                          heldout_acc=float(((p[k] > 0.5) == (yt[k] == 1)).mean()))
    return logit, diag


def _balanced(y):
    w = np.ones(len(y))
    for c in (0, 1):
        m = y == c
        if m.any():
            w[m] = len(y) / (2.0 * m.sum())
    return w
