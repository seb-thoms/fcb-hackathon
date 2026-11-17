"""Artificial-blackout test (label-free): hide the ball where it was confidently visible, rerun the
pipeline, and check how often the decoded label recovers the algorithmic pseudo-label.

    python solution/blackout_test.py
Truth here is the pseudo-label from the *unmasked* ball, so this measures the ball-blind logic
only, and is optimistic (easy frames). Public GT is not used."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))
sys.path.insert(0, os.path.dirname(HERE))
import atd, load_smart_tagging  # noqa: E402
import pipeline as pl  # noqa: E402
import shape_model  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402


def truth(data, tags, P):
    scores, aux = pl.frame_scores(data, tags, P)
    path = pl.decode_all(scores, tags)
    ball_i = pl.interp_short_gaps(data["ball"], P["interp_gap"])
    return shape_model.pseudo_labels(ball_i, data["players"], data["team"], aux["bspeed"], path, P), aux


def choose_blocks(y, present, L, n_per_half, seed):
    rng = np.random.RandomState(seed)
    blocks = []
    for half, first, last in half_video_blocks(GAME):
        taken = []
        tries = 0
        while len(taken) < n_per_half and tries < 20000:
            tries += 1
            a = rng.randint(first + 100, last - L - 100)
            if any(abs(a - b) < L + 1500 for b in taken):
                continue
            seg = slice(a, a + L)
            if present[seg].mean() < 0.9 or (y[seg] >= 0).mean() < 0.3:
                continue
            taken.append(a)
        blocks += [(a, a + L) for a in taken]
    return blocks


def evaluate(path, y, blocks):
    ok = tot = sw_pred = sw_true = 0
    for a, b in blocks:
        yy, pp = y[a:b], path[a:b]
        k = yy >= 0
        ok += int(((pp == 0) & (yy == 1) | (pp == 1) & (yy == 0))[k].sum())
        tot += int(k.sum())
        sw_pred += int(np.sum((pp[1:] != pp[:-1]) & (pp[1:] < 2) & (pp[:-1] < 2)))
        t = yy[k]
        sw_true += int(np.sum(t[1:] != t[:-1])) if len(t) > 1 else 0
    dead = sum(int((path[a:b][y[a:b] >= 0] == 2).sum()) for a, b in blocks)
    return ok / max(tot, 1), tot, sw_pred, sw_true, dead / max(tot, 1)


if __name__ == "__main__":
    data = atd.load()
    tags = load_smart_tagging.load()
    base = pl.PARAMS
    y, aux = truth(data, tags, base)
    present = aux["present"]
    print("pseudo-labelled frames: {} ({:.1f}% of graded)".format((y >= 0).sum(), 100 * (y >= 0).sum() / 158857))
    for L_s in (5, 15):
        L = L_s * 25
        blocks = choose_blocks(y, present, L, 40, seed=0)
        print("\nblackout {}s, {} blocks".format(L_s, len(blocks)))
        masked = data["ball"].copy()
        for a, b in blocks:
            masked[a:b] = np.nan
        for w in (0.0, 0.05, 0.1, 0.2, 0.4):
            P = dict(base, shape_weight=w)
            labels, _ = pl.run(P, data=data, tags=tags, ball=masked)
            path = np.where(labels == "X", -1, np.array([{"H": 0, "A": 1, "D": 2}.get(l, -1) for l in labels]))
            acc, tot, sp, st, dd = evaluate(path, y, blocks)
            print("  shape_weight {:<5} agreement {:.3f} (n={})  decoded-D {:.2f}  live switches: pred {} vs pseudo-truth {}".format(w, acc, tot, dd, sp, st))
