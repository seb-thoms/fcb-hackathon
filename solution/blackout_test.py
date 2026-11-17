"""Artificial-blackout test (label-free): hide the ball inside confidently-live possessions, rerun
the pipeline, and measure how much of the hidden stretch keeps the label the unmasked run gave it
(and how much turns into a false dead ball).

    python solution/blackout_test.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as pl  # noqa: E402


def choose_blocks(labels, present, halves, L, n_per_half, seed):
    """Blocks fully inside one decoded possession, ball visible on >= 90% of them."""
    rng = np.random.RandomState(seed)
    blocks = []
    for half, first, last in halves:
        taken, tries = [], 0
        while len(taken) < n_per_half and tries < 50000:
            tries += 1
            a = rng.randint(first + 250, last - L - 250)
            seg = labels[a - 50:a + L + 50]
            if seg[0] not in "HA" or np.any(seg != seg[0]):
                continue
            if present[a:a + L].mean() < 0.9 or any(abs(a - b) < L + 1500 for b in taken):
                continue
            taken.append(a)
        blocks += [(a, a + L, labels[a]) for a in taken]
    return blocks


def run(data, halves, kwargs, n_per_half=40, seed=0):
    base, _ = pl.label_match(data, halves, **kwargs)
    present = ~np.isnan(data["ball"][:, 0])
    for L_s in (5, 15, 30):
        L = L_s * 25
        blocks = choose_blocks(base, present, halves, L, n_per_half, seed)
        masked = data["ball"].copy()
        for a, b, _ in blocks:
            masked[a:b] = np.nan
        lab, _ = pl.label_match(dict(data, ball=masked), halves, **kwargs)
        same = np.mean([np.mean(lab[a:b] == t) for a, b, t in blocks])
        dead = np.mean([np.mean(lab[a:b] == "D") for a, b, t in blocks])
        print("blackout {:>2}s x {:>2}: label kept {:.3f}  turned dead {:.3f}  flipped team {:.3f}".format(
            L_s, len(blocks), same, dead, 1 - same - dead))


if __name__ == "__main__":
    import atd
    import load_smart_tagging
    import broadcast
    from games import GAME, half_video_blocks
    d = atd.load()
    d = dict(d, pitch=(atd.PITCH_L, atd.PITCH_W))
    tags = load_smart_tagging.load()
    boff, _ = broadcast.broadcast_off(d["n_frames"])
    run(d, half_video_blocks(GAME), dict(home_pos_first=True, tags=tags, bcast_off=boff))
