"""Broadcast feed as a dead-ball signal, from the two homographies alone.

The TV feed cuts to replays and close-ups when play stops. Where the broadcast homography is
invalid (no pitch view), or its camera points somewhere else than the tactical camera, the
broadcast is "off" -- and that happens far more often during stoppages than during play.

Alignment is estimated here, not hard-coded: each homography gives the pitch point at the centre of
its image; one constant offset per half (tactical = broadcast + offset) is found by sliding the
broadcast camera track against the tactical one and minimising the median distance.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
IMAGE_CENTRE = (960.0, 540.0)            # both recordings are 1920x1080
OFF_DISTANCE = 15.0                      # m along the pitch between the two view centres -> other shot
# Only x (along the pitch) is compared: the broadcast camera sits elsewhere, so its view-centre y is
# systematically offset from the tactical one even when both show the same play.
MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]


def view_centres(source):
    """(n, 2) pitch point (metres) at the image centre; NaN where no valid homography."""
    path = require(["homography"], GAME, None, source)[0]
    table = pd.read_csv(path)
    valid = table["valid"].values.astype(bool)
    mats = table[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
    out = np.full((len(table), 2), np.nan)
    idx = np.flatnonzero(valid)
    inv = np.linalg.inv(mats[idx])
    p = np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
    q = inv @ p
    out[idx, 0] = q[:, 0] / q[:, 2] * PITCH_L
    out[idx, 1] = q[:, 1] / q[:, 2] * PITCH_W
    # a centre far outside the pitch is a camera pointed at the stands
    bad = (out[:, 0] < -20) | (out[:, 0] > PITCH_L + 20) | (out[:, 1] < -20) | (out[:, 1] > PITCH_W + 20)
    out[bad] = np.nan
    return out


def estimate_offset(tac, bro, first, last, lo=-60000, hi=60000, step=25, sub=5):
    """Offset (tactical = broadcast + offset) minimising the median |dx| of the view centres."""
    t = np.arange(first, last + 1, sub)
    tx = tac[t, 0]

    def cost(off):
        b = t - off
        ok = (b >= 0) & (b < len(bro))
        if ok.sum() < len(t) * 0.2:
            return np.inf
        d = np.abs(tx[ok] - bro[b[ok], 0])
        d = d[~np.isnan(d)]
        return np.median(d) if len(d) > len(t) * 0.2 else np.inf

    coarse = np.arange(lo, hi + 1, step)
    costs = np.array([cost(o) for o in coarse])
    best = coarse[int(np.argmin(costs))]
    fine = np.arange(best - step, best + step + 1)
    fcost = np.array([cost(o) for o in fine])
    return int(fine[int(np.argmin(fcost))]), float(fcost.min())


def broadcast_off(n_frames, verbose=False):
    """Per tactical frame: 1.0 broadcast off (replay / close-up / other shot), 0.0 on (same view),
    NaN where the broadcast does not cover that frame."""
    tac = view_centres("tactical")
    bro = view_centres("broadcast")
    off = np.full(n_frames, np.nan)
    offsets = {}
    for half, first, last in half_video_blocks(GAME):
        o, c = estimate_offset(tac, bro, first, last)
        offsets[half] = (o, c)
        t = np.arange(first, last + 1)
        b = t - o
        ok = (b >= 0) & (b < len(bro))
        bc = np.full((len(t), 2), np.nan)
        bc[ok] = bro[b[ok]]
        dist = np.abs(bc[:, 0] - tac[t, 0])
        val = np.where(np.isnan(bc[:, 0]), 1.0, (dist > OFF_DISTANCE).astype(float))
        val[~ok] = np.nan
        off[t] = val
    if verbose:
        print("broadcast offsets (tactical = broadcast + offset, median |dx| m):", offsets)
    return off, offsets


if __name__ == "__main__":
    from games import source_frames
    off, offsets = broadcast_off(source_frames("tactical", GAME), verbose=True)
    g = ~np.isnan(off)
    print("broadcast off on {:.1f}% of covered graded frames".format(100 * np.nanmean(off)))
