"""Score pipeline variants on OTHER matches (Metrica public sample games) — never on BAR-ATH labels.

    python solution/tune_external.py                 # default parameters
    python solution/tune_external.py --sweep         # one-at-a-time sweep over key parameters

Smart Tagging and broadcast inputs do not exist for these games, so this measures the
tracking-only core (touches, control, crossings, slow-downs, restarts, decoder costs).
"""
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "validation"))
import pipeline as pl  # noqa: E402
import metrica_sample as ms  # noqa: E402

FPS = 25
_CACHE = {}


def smooth_ball(ball, win):
    """Centred moving average of the ball track (the BAR-ATH ATD ball is visibly smoothed)."""
    if win <= 0:
        return ball
    out = ball.copy()
    for c in range(2):
        x = ball[:, c]
        valid = ~np.isnan(x)
        v = np.where(valid, x, 0.0)
        cs = np.concatenate(([0.0], np.cumsum(v)))
        cn = np.concatenate(([0], np.cumsum(valid)))
        i = np.arange(len(x))
        lo, hi = np.maximum(0, i - win), np.minimum(len(x), i + win + 1)
        n = cn[hi] - cn[lo]
        out[:, c] = np.where(valid & (n > 0), (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return out


def games(ids, degrade, seed, ball_smooth):
    key = (tuple(ids), degrade, seed, ball_smooth)
    if key not in _CACHE:
        out = []
        for g in ids:
            d = ms.load_game(g, degrade=degrade, seed=seed)
            d = dict(d, ball=smooth_ball(np.asarray(d["ball"], dtype=np.float64), ball_smooth))
            out.append(d)
        _CACHE[key] = out
    return _CACHE[key]


def score_params(P, ids=(1, 2, 3), degrade=True, seed=0, ball_smooth=0, verbose=False):
    res = []
    for d in games(ids, degrade, seed, ball_smooth):
        labels, aux = pl.label_match(d, d["halves"], home_pos_first=d["home_attacks_positive_x_first_half"],
                                     tags=None, bcast_off=None, P=P)
        s = ms.score(labels, d)
        res.append(s)
        if verbose:
            print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                          if not isinstance(v, (dict, list))}, aux)
    keys = [k for k, v in res[0].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in res])) for k in keys}, res


SWEEP = {
    "touch_k": [3, 5, 7],
    "touch_dv": [2.0, 3.0, 4.0],
    "touch_r": [2.0, 2.5, 3.0],
    "ctrl_min": [10, 20, 30],
    "ctrl_episode": [5, 8, 12],
    "pen_switch": [8.0, 12.0, 16.0],
    "pen_switch_event": [0.5, 1.5, 3.0],
    "w_runteam": [0.1, 0.25, 0.5],
    "w_ctrl": [0.25, 0.5, 1.0],
    "pen_in": [6.0, 10.0, 14.0],
    "pen_out": [12.0, 20.0, 30.0],
    "w_still": [0.2, 0.4, 0.8],
    "still_mid": [1.3, 1.5, 1.7],
    "min_dead": [25, 50, 75],
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--clean", action="store_true", help="undegraded tracking")
    ap.add_argument("--games", default="1,2,3")
    ap.add_argument("--ball-smooth", type=int, default=0)
    args = ap.parse_args()
    ids = tuple(int(g) for g in args.games.split(","))
    base, _ = score_params({}, ids, not args.clean, 0, args.ball_smooth, verbose=True)
    print("default:", {k: round(v, 4) for k, v in base.items()})
    if args.sweep:
        for name, values in SWEEP.items():
            row = []
            for v in values:
                m, _ = score_params({name: v}, ids, not args.clean, 0, args.ball_smooth)
                row.append("{}={}: spell {:.4f} macro {:.4f}".format(
                    name, v, m.get("spell_f1", m.get("spellF1", np.nan)), m.get("macro_f1", m.get("macroF1", np.nan))))
            print(" | ".join(row))
