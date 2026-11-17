"""Choose parameters on OTHER matches (Metrica public sample games) — never on BAR-ATH labels.

    python solution/tune_external.py                                   # score the defaults
    python solution/tune_external.py --ab '{"name": {"param": value}, ...}'
    python solution/tune_external.py --sweep '{"param": [v1, v2], ...}' [--base '{...}']

Scores = mean spellF1 over games 1-3 x seeds, for both event-derived label rules ("control": possession
changes on controlled possession; "touch": on any opponent touch), at two ball-smoothing levels
(0 = harness ball, 8 = smoothed towards the BAR-ATH ATD ball's acceleration profile).
Smart Tagging and broadcast inputs do not exist for these games: this measures the tracking core.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "validation"))
import pipeline as pl  # noqa: E402
import metrica_sample as ms  # noqa: E402

FPS = 25


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


_GAMES = {}


def games(seeds, smooth):
    key = (tuple(seeds), smooth)
    if key not in _GAMES:
        out = []
        for s in seeds:
            for g in (1, 2, 3):
                dc = ms.load_game(g, degrade=True, seed=s, rule="control")
                dt = ms.load_game(g, degrade=True, seed=s, rule="touch")
                out.append((dict(dc, ball=smooth_ball(np.asarray(dc["ball"], dtype=np.float64), smooth)), dt))
        _GAMES[key] = out
    return _GAMES[key]


def score(P, seeds=(0, 1, 2), smooth=0):
    c, t, m = [], [], []
    for dc, dt in games(seeds, smooth):
        lab = pl.label_match(dc, dc["halves"], dc["home_attacks_positive_x_first_half"], P=P)[0]
        sc = ms.score(lab, dc)
        c.append(sc["spell_f1"]); m.append(sc["macro_f1"])
        t.append(ms.score(lab, dt)["spell_f1"])
    return dict(control=float(np.mean(c)), touch=float(np.mean(t)), macro=float(np.mean(m)),
                worst=float(np.min(c)))


def fmt(r):
    return "control {control:.4f}  touch {touch:.4f}  macro {macro:.4f}  worst {worst:.3f}".format(**r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", help="JSON {name: params}")
    ap.add_argument("--sweep", help="JSON {param: [values]} (one at a time from --base)")
    ap.add_argument("--base", default="{}")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smooth", default="0,8")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    smooths = [int(s) for s in args.smooth.split(",")]
    base = json.loads(args.base)
    for sm in smooths:
        if args.ab:
            for name, P in json.loads(args.ab).items():
                print("smooth {:>2} {:30s} {}".format(sm, name, fmt(score(P, seeds, sm))), flush=True)
        elif args.sweep:
            ref = score(base, seeds, sm)
            print("smooth {:>2} base {}".format(sm, fmt(ref)), flush=True)
            for name, values in json.loads(args.sweep).items():
                for v in values:
                    r = score(dict(base, **{name: v}), seeds, sm)
                    print("smooth {:>2}   {}={}: {}  ({:+.4f} / {:+.4f})".format(
                        sm, name, v, fmt(r), r["control"] - ref["control"], r["touch"] - ref["touch"]), flush=True)
        else:
            print("smooth {:>2} defaults {}".format(sm, fmt(score({}, seeds, sm))), flush=True)
