"""Where does the pipeline lose points on OTHER matches? (Metrica public sample games.)

    python solution/external_report.py [--clean] [--games 1,2,3]

Per game: spellF1, boundary latency per kind (team switch / stoppage start / restart), and oracle
ablations (true dead-ball mask + our teams; our dead-ball mask + true teams).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "validation"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import pipeline as pl  # noqa: E402
import tune_external as te  # noqa: E402
import possession_metrics as pm  # noqa: E402

CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def codes(labels):
    return np.array([CODE[l] for l in labels], dtype=np.int8)


def spell_f1(gt, pr, runs):
    return pm.spell_scores(gt, pm.fill_abstentions(pr, runs), runs)["mean_f1"]


def fill_live(c, need_live):
    out = c.copy()
    live = np.where(c < 2, c, -1)
    prev = live.copy()
    for i in range(1, len(prev)):
        if prev[i] < 0:
            prev[i] = prev[i - 1]
    nxt = live.copy()
    for i in range(len(nxt) - 2, -1, -1):
        if nxt[i] < 0:
            nxt[i] = nxt[i + 1]
    fill = np.where(prev >= 0, prev, nxt)
    m = need_live & (c == 2)
    out[m] = fill[m]
    return out


def boundaries(c, runs):
    out = []
    for a, b in runs:
        s = c[a:b]
        for i in np.flatnonzero(s[1:] != s[:-1]) + 1:
            f, t = s[i - 1], s[i]
            kind = "switch" if (f < 2 and t < 2) else ("stop" if t == 2 else "restart")
            out.append((a + i, kind))
    return out


def latency_table(gt, pr, runs):
    gb, pb = boundaries(gt, runs), boundaries(pr, runs)
    rows = {}
    for kind in ("switch", "stop", "restart"):
        g = np.array([f for f, k in gb if k == kind]); p = np.array([f for f, k in pb if k == kind])
        if len(g) == 0:
            continue
        lat = np.array([(p[np.argmin(np.abs(p - f))] - f) if len(p) else 10 ** 6 for f in g])
        rows[kind] = dict(n_gt=len(g), n_pred=len(p), within_0_5s=float(np.mean(np.abs(lat) <= 12)),
                          within_1s=float(np.mean(np.abs(lat) <= 25)),
                          within_3s=float(np.mean(np.abs(lat) <= 75)),
                          median_abs_s=float(np.median(np.abs(lat)) / 25.0),
                          median_signed_s=float(np.median(lat[np.abs(lat) <= 125]) / 25.0)
                          if np.any(np.abs(lat) <= 125) else float("nan"))
    return rows


def report(P=None, ids=(1, 2, 3), degrade=True, ball_smooth=0):
    import metrica_sample as ms
    for g in ids:
        d = ms.load_game(g, degrade=degrade, seed=0)
        d = dict(d, ball=te.smooth_ball(np.asarray(d["ball"], dtype=np.float64), ball_smooth))
        labels, aux = pl.label_match(d, d["halves"], home_pos_first=d["home_attacks_positive_x_first_half"],
                                     P=P or {})
        gt, pr = codes(d["labels"]), codes(labels)
        runs = pm.valid_runs(gt)
        oa = fill_live(pr, gt != 2); oa[gt == 2] = 2
        ob = fill_live(gt, pr != 2); ob[pr == 2] = 2
        ng = len(pm.possession_spells(gt, runs)); npred = len(pm.possession_spells(pr, runs))
        print("game {}: spellF1 {:.3f} | oracle-D + our teams {:.3f} | our D + oracle teams {:.3f} | spells pred {} gt {}".format(
            d.get("game_id", "?"), spell_f1(gt, pr, runs), spell_f1(gt, oa, runs), spell_f1(gt, ob, runs), npred, ng))
        for kind, r in latency_table(gt, pr, runs).items():
            print("   {:8s} gt {:4d} pred {:4d} | within 0.5s {:.2f} 1s {:.2f} 3s {:.2f} | median |lat| {:.1f}s, signed {:+.1f}s".format(
                kind, r["n_gt"], r["n_pred"], r["within_0_5s"], r["within_1s"], r["within_3s"], r["median_abs_s"], r["median_signed_s"]))
        print("   events:", aux)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--games", default="1,2,3")
    ap.add_argument("--ball-smooth", type=int, default=0)
    args = ap.parse_args()
    report(ids=tuple(int(g) for g in args.games.split(",")), degrade=not args.clean,
           ball_smooth=args.ball_smooth)
