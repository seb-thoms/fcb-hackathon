"""Proposal A: rule-based possession state machine, decoded with a sticky 3-state Viterbi.

    python solution/pipeline.py --out evaluation/results/solution_v1.csv

Per-frame evidence -> per-frame scores for H / A / D -> Viterbi inside each graded half.
All constants are set a priori from football knowledge (see PARAMS); none is tuned on the public
window. Deterministic: numpy only, no randomness.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import atd  # noqa: E402
import load_smart_tagging  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = dict(
    interp_gap=10,          # frames; bridge detector flicker only
    ball_speed_win=5,       # frames for ball velocity
    control_radius=3.0,     # m; evidence decays with exp(-d / radius)
    max_live_speed=25.0,    # m/s; faster ball is a clear pass/shot, ignore as control evidence
    pass_speed=9.0,         # m/s; above this the ball is in flight -> weak control evidence
    switch_penalty=8.0,     # nats for H<->A: ~ sustained control for >= ~0.5-1 s
    dead_penalty=6.0,       # nats to enter / leave D
    oob_margin=1.0,         # m outside the lines before the ball counts as out
    still_lo=1.1,           # m/s mean player speed (smoothed) at/below which D is favoured
    still_hi=2.0,           # m/s at/above which live is favoured
    speed_smooth=75,        # frames
)


# ----------------------------------------------------------------------------- features
def interp_short_gaps(ball, max_gap):
    out = ball.copy()
    n = len(out)
    present = ~np.isnan(out[:, 0])
    idx = np.flatnonzero(present)
    if len(idx) < 2:
        return out
    gaps = np.diff(idx)
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[k], idx[k + 1]
        t = (np.arange(a + 1, b) - a) / (b - a)
        out[a + 1:b] = out[a] + t[:, None] * (out[b] - out[a])
    return out


def moving_mean(x, win):
    """Centred moving mean ignoring NaN; NaN where the window has no data."""
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - win), np.minimum(len(x), i + win + 1)
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def ball_speed(ball, win):
    n = len(ball)
    sp = np.full(n, np.nan)
    d = np.linalg.norm(ball[win:] - ball[:-win], axis=1) * FPS / win
    sp[win // 2: win // 2 + len(d)] = d
    return sp


def player_speed(players, win=5):
    """Mean speed (m/s) over tracked players per frame (slots tracked at both ends)."""
    d = np.linalg.norm(players[win:] - players[:-win], axis=2) * FPS / win  # (n-win, S)
    m = np.nanmean(np.where(d < 12, d, np.nan), axis=1)
    out = np.full(len(players), np.nan)
    out[win // 2: win // 2 + len(m)] = m
    return out


def nearest_by_team(players, ball, team):
    d = np.linalg.norm(players - ball[:, None, :], axis=2)  # (n, S), NaN where unseen
    big = np.float32(1e3)
    dh = np.nanmin(np.where(np.isnan(d[:, team == 0]), big, d[:, team == 0]), axis=1)
    da = np.nanmin(np.where(np.isnan(d[:, team == 1]), big, d[:, team == 1]), axis=1)
    return dh, da


def frame_scores(data, tags, P=PARAMS):
    """Per-frame log-score for each state, shape (n, 3). Higher = more likely."""
    ball = interp_short_gaps(data["ball"], P["interp_gap"])
    present = ~np.isnan(ball[:, 0])
    n = len(ball)

    bspeed = ball_speed(ball, P["ball_speed_win"])
    dh, da = nearest_by_team(data["players"], ball, data["team"])
    dmin = np.minimum(dh, da)

    # --- H/A evidence: signed log-likelihood ratio favouring H (>0) or A (<0), ball visible.
    diff = da - dh                                   # >0: home player is closer
    llr = np.tanh(diff / 2.0)                        # saturates once one side is clearly closer
    weight = np.exp(-dmin / P["control_radius"])     # far from everybody -> no information
    flight = np.where(np.isnan(bspeed), 0.0, np.clip((bspeed - P["pass_speed"]) / 6.0, 0, 1))
    weight = weight * (1.0 - 0.8 * flight)
    e = np.where(present & (dmin < 1e2), llr * weight, 0.0)
    e = np.where(np.isnan(e), 0.0, e)

    # --- Dead-ball evidence: positive favours D, negative favours live.
    dscore = np.zeros(n)
    # ball clearly outside the lines
    oob = present & ((ball[:, 0] < -P["oob_margin"]) | (ball[:, 0] > atd.PITCH_L + P["oob_margin"]) |
                     (ball[:, 1] < -P["oob_margin"]) | (ball[:, 1] > atd.PITCH_W + P["oob_margin"]))
    dscore += np.where(oob, 1.5, 0.0)
    # ball visibly moving fast => live
    dscore -= np.where(present & (np.nan_to_num(bspeed) > P["pass_speed"]), 1.0, 0.0)
    # global stillness of the players (smoothed): walking => stoppage, running => play
    ps = moving_mean(player_speed(data["players"]), P["speed_smooth"])
    still = np.clip((P["still_hi"] - np.nan_to_num(ps, nan=P["still_hi"])) /
                    (P["still_hi"] - P["still_lo"]), -1, 1)   # +1 very still, -1 clearly running
    dscore += 0.25 * still

    scores = np.zeros((n, 3))
    scores[:, H] = 0.5 * e
    scores[:, A] = -0.5 * e
    scores[:, D] = dscore
    return scores, dict(present=present, bspeed=bspeed, ps=ps, dmin=dmin, oob=oob)


# ----------------------------------------------------------------------------- decoding
def viterbi(scores, P=PARAMS, start_dead_until=0):
    """Max-score path over 3 states with sticky transitions. scores: (n,3)."""
    n = len(scores)
    trans = np.zeros((3, 3))
    trans[H, A] = trans[A, H] = -P["switch_penalty"]
    trans[H, D] = trans[A, D] = trans[D, H] = trans[D, A] = -P["dead_penalty"]
    s = scores.copy()
    if start_dead_until > 0:
        s[:start_dead_until, H] = -1e3
        s[:start_dead_until, A] = -1e3
    dp = np.zeros((n, 3))
    bp = np.zeros((n, 3), dtype=np.int8)
    dp[0] = s[0]
    for t in range(1, n):
        cand = dp[t - 1][:, None] + trans            # [from, to]
        bp[t] = np.argmax(cand, axis=0)
        dp[t] = cand[bp[t], np.arange(3)] + s[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(dp[-1])
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t][path[t]]
    return path


def kickoff_frame(tags, half_label):
    """Centre of the 'Kick Off (Start)' tag of the half: the tag window contains the restart."""
    for t in tags:
        if t["code"].lower() == "set pieces" and \
                t["tags"].get("Type", "").lower() == "kick off (start)" and \
                t["tags"].get("Half") == half_label:
            return (t["start_frame"] + t["end_frame"]) // 2
    return None


def run():
    data = atd.load()
    n = data["n_frames"]
    tags = load_smart_tagging.load()
    scores, aux = frame_scores(data, tags)
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        label = "1st Half" if half == 1 else "2nd Half"
        ko = kickoff_frame(tags, label)
        dead_until = max(0, ko - first) if ko else 0
        path = viterbi(scores[first:last + 1], start_dead_until=dead_until)
        labels[first:last + 1] = LABELS[path]
    return labels, aux


def spell_stats(labels, first_last):
    out = []
    for first, last in first_last:
        seg = labels[first:last + 1]
        runs = [(seg[i], 0) for i in range(len(seg)) if i == 0 or seg[i] != seg[i - 1]]
        spells = sum(1 for r in runs if r[0] in "HA")
        out.append((spells, int((seg == "D").sum()), int((seg == "H").sum()), int((seg == "A").sum())))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "solution_v1.csv"))
    args = ap.parse_args()
    labels, _ = run()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    blocks = [(f, l) for _, f, l in half_video_blocks(GAME)]
    for (sp, d, h, a), (f, l) in zip(spell_stats(labels, blocks), blocks):
        print("half {}-{}: spells={} mean={:.1f}s  H={} A={} D={}".format(
            f, l, sp, (h + a) / max(sp, 1) / FPS, h, a, d))
    print("wrote", args.out)
