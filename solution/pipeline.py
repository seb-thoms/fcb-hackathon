"""Possession labelling: event-gated Viterbi over H / A / D. No learning.

    python solution/pipeline.py                      # writes evaluation/results/solution_final.csv

1. Clean player tracks (carry lost players forward, drop ghosts/duplicates).
2. Detect events from the tracking: ball touches and control (who has the ball), line crossings,
   balls lost near a line, collective slow-downs (fouls), restarts (ball still, then kicked).
3. Per-frame scores: team evidence (H vs A) and dead-ball evidence (ball out, stillness, broadcast
   replays/close-ups, set-piece tags), against live evidence (fast ball, players on the ball).
4. Decode each half with a Viterbi whose transition costs depend on the frame: a team switch is
   cheap only at the first touch of the new team's controlled run, entering D only at a crossing /
   ball lost at a line / slow-down onset, leaving D only at a detected restart.
5. Clean-up, then place switches inside live ball blackouts where both teams' shapes reverse.

Smart Tagging and the broadcast feed are optional inputs (the core runs on tracking alone).
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
import events as ev  # noqa: E402
import tracks  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = dict(
    # touches and control
    touch_k=5, touch_dv=3.0, touch_r=2.5,   # the ATD ball is smoothed: changes span ~0.4 s
    ctrl_r=1.5, ctrl_relv=3.0, ctrl_margin=0.5, ctrl_episode=8, ctrl_min=20, run_gap=5 * FPS,
    # dead-ball entry
    out_margin=0.3, lost_gap_min=2 * FPS, lost_dist=4.0, lost_speed=3.0,
    slow_before=1.6, slow_after=1.3, slow_drop=0.6,
    # restarts
    still_speed=1.0, still_frames=8, kick_within=8, kick_speed=4.0, taker_r=3.0,
    reappear_gap=3 * FPS,
    onset_pre=15, onset_post=12, onset_pre_speed=1.8, onset_post_speed=5.0,
    lost_any_gap=3 * FPS,
    # emissions
    w_runteam=0.25, w_ctrl=0.5,
    w_oob=1.0, w_fast=0.4, w_dribble=0.3, w_still=0.4, still_mid=1.5, still_half=0.5,
    w_boff=0.3, w_bon=0.25, w_tag=0.3, w_longgap=0.1,
    # transitions (costs in nats)
    pen_switch=12.0, pen_switch_event=1.5,
    pen_in=10.0, pen_in_cross=0.5, pen_in_lost=2.0, pen_in_slow=3.0, pen_in_gap=3.0,
    pen_out=20.0, pen_out_kick=1.0, pen_out_kick_other=6.0, pen_out_reappear=3.0,
    exit_ctx=45 * FPS, throw_dx=20.0, pen_zone_mismatch=3.0,
    # clean-up
    min_dead=2 * FPS, min_blip=12,
    # smart-tagging restart window around a set-piece tag start
    tag_lo=-2 * FPS, tag_hi=15 * FPS,
    # blackout change point
    gap_cp_min=2 * FPS, gap_cp_look=3 * FPS, gap_cp_gain=5.0,
)


# ----------------------------------------------------------------------------- decoding
def viterbi_tv(sH, sA, sD, swH, swA, inD, outH, outA):
    """Viterbi over H/A/D with per-frame transition costs (positive = cost)."""
    m = len(sH)
    bp = np.zeros((m, 3), dtype=np.int8)
    h, a, d = float(sH[0]), float(sA[0]), float(sD[0])
    sH = sH.tolist(); sA = sA.tolist(); sD = sD.tolist()
    swH = swH.tolist(); swA = swA.tolist(); inD = inD.tolist()
    outH = outH.tolist(); outA = outA.tolist()
    for t in range(1, m):
        # to H
        c0, c1, c2 = h, a - swH[t], d - outH[t]
        if c0 >= c1 and c0 >= c2:
            nh, bh = c0, 0
        elif c1 >= c2:
            nh, bh = c1, 1
        else:
            nh, bh = c2, 2
        # to A
        c0, c1, c2 = h - swA[t], a, d - outA[t]
        if c1 >= c0 and c1 >= c2:
            na, ba = c1, 1
        elif c0 >= c2:
            na, ba = c0, 0
        else:
            na, ba = c2, 2
        # to D
        c0, c1, c2 = h - inD[t], a - inD[t], d
        if c2 >= c0 and c2 >= c1:
            nd, bd = c2, 2
        elif c0 >= c1:
            nd, bd = c0, 0
        else:
            nd, bd = c1, 1
        h, a, d = nh + sH[t], na + sA[t], nd + sD[t]
        bp[t, 0], bp[t, 1], bp[t, 2] = bh, ba, bd
    path = np.empty(m, dtype=np.int8)
    path[-1] = int(np.argmax([h, a, d]))
    for t in range(m - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


# ----------------------------------------------------------------------------- main labeller
def set_piece_tags(tags):
    out = []
    for t in tags or []:
        if t["code"].lower() != "set pieces":
            continue
        typ = t["tags"].get("Type", "").lower()
        who = t["tags"].get("Team", "N/A")
        team = 0 if who == "Barcelona" else (1 if who in ("Atletic Club", "Athletic Club") else -1)
        out.append(dict(start=t["start_frame"], end=t["end_frame"], type=typ, team=team,
                        half=t["tags"].get("Half")))
    return out


def label_match(data, halves, home_pos_first=True, tags=None, bcast_off=None, P=None, verbose=False):
    """data: players (n,S,2) m, ball (n,2) m, team (S,), gk (S,), pitch (L, W) optional.
    halves: [(half, first, last)] inclusive. Returns (labels array 'H'/'A'/'D'/'X', aux dict)."""
    P = dict(PARAMS, **(P or {}))
    L, W = data.get("pitch", (105.3, 68.0))
    team, gk = np.asarray(data["team"]), np.asarray(data["gk"])
    ball = np.asarray(data["ball"], dtype=np.float64)
    n = len(ball)
    players, real = tracks.clean(np.asarray(data["players"], dtype=np.float64), team)
    players_real = np.where(real[:, :, None], players, np.nan)
    present = ~np.isnan(ball[:, 0])

    bvel = ev.velocity(ball, 2)
    pvel = ev.velocity(players, 2)
    bspeed = np.linalg.norm(bvel, axis=1)
    touch = ev.detect_touches(ball, bvel, players, pvel, team, P)
    ctrl, dn, tj = ev.control_frames(ball, bvel, players, pvel, team, P)
    sw_f, sw_t, run_team, ev_f, ev_t = ev.possession_events(touch, ctrl, present, P)
    ps = ev.mean_player_speed(players_real)
    cross_f, cross_goal = ev.line_crossings(ball, bvel, present, L, W, P)
    lost_f = ev.lost_near_line(ball, bvel, present, L, W, P)
    slow_f = ev.slowdown_onsets(ps, P)
    gap_f = ev.gap_starts(present, P)
    restarts = ev.detect_restarts(ball, bvel, present, players, team, L, W, P)
    sp_tags = set_piece_tags(tags)

    # restart team override from set-piece tags
    for r in restarts:
        for t in sp_tags:
            if t["team"] >= 0 and t["start"] + P["tag_lo"] <= r["frame"] <= t["start"] + P["tag_hi"]:
                r["team"] = t["team"]

    # ---- per-frame scores
    rel = np.zeros(n)
    rel += np.where(present & (run_team == 0), P["w_runteam"], 0.0)
    rel -= np.where(present & (run_team == 1), P["w_runteam"], 0.0)
    rel += np.where(ctrl == 0, P["w_ctrl"], 0.0)
    rel -= np.where(ctrl == 1, P["w_ctrl"], 0.0)
    sH = np.minimum(rel, 0.0)
    sA = -np.maximum(rel, 0.0)

    x, y = ball[:, 0], ball[:, 1]
    oob = present & ((x < -0.5) | (x > L + 0.5) | (y < -0.5) | (y > W + 0.5))
    inb = present & ~oob
    sD = np.zeros(n)
    sD += np.where(oob, P["w_oob"], 0.0)
    sD -= np.where(inb & (np.nan_to_num(bspeed) > 8.0), P["w_fast"], 0.0)
    pspeed_ctrl = np.linalg.norm(pvel[np.arange(n), np.argmin(np.where(np.isnan(
        np.linalg.norm(players - ball[:, None, :], axis=2)), np.inf,
        np.linalg.norm(players - ball[:, None, :], axis=2)), axis=1)], axis=1)
    sD -= np.where((ctrl >= 0) & (np.nan_to_num(pspeed_ctrl) > 1.5), P["w_dribble"], 0.0)
    ps_s = ev.moving_mean(ps, FPS)
    still = np.clip((P["still_mid"] - np.nan_to_num(ps_s, nan=P["still_mid"])) / P["still_half"], -1, 1)
    sD += P["w_still"] * still
    if bcast_off is not None:
        sD += np.where(bcast_off == 1.0, P["w_boff"], 0.0)
        sD -= np.where(bcast_off == 0.0, P["w_bon"], 0.0)
    for t in sp_tags:
        if "kick off" in t["type"]:
            continue
        a, b = max(0, t["start"] - 3 * FPS), min(n, t["start"] + 3 * FPS)
        sD[a:b] += P["w_tag"]
    for s, e in ev.runs_of(~present):
        if e - s > 10 * FPS:
            sD[s + 2 * FPS:e] += P["w_longgap"]

    # ---- per-frame transition costs
    swH = np.full(n, P["pen_switch"]); swA = np.full(n, P["pen_switch"])
    for f, t in zip(sw_f, sw_t):
        if t == 0:
            swH[f] = min(swH[f], P["pen_switch_event"])
        else:
            swA[f] = min(swA[f], P["pen_switch_event"])
    inD = np.full(n, P["pen_in"])
    inD[cross_f] = np.minimum(inD[cross_f], P["pen_in_cross"])
    inD[lost_f] = np.minimum(inD[lost_f], P["pen_in_lost"])
    inD[slow_f] = np.minimum(inD[slow_f], P["pen_in_slow"])
    inD[gap_f] = np.minimum(inD[gap_f], P["pen_in_gap"])
    outH = np.full(n, P["pen_out"]); outA = np.full(n, P["pen_out"])
    # restart must fit how the ball went dead: touchline exit -> touchline near the exit point;
    # goal-line exit -> corner or goal area (or centre after a goal); no exit seen -> anywhere
    exits = sorted([(int(f), bool(g), ball[f].copy()) for f, g in zip(cross_f, cross_goal)] +
                   [(int(f), None, ball[max(0, f - 1)].copy()) for f in lost_f])
    exit_frames = np.array([e[0] for e in exits], dtype=np.int64)
    for r in restarts:
        f = r["frame"]
        if r["kind"] == "kick":
            own, oth = P["pen_out_kick"], P["pen_out_kick_other"]
        else:
            own, oth = P["pen_out_reappear"], P["pen_out_reappear"] + 3.0
        k = np.searchsorted(exit_frames, f) - 1
        if k >= 0 and f - exit_frames[k] <= P["exit_ctx"]:
            _, goal_line, xy = exits[k]
            z = r["zone"]
            if goal_line is None:
                fits = True
            elif goal_line:
                fits = z in ("corner", "goal_area", "centre")
            else:
                fits = z == "touchline" and abs(ball[f, 0] - xy[0]) < P["throw_dx"]
            if not fits:
                own += P["pen_zone_mismatch"]; oth += P["pen_zone_mismatch"]
        if r["team"] == 0:
            outH[f] = min(outH[f], own); outA[f] = min(outA[f], oth)
        else:
            outA[f] = min(outA[f], own); outH[f] = min(outH[f], oth)

    # ---- decode each half
    path = np.full(n, -1, dtype=np.int8)
    kickoffs = {t["half"]: t["start"] for t in sp_tags if t["type"] == "kick off (start)"}
    for half, first, last in halves:
        sl = slice(first, last + 1)
        h_s, a_s, d_s = sH[sl].copy(), sA[sl].copy(), sD[sl].copy()
        ko = kickoffs.get("1st Half" if half == 1 else "2nd Half")
        if ko is not None and ko > first:
            k = min(ko - first, last - first)
            h_s[:k] = -1e3; a_s[:k] = -1e3
        path[sl] = viterbi_tv(h_s, a_s, d_s, swH[sl], swA[sl], inD[sl], outH[sl], outA[sl])

    # ---- clean-up
    cross_set = np.zeros(n, dtype=bool); cross_set[cross_f] = True
    claim = {0: np.zeros(n, dtype=bool), 1: np.zeros(n, dtype=bool)}
    for f, t in zip(ev_f, ev_t):
        claim[int(t)][f] = True
    for half, first, last in halves:
        seg = path[first:last + 1]
        for s, e in ev.runs_of(seg == D):
            if e - s < P["min_dead"] and not cross_set[first + s:first + e].any() and s > 0:
                seg[s:e] = seg[s - 1]
        changed = True
        while changed:
            changed = False
            for lab in (H, A):
                for s, e in ev.runs_of(seg == lab):
                    if e - s < P["min_blip"] and s > 0 and not claim[lab][first + s:first + e].any():
                        seg[s:e] = seg[s - 1]
                        changed = True
        path[first:last + 1] = seg

    # ---- live blackouts: switch where both teams' shapes reverse
    sign_home = np.zeros(n)
    for half, first, last in halves:
        sign_home[first:last + 1] = 1.0 if (half == 1) == home_pos_first else -1.0
    vh, va = ev.team_forward_velocity(players_real, team, gk, sign_home)
    n_cp = 0
    for half, first, last in halves:
        for s, e in ev.runs_of(~present[first:last + 1]):
            s, e = s + first, e + first
            if e - s < P["gap_cp_min"] or s == first or e > last:
                continue
            before = path[s - 1]
            if before == D or np.any(path[s:e] != before):
                continue
            # switches can only be decoded at touches, so a turnover inside the gap shows up as a
            # switch at the first claim after it: look a little past the gap end
            look = path[e:min(last + 1, e + P["gap_cp_look"])]
            chg = np.flatnonzero(look != before)
            if len(chg) == 0 or look[chg[0]] == D:
                continue
            after, sw_at = look[chg[0]], e + int(chg[0])
            v1 = vh[s:e] if before == H else va[s:e]
            v2 = va[s:e] if before == H else vh[s:e]
            sc = np.nan_to_num(v1 - v2)                 # > 0 while the old team attacks
            cum = np.concatenate(([0.0], np.cumsum(sc)))
            gain = cum - (cum[-1] - cum)
            k = int(np.argmax(gain))
            if k >= e - s or gain[k] <= gain[-1] + P["gap_cp_gain"]:
                continue                                # no clear reversal inside the gap
            path[s + k:sw_at] = after
            n_cp += 1

    labels = np.full(n, "X", dtype="<U1")
    m = path >= 0
    labels[m] = LABELS[path[m]]
    aux = dict(n_touch=len(touch["frame"]), n_switch_events=len(sw_f), n_cross=len(cross_f),
               n_lost=len(lost_f), n_gap=len(gap_f), n_slow=len(slow_f), n_restart=len(restarts), n_cp=n_cp)
    if verbose:
        print("events:", aux)
    return labels, aux


# ----------------------------------------------------------------------------- BAR-ATH entry point
def run_barath(P=None, use_tags=True, use_broadcast=True, verbose=False):
    import atd
    import load_smart_tagging
    from games import GAME, half_video_blocks
    data = atd.load()
    data = dict(data, pitch=(atd.PITCH_L, atd.PITCH_W))
    tags = load_smart_tagging.load() if use_tags else None
    boff = None
    if use_broadcast:
        import broadcast
        boff, _ = broadcast.broadcast_off(data["n_frames"])
    halves = half_video_blocks(GAME)
    return label_match(data, halves, home_pos_first=True, tags=tags, bcast_off=boff, P=P,
                       verbose=verbose)


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "solution_final.csv"))
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--no-broadcast", action="store_true")
    args = ap.parse_args()
    labels, aux = run_barath(use_tags=not args.no_tags, use_broadcast=not args.no_broadcast,
                             verbose=True)
    write_csv(args.out, labels)
    from games import GAME, half_video_blocks
    for half, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        runs = [seg[i] for i in range(len(seg)) if i == 0 or seg[i] != seg[i - 1]]
        sp = sum(1 for r in runs if r in "HA")
        print("half {}: spells={} D-runs={} H={} A={} D={}".format(
            half, sp, sum(1 for r in runs if r == "D"), (seg == "H").sum(), (seg == "A").sum(),
            (seg == "D").sum()))
    print("wrote", args.out)
