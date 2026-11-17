"""Event detection on tracking data: touches, control, line crossings, collective slow-downs,
restarts, and team-shape velocities. Everything here is a general rule; nothing is tied to frames
of a specific match. Inputs are in metres; frames at 25 fps.
"""
import numpy as np

FPS = 25


# ----------------------------------------------------------------------------- helpers
def moving_mean(x, win):
    """Centred moving mean over +/- win frames, ignoring NaN (NaN where no data)."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - win), np.minimum(len(x), i + win + 1)
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def window_mean(x, a, b):
    """Mean of x over [t+a, t+b) for every t (a<b may be negative), NaN-aware."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.clip(i + a, 0, len(x)), np.clip(i + b, 0, len(x))
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def rolling_max(x, w):
    """Centred rolling max over +/- w frames (NaN treated as -inf)."""
    y = np.where(np.isnan(x), -np.inf, x)
    out = y.copy()
    for k in range(1, w + 1):
        out[:-k] = np.maximum(out[:-k], y[k:])
        out[k:] = np.maximum(out[k:], y[:-k])
    return out


def runs_of(mask):
    """[(start, stop)) runs where mask is True."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


# ----------------------------------------------------------------------------- kinematics
def velocity(pos, k=2):
    """Central-difference velocity (m/s) over +/- k frames; works for (n,2) and (n,S,2)."""
    v = np.full(pos.shape, np.nan, dtype=np.float64)
    v[k:-k] = (pos[2 * k:] - pos[:-2 * k]) * FPS / (2.0 * k)
    return v


def proximity(players, team, ball):
    """Distances from the ball to every slot and per-team nearest. inf where unknown."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    dn = d[np.arange(len(d)), j]
    dh = np.min(np.where(team[None, :] == 0, d, np.inf), axis=1)
    da = np.min(np.where(team[None, :] == 1, d, np.inf), axis=1)
    return d, j, dn, dh, da


def mean_player_speed(players_real, k=2):
    """Mean speed (m/s) of genuinely observed players per frame."""
    v = velocity(players_real, k)
    s = np.linalg.norm(v, axis=2)
    s = np.where(s < 12, s, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(s, axis=1)


# ----------------------------------------------------------------------------- touches & control
def detect_touches(ball, bvel, players, pvel, team, P):
    """Ball-velocity change peaks with a player close by.
    Returns dict of arrays: frame, team, slot, contested (bool), kick (bool)."""
    n = len(ball)
    k = P["touch_k"]
    dv = np.full(n, np.nan)
    dv[k:-k] = np.linalg.norm(bvel[2 * k:] - bvel[:-2 * k], axis=1)
    peak = (dv > P["touch_dv"]) & (dv >= rolling_max(dv, k))
    frames = np.flatnonzero(peak)
    # remove peaks within k frames of an earlier kept one (plateaus)
    keep, last = [], -10 ** 9
    for f in frames:
        if f - last > k:
            keep.append(f)
            last = f
    frames = np.array(keep, dtype=np.int64)
    if len(frames) == 0:
        return dict(frame=frames, team=frames, slot=frames, contested=frames.astype(bool),
                    kick=frames.astype(bool))
    d = np.linalg.norm(players[frames] - ball[frames][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    dn = d[np.arange(len(frames)), j]
    tj = team[j]
    other = np.min(np.where(team[None, :] != tj[:, None], d, np.inf), axis=1)
    ok = dn < P["touch_r"]
    sp_before = np.linalg.norm(bvel[np.maximum(frames - k, 0)], axis=1)
    sp_after = np.linalg.norm(bvel[np.minimum(frames + k, n - 1)], axis=1)
    out = dict(frame=frames[ok], team=tj[ok].astype(np.int8), slot=j[ok],
               contested=(other[ok] < P["touch_r"]) & (other[ok] < dn[ok] + 1.0),
               kick=(sp_after[ok] > sp_before[ok]))
    return out


def control_frames(ball, bvel, players, pvel, team, P):
    """Per frame: team (0/1) of a player controlling the ball (close and moving with it), else -1.
    Contested frames (an opponent equally close) are -1."""
    n = len(ball)
    d, j, dn, dh, da = proximity(players, team, ball)
    pv = pvel[np.arange(n), j]
    rel = np.linalg.norm(bvel - pv, axis=1)
    tj = team[j]
    dother = np.where(tj == 0, da, dh)
    ok = (dn < P["ctrl_r"]) & (rel < P["ctrl_relv"]) & (dother > dn + P["ctrl_margin"])
    ok &= ~np.isnan(ball[:, 0])
    out = np.where(ok, tj, -1).astype(np.int8)
    return out, dn, tj


def possession_events(touch, ctrl, present, P, forced=None):
    """Event-level possession: merge touches and control-episode onsets into a time-ordered list
    of team claims, group consecutive same-team claims into runs, and keep runs that show control
    (>= 2 claims, or >= ctrl_min frames of control in the first 2 s). `forced` = [(frame, team)]
    claims that are always controlled and always start a new run (restarts: after every restart
    there is a clear possessor). Returns
      switch_frame, switch_team: first (uncontested if possible) claim of each controlled run whose
                                 team differs from the current possessor
      run_team: per-frame team of the latest controlled run (-1 before the first).
    """
    n = len(ctrl)
    ev_f, ev_t, ev_c = list(touch["frame"]), list(touch["team"]), list(touch["contested"])
    ev_x = [False] * len(ev_f)
    # control-episode onsets act as receptions (slow balls do not make velocity peaks)
    for team_id in (0, 1):
        for s, e in runs_of(ctrl == team_id):
            if e - s >= P["ctrl_episode"]:
                ev_f.append(s); ev_t.append(team_id); ev_c.append(False); ev_x.append(False)
    for f, t in (forced or []):
        ev_f.append(int(f)); ev_t.append(int(t)); ev_c.append(False); ev_x.append(True)
    order = np.lexsort((~np.array(ev_x, dtype=bool), np.array(ev_f, dtype=np.int64)))
    ev_f = np.array(ev_f, dtype=np.int64)[order]
    ev_t = np.array(ev_t, dtype=np.int8)[order]
    ev_c = np.array(ev_c, dtype=bool)[order]
    ev_x = np.array(ev_x, dtype=bool)[order]

    cum_ctrl = {tid: np.concatenate(([0], np.cumsum(ctrl == tid))) for tid in (0, 1)}
    runs = []
    i = 0
    while i < len(ev_f):
        j = i
        while (j + 1 < len(ev_f) and not ev_x[j + 1] and ev_t[j + 1] == ev_t[i]
               and ev_f[j + 1] - ev_f[j] <= P["run_gap"]):
            j += 1
        runs.append((i, j))
        i = j + 1
    switch_f, switch_t = [], []
    run_team = np.full(n, -1, dtype=np.int8)
    cur = -1
    marks = []
    for i, j in runs:
        tid = int(ev_t[i])
        f0 = int(ev_f[i])
        f1 = min(n, f0 + 2 * FPS)
        nctrl = cum_ctrl[tid][f1] - cum_ctrl[tid][f0]
        controlled = bool(ev_x[i]) or (j - i + 1 >= 2) or (nctrl >= P["ctrl_min"])
        if not controlled:
            continue
        if ev_x[i]:
            marks.append((f0, tid))
            cur = tid
            continue
        if tid != cur:
            # first uncontested claim of the run, else the first claim
            unc = [k for k in range(i, j + 1) if not ev_c[k]]
            f = int(ev_f[unc[0]]) if unc else f0
            switch_f.append(f); switch_t.append(tid)
            marks.append((f, tid))
            cur = tid
    for idx, (f, tid) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else n
        run_team[f:end] = tid
    keep = ~ev_x
    return (np.array(switch_f, dtype=np.int64), np.array(switch_t, dtype=np.int8), run_team,
            ev_f[keep], ev_t[keep])


# ----------------------------------------------------------------------------- dead-ball entry
def line_crossings(ball, bvel, present, L, W, P):
    """Frames where the visible ball crosses a pitch line and stays out / disappears.
    Returns frames (int array) and kind ('touch' or 'goal' line) as bool is_goal_line."""
    n = len(ball)
    x, y = ball[:, 0], ball[:, 1]
    m = P["out_margin"]
    out = present & ((x < -m) | (x > L + m) | (y < -m) | (y > W + m))
    inside = present & (x >= 0) & (x <= L) & (y >= 0) & (y <= W)
    frames, goal = [], []
    for s, e in runs_of(out):
        # must be preceded by the ball inside the pitch within 0.5 s
        prev_in = np.flatnonzero(inside[max(0, s - 13):s])
        if len(prev_in) == 0:
            continue
        # stays out or disappears for >= 0.4 s
        tail = slice(e, min(n, e + 10))
        if e - s < 3 and np.any(inside[tail]):
            continue
        # walk back to the first frame beyond the line itself
        f = s
        while f - 1 >= 0 and present[f - 1] and not inside[f - 1]:
            f -= 1
        frames.append(f)
        goal.append(bool((x[s] < -m) or (x[s] > L + m)))
    return np.array(frames, dtype=np.int64), np.array(goal, dtype=bool)


def lost_near_line(ball, bvel, present, L, W, P):
    """Ball last seen close to a line moving towards it, then missing >= gap_min: estimated exit."""
    n = len(ball)
    frames = []
    for s, e in runs_of(~present):
        if e - s < P["lost_gap_min"] or s < 3:
            continue
        f = s - 1
        if not present[f]:
            continue
        x, y = ball[f]
        vx, vy = bvel[f - 2] if not np.isnan(bvel[f - 2, 0]) else (np.nan, np.nan)
        if np.isnan(vx):
            continue
        cands = [(x, -vx), (L - x, vx), (y, -vy), (W - y, vy)]   # (distance to line, speed towards it)
        best = None
        for dist, vt in cands:
            if dist < P["lost_dist"] and vt > P["lost_speed"]:
                t_cross = f + int(round(min(FPS, max(0.0, dist) / vt * FPS)))
                best = t_cross if best is None else min(best, t_cross)
        if best is not None:
            frames.append(min(best, e - 1))
    return np.array(frames, dtype=np.int64)


def gap_starts(present, P):
    """Ball lost (anywhere) and missing for >= lost_any_gap: the last seen frame + 1."""
    return np.array([s for s, e in runs_of(~present) if e - s >= P["lost_any_gap"] and s > 0],
                    dtype=np.int64)


def slowdown_onsets(ps, P):
    """Collective deceleration onsets (whistle while the ball is still on the pitch).
    ps: mean player speed per frame. Returns onset frames."""
    s = moving_mean(ps, 5)
    before = window_mean(s, -2 * FPS, 0)
    after = window_mean(s, 1 * FPS, 5 * FPS)
    drop = before - after
    cand = (before > P["slow_before"]) & (after < P["slow_after"]) & (drop > P["slow_drop"])
    cand &= drop >= rolling_max(np.where(cand, drop, np.nan), 2 * FPS)
    frames = []
    ds = np.gradient(s)
    for f in np.flatnonzero(cand):
        lo, hi = max(1, f - 2 * FPS), min(len(s) - 1, f + 2 * FPS)
        steep = lo + int(np.nanargmin(ds[lo:hi])) if np.any(~np.isnan(ds[lo:hi])) else f
        g = steep
        while g - 1 > lo and ds[g - 1] < 0:     # back to the local maximum before the descent
            g -= 1
        frames.append(g)
    frames = sorted(set(frames))
    return np.array(frames, dtype=np.int64)


# ----------------------------------------------------------------------------- restarts
def restart_zone(x, y, L, W):
    if (x < 3 or x > L - 3) and (y < 3 or y > W - 3):
        return "corner"
    if y < 2.5 or y > W - 2.5:
        return "touchline"
    if (x < 7 or x > L - 7) and abs(y - W / 2) < 11:
        return "goal_area"
    if abs(x - L / 2) < 2.5 and abs(y - W / 2) < 2.5:
        return "centre"
    return "open"


def detect_restarts(ball, bvel, present, players, team, L, W, P):
    """Restart candidates: (a) ball still for a moment then kicked; (b) ball reappears moving after
    a gap. Returns list of dicts: frame, team, zone, kind ('kick' / 'reappear')."""
    n = len(ball)
    sp = np.linalg.norm(bvel, axis=1)
    out = []
    still = present & (sp < P["still_speed"])
    for s, e in runs_of(still):
        if e - s < P["still_frames"] or e >= n - 10:
            continue
        # kicked: speed rises above kick_speed within kick_within frames after the still run
        w = sp[e:min(n, e + P["kick_within"])]
        hit = np.flatnonzero(w > P["kick_speed"])
        if len(hit) == 0:
            continue
        f = e  # first frame after the still run: the moment the ball starts to move
        d = np.linalg.norm(players[f] - ball[f], axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        jn = int(np.argmin(d))
        if d[jn] > P["taker_r"]:
            continue
        out.append(dict(frame=int(f), team=int(team[jn]), zone=restart_zone(*ball[f], L, W), kind="kick"))
    # (c) windowed onset: slow for a moment, then fast (robust to jitter in the speed estimate)
    pre = window_mean(np.where(present, sp, np.nan), -P["onset_pre"], 0)
    post = window_mean(np.where(present, sp, np.nan), 3, 3 + P["onset_post"])
    cand = present & (pre < P["onset_pre_speed"]) & (post > P["onset_post_speed"])
    jump = np.where(cand, post - pre, np.nan)
    pk = cand & (jump >= rolling_max(jump, FPS // 2))
    have = [r["frame"] for r in out]
    for f in np.flatnonzero(pk):
        if any(abs(f - h) <= 10 for h in have):
            continue
        d = np.linalg.norm(players[f] - ball[f], axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        jn = int(np.argmin(d))
        if d[jn] > P["taker_r"]:
            continue
        out.append(dict(frame=int(f), team=int(team[jn]), zone=restart_zone(*ball[f], L, W), kind="kick"))
        have.append(f)
    for s, e in runs_of(~present):
        if e - s < P["reappear_gap"] or e >= n - 3:
            continue
        f = e
        if sp[min(n - 1, f + 2)] < P["kick_speed"] if not np.isnan(sp[min(n - 1, f + 2)]) else True:
            continue
        d = np.linalg.norm(players[f] - ball[f], axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        jn = int(np.argmin(d))
        out.append(dict(frame=int(f), team=int(team[jn]), zone=restart_zone(*ball[f], L, W), kind="reappear"))
    out.sort(key=lambda r: r["frame"])
    return out


# ----------------------------------------------------------------------------- team shape
def team_forward_velocity(players_real, team, gk, sign_home):
    """Per-frame forward (attack-direction) velocity of each team's outfield centroid, m/s,
    smoothed over +/- 1 s. sign_home: (n,) +1 where home attacks +x."""
    out = []
    for team_id, sgn in ((0, sign_home), (1, -sign_home)):
        m = (team == team_id) & ~gk
        with np.errstate(all="ignore"):
            cx = np.nanmean(players_real[:, m, 0], axis=1)
        v = np.full(len(cx), np.nan)
        v[FPS // 2: FPS // 2 + len(cx) - FPS] = (cx[FPS:] - cx[:-FPS])
        out.append(moving_mean(v * sgn, FPS))
    return out[0], out[1]
