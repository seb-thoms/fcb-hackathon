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
    return out, dn, tj, j


def possession_events(touch, ctrl, present, P, forced=None, ctrl_slot=None, near=None):
    """Event-level possession.

    Claims = touches + control-episode onsets (+ `forced` restart claims, always controlled).
    Consecutive same-team claims form a run; a run is *controlled* when it shows control:
      - >= min_claims claims (contested ones counted only if count_contested) spanning >= claim_span
        frames, or
      - >= ctrl_min frames of control in its first 2 s, or
      - (pass_confirm) a kick by one player followed by a claim by a team-mate within pass_window, or
      - (clean_single) a single uncontested reception followed by >= single_ctrl control frames.
    A controlled run of the other team switches possession. The switch is placed at the run's
    first uncontested claim; with `backdate` it moves back to the new team's earliest claim since
    the old team's last controlled run (skipping uncontrolled touches in between, and with
    `backdate_skip_duels` also runs made only of contested claims; at most `backdate` frames), and
    with `backdate_contact` further back over the frames where a player of
    the new team was already at the ball (`near` = (nearest team, distance) per frame).
    Returns switch frames/teams, per-frame run_team, and all non-forced claim frames/teams.
    With the default parameters this reproduces the original rule exactly.
    """
    n = len(ctrl)
    F, T, C, X, K, S = [], [], [], [], [], []
    for f, t, c, k, sl in zip(touch["frame"], touch["team"], touch["contested"], touch["kick"], touch["slot"]):
        F.append(int(f)); T.append(int(t)); C.append(bool(c)); X.append(False); K.append(bool(k)); S.append(int(sl))
    # control-episode onsets act as receptions (slow balls do not make velocity peaks)
    for team_id in (0, 1):
        for s, e in runs_of(ctrl == team_id):
            if e - s >= P["ctrl_episode"]:
                F.append(int(s)); T.append(team_id); C.append(False); X.append(False); K.append(False)
                S.append(int(ctrl_slot[s]) if ctrl_slot is not None else -1)
    for f, t in (forced or []):
        F.append(int(f)); T.append(int(t)); C.append(False); X.append(True); K.append(False); S.append(-1)
    order = np.lexsort((~np.array(X, dtype=bool), np.array(F, dtype=np.int64)))
    F = np.array(F, dtype=np.int64)[order]; T = np.array(T, dtype=np.int8)[order]
    C = np.array(C, dtype=bool)[order]; X = np.array(X, dtype=bool)[order]
    K = np.array(K, dtype=bool)[order]; S = np.array(S, dtype=np.int64)[order]

    # optional: one action often yields two claims (a velocity peak and a control onset) -> merge
    merge = P.get("claim_merge", 0)
    if merge > 0 and len(F):
        keep = np.ones(len(F), dtype=bool)
        last = 0
        for k in range(1, len(F)):
            if (not X[k] and not X[last] and T[k] == T[last] and F[k] - F[last] <= merge
                    and (S[k] == S[last] or S[k] < 0 or S[last] < 0)):
                keep[k] = False
                C[last] = C[last] and C[k]
                K[last] = K[last] or K[k]
                if S[last] < 0:
                    S[last] = S[k]
            else:
                last = k
        F, T, C, X, K, S = F[keep], T[keep], C[keep], X[keep], K[keep], S[keep]

    cum_ctrl = {tid: np.concatenate(([0], np.cumsum(ctrl == tid))) for tid in (0, 1)}
    runs = []
    i = 0
    while i < len(F):
        j = i
        while (j + 1 < len(F) and not X[j + 1] and T[j + 1] == T[i]
               and F[j + 1] - F[j] <= P["run_gap"]):
            j += 1
        runs.append((i, j))
        i = j + 1

    count_contested = P.get("count_contested", True)
    min_claims = P.get("min_claims", 2)
    claim_span = P.get("claim_span", 0)
    pass_confirm = P.get("pass_confirm", False)
    pass_window = P.get("pass_window", 4 * FPS)
    clean_single = P.get("clean_single", False)

    def is_controlled(i, j):
        if X[i]:
            return True
        tid = int(T[i])
        f0 = int(F[i])
        nctrl = cum_ctrl[tid][min(n, f0 + 2 * FPS)] - cum_ctrl[tid][f0]
        if nctrl >= P["ctrl_min"]:
            return True
        ks = [k for k in range(i, j + 1) if count_contested or not C[k]]
        if len(ks) >= min_claims and F[ks[-1]] - F[ks[0]] >= claim_span:
            return True
        if pass_confirm:
            for a in range(i, j + 1):
                if not K[a] or S[a] < 0:
                    continue
                for b in range(a + 1, j + 1):
                    if F[b] - F[a] > pass_window:
                        break
                    if S[b] >= 0 and S[b] != S[a]:
                        return True
        if clean_single and j == i and not C[i] and not K[i] and nctrl >= P.get("single_ctrl", 8):
            return True
        return False

    controlled = [is_controlled(i, j) for i, j in runs]
    backdate = P.get("backdate", 0)
    backdate_contact = P.get("backdate_contact", 0)
    switch_f, switch_t = [], []
    run_team = np.full(n, -1, dtype=np.int8)
    cur = -1
    marks = []
    for r, (i, j) in enumerate(runs):
        if not controlled[r]:
            continue
        tid = int(T[i])
        if X[i]:
            marks.append((int(F[i]), tid))
            cur = tid
            continue
        if tid == cur:
            continue
        unc = [k for k in range(i, j + 1) if not C[k]]
        kb = unc[0] if unc else i
        if backdate > 0:
            kb = i
            limit = F[i] - backdate
            rr = r - 1
            skip_duels = P.get("backdate_skip_duels", False)
            while rr >= 0:
                a, b = runs[rr]
                duel = skip_duels and bool(np.all(C[a:b + 1]))
                if X[a] or (controlled[rr] and T[a] != tid and not duel) or F[b] < limit:
                    break
                if T[a] == tid:
                    ks = [k for k in range(a, b + 1) if F[k] >= limit]
                    if ks:
                        kb = ks[0]
                rr -= 1
        f_sw = int(F[kb])
        if backdate_contact > 0 and near is not None:
            near_team, near_d = near
            prev_other = [k for k in range(kb - 1, -1, -1) if T[k] != tid]
            floor = int(F[prev_other[0]]) + 1 if prev_other else 0
            t = f_sw
            while (t - 1 > floor and f_sw - (t - 1) <= backdate_contact and near_team[t - 1] == tid
                   and near_d[t - 1] < P.get("contact_r", 2.0)):
                t -= 1
            f_sw = t
        if marks and f_sw <= marks[-1][0]:
            f_sw = marks[-1][0] + 1
        switch_f.append(f_sw); switch_t.append(tid)
        marks.append((f_sw, tid))
        cur = tid
    for idx, (f, tid) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else n
        run_team[f:end] = tid
    keep = ~X
    return (np.array(switch_f, dtype=np.int64), np.array(switch_t, dtype=np.int8), run_team,
            F[keep], T[keep])


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
