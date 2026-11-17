"""Player-track cleaning for the automatic tracking.

1. Carry a lost player forward: when a slot stops being observed, keep its last position for a
   short time, so a ball receiver whose track just broke does not vanish from the nearest-player
   check.
2. Drop ghosts and duplicates: a carried (stale) position within GHOST_R of a real same-team
   observation is the same player re-acquired in a new slot -> drop the ghost; two real same-team
   observations within DUP_R are one player tracked twice -> keep one.
"""
import numpy as np

CARRY_FRAMES = 37      # 1.5 s
GHOST_R = 2.5          # m
DUP_R = 1.0            # m


def carry_forward(players, max_frames=CARRY_FRAMES):
    """Returns (players_filled, is_real) where is_real marks genuine observations."""
    n, S, _ = players.shape
    real = ~np.isnan(players[:, :, 0])
    idx = np.where(real, np.arange(n)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)
    age = np.arange(n)[:, None] - last
    fill = (~real) & (last >= 0) & (age <= max_frames)
    out = players.copy()
    t, s = np.nonzero(fill)
    out[t, s] = players[last[t, s], s]
    return out, real


def drop_ghosts_and_duplicates(players, real, team, chunk=8000):
    out = players.copy()
    n, S, _ = players.shape
    for team_id in (0, 1):
        cols = np.flatnonzero(team == team_id)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            p = players[a:b][:, cols]                       # (m, k, 2)
            r = real[a:b][:, cols]
            d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=3)  # (m, k, k)
            d = np.where(np.isnan(d), np.inf, d)
            k = len(cols)
            eye = np.eye(k, dtype=bool)[None]
            d = np.where(eye, np.inf, d)
            # ghost i near any real j
            ghost = (~r) & ~np.isnan(p[:, :, 0])
            near_real = np.any((d < GHOST_R) & r[:, None, :], axis=2)
            kill = ghost & near_real
            # real duplicates: drop the higher slot index of a close real pair
            upper = np.triu(np.ones((k, k), dtype=bool), 1)[None]
            dup = np.any((d < DUP_R) & r[:, :, None] & r[:, None, :] & upper, axis=1)  # j has a lower close twin
            kill |= dup & r
            sub = out[a:b]
            m_idx, c_idx = np.nonzero(kill)
            sub[m_idx, cols[c_idx]] = np.nan
            out[a:b] = sub
    return out


def clean(players, team):
    filled, real = carry_forward(players)
    cleaned = drop_ghosts_and_duplicates(filled, real, team)
    real_after = real & ~np.isnan(cleaned[:, :, 0])
    return cleaned, real_after
