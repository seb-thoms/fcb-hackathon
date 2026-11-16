"""Parse the tactical ATD into dense numpy arrays (metres), cached under data/cache/."""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
CACHE = os.path.join(ROOT, "data", "cache", "atd_arrays.npz")  # data/ is gitignored


def slot_meta(meta_path):
    """(team per slot: 0=H 1=A, is_goalkeeper per slot)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    players = list(root.iter("Player"))
    team = np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in players], dtype=np.int8)
    gk = np.array([(p.findtext("ParameterValue") or "") == "Goalkeeper" or
                   "oalkeeper" in ET.tostring(p, encoding="unicode") for p in players])
    return team, gk


def _parse(raw_path, n_frames, n_slots):
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    def xy(group):
        if group[0] == "N":
            return None
        a = group.split(",")
        return float(a[0]), float(a[1])

    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            for s, g in enumerate(parts[1].split(";")):
                v = xy(g)
                if v is not None:
                    players[f, s] = v
            v = xy(parts[2])
            if v is not None:
                ball[f] = v
    players[..., 0] *= PITCH_L
    players[..., 1] *= PITCH_W
    ball[:, 0] *= PITCH_L
    ball[:, 1] *= PITCH_W
    return players, ball


def load(use_cache=True):
    """Returns dict: players (N,S,2) m, ball (N,2) m, team (S,), gk (S,), n_frames."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    team, gk = slot_meta(meta_path)
    if use_cache and os.path.exists(CACHE):
        z = np.load(CACHE)
        return dict(players=z["players"], ball=z["ball"], team=team, gk=gk,
                    n_frames=z["ball"].shape[0])
    n = source_frames("tactical", GAME)
    players, ball = _parse(raw_path, n, len(team))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, players=players, ball=ball)
    return dict(players=players, ball=ball, team=team, gk=gk, n_frames=n)


if __name__ == "__main__":
    d = load(use_cache=False)
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in d.items()})
    print("slots per team:", np.bincount(d["team"]), "gk slots:", np.where(d["gk"])[0])
