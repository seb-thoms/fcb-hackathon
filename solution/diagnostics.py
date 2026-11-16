"""Label-free plausibility checks for a candidate submission (no ground truth involved).

    python solution/diagnostics.py evaluation/results/solution_v1.csv
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

FPS = 25


def runs(seg):
    out, s = [], 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[s]:
            out.append((seg[s], s, i - 1))
            s = i
    return out


def main(path):
    labels = np.array([l.strip() for l in open(path).read().split("\n")[1:] if l])
    labels = np.array([l.split(",")[1] for l in open(path).read().strip().split("\n")[1:]])
    tags = load_smart_tagging.load()
    sp = [t for t in tags if t["code"].lower() == "set pieces"]
    for half, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        r = runs(seg)
        spells = [(l, a, b) for l, a, b in r if l in "HA"]
        dead = [(l, a, b) for l, a, b in r if l == "D"]
        dur = np.array([(b - a + 1) / FPS for _, a, b in spells])
        ddur = np.array([(b - a + 1) / FPS for _, a, b in dead])
        print("half {}: {} spells (median {:.1f}s, <2s: {}, >60s: {}), {} D runs (median {:.1f}s, <2s: {}), D share {:.2f}, home share of live {:.2f}".format(
            half, len(spells), np.median(dur), (dur < 2).sum(), (dur > 60).sum(), len(dead),
            np.median(ddur) if len(ddur) else 0, (ddur < 2).sum(), (seg == "D").mean(),
            (seg == "H").sum() / max(1, (seg != "D").sum())))
        # set pieces (not kick-offs): is there a predicted D shortly before the restart?
        label = "1st Half" if half == 1 else "2nd Half"
        hit = tot = 0
        for t in sp:
            if t["tags"].get("Half") != label or "kick off" in t["tags"].get("Type", "").lower():
                continue
            tot += 1
            a = t["start_frame"] - first
            lo, hi = max(0, a - 10 * FPS), min(len(seg), a + 20 * FPS)
            hit += int((seg[lo:hi] == "D").any())
        print("   set pieces with a predicted D in [-10s,+20s] of the tag start: {}/{}".format(hit, tot))


if __name__ == "__main__":
    main(sys.argv[1])
