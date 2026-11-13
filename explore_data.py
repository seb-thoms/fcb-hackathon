"""Aggregate-only summary of the tactical data: counts, rates and percentiles, never raw rows.

    python explore_data.py > explore_output.txt
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N = GAMES[GAME]["n_video_frames"]
SET_PIECE_WINDOW = 2 * FPS  # frames either side of a set-piece start
ATTACK = {"build up", "progression", "creating chances", "attacking transition", "long ball",
          "ball in final third", "ball in the box", "players in the box"}
DEFEND = {"defending in attacking third", "defending in middle third",
          "defending in defensive third", "defensive transition"}
TEAM_CODE = {"barcelona": 0, "atletic club": 1}
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def pct(values, q=(5, 25, 50, 75, 95)):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "n/a"
    return " ".join("p{}={:.2f}".format(p, v) for p, v in zip(q, np.percentile(values, q)))


def runs(mask):
    """(starts, lengths) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return starts, ends - starts


def load_atd(raw_path):
    """(players[N, slots, 3], ball[N, 3]) as x, y, speed, NaN where missing."""
    players = None
    ball = np.full((N, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, groups, ball_group = line.rstrip("\n").split(":")
            groups = groups.split(";")
            if players is None:
                players = np.full((N, len(groups), 3), np.nan, dtype=np.float32)
            row = int(frame_id) - 1
            for slot, group in enumerate(groups):
                if group[0] != "N":
                    players[row, slot] = [float(v) for v in group.split(",")]
            ball[row] = [float(v) for v in ball_group.split(",")]
    return players, ball


def section(title):
    print("\n== {} ==".format(title))


def main():
    paths = asset_paths()
    graded = np.zeros(N, dtype=bool)
    half_of = np.zeros(N, dtype=np.int8)
    for half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
        half_of[first:last + 1] = half

    section("0. metadata")
    team = team_of_each_track(paths["atd_meta"])
    print("home (H): {} | away (A): {} | slots H/A: {}/{}".format(
        *team_names(paths["atd_meta"]), int((team == 0).sum()), int((team == 1).sum())))
    root = ET.parse(paths["atd_meta"]).getroot()
    for element in root.iter("Team"):
        print("  team", element.findtext("Name"), {p.findtext("Name"): p.findtext("Value")
                                                   for p in element.iter("ProviderParameter")})
    for channel in root.iter("Channel"):
        print("  channel", channel.get("id"), "unit:", channel.findtext("Unit"))
    position_types = [p.findtext("Value") for p in root.iter("ProviderParameter")
                      if p.findtext("Name") == "position_type"]
    print("  position_type values:", dict(pd.Series(position_types).value_counts()))

    players, ball = load_atd(paths["atd_raw"])
    present = ~np.isnan(players[:, :, 0])
    # The ATD ball speed channel is mostly NaN and does not track the ball's motion, so ball speed
    # is derived from consecutive positions instead (m/s; NaN where either frame lacks the ball).
    print("  ball speed channel filled on {:.1%} of graded frames (ball present on {:.1%})".format(
        np.mean(~np.isnan(ball[graded, 2])), np.mean(~np.isnan(ball[graded, 0]))))
    ball[1:, 2] = np.hypot(np.diff(ball[:, 0]) * LENGTH, np.diff(ball[:, 1]) * WIDTH) * FPS
    ball[0, 2] = np.nan

    section("1. players (graded frames)")
    active = present[graded].sum(1)
    print("active slots per frame:", pct(active),
          "| <18: {:.1%}  >22: {:.1%}".format(np.mean(active < 18), np.mean(active > 22)))
    for code, name in ((0, "H"), (1, "A")):
        side = present[graded][:, team == code].sum(1)
        print("  {} active: {} | >11: {:.1%}".format(name, pct(side), np.mean(side > 11)))
    lengths, used, both_halves = [], 0, 0
    for slot in range(players.shape[1]):
        _, slot_runs = runs(present[:, slot] & graded)
        lengths.extend(slot_runs)
        used += slot_runs.size > 0
        both_halves += all((present[:, slot] & (half_of == h)).any() for h in (1, 2))
    lengths = np.array(lengths) / FPS
    print("slots ever active: {}/{} | active in both halves: {}".format(
        used, players.shape[1], both_halves))
    print("tracklets: {} | length s: {} | <2 s: {:.1%}  >10 min: {:.1%}".format(
        lengths.size, pct(lengths), np.mean(lengths < 2), np.mean(lengths > 600)))
    speed = players[graded][:, :, 2][present[graded]]
    print("player speed m/s:", pct(speed, (50, 95, 99)),
          "max={:.1f} | >12: {:.3%}".format(np.nanmax(speed), np.mean(speed > 12)))
    step = np.hypot(np.diff(players[:, :, 0], axis=0) * LENGTH,
                    np.diff(players[:, :, 1], axis=0) * WIDTH)
    step = step[graded[1:] & graded[:-1]]
    step = step[~np.isnan(step)]
    print("frame-to-frame jumps >1 m (teleports): {} ({:.3%})".format(
        int((step > 1).sum()), np.mean(step > 1)))
    x, y = players[graded][:, :, 0], players[graded][:, :, 1]
    print("positions outside 0-1: x {:.2%}  y {:.2%}".format(
        np.nanmean((x < 0) | (x > 1)), np.nanmean((y < 0) | (y > 1))))

    section("1b. orientation per half")
    for half in (1, 2):
        in_half = half_of == half
        for code, name in ((0, "H"), (1, "A")):
            xs = players[in_half][:, team == code, 0]
            slot_share = (~np.isnan(xs)).mean(0)
            regular = slot_share > 0.5
            if not regular.any():
                print("  half {} {}: no slot present >50%".format(half, name))
                continue
            slot_mean = np.full(xs.shape[1], np.nan)
            slot_mean[regular] = np.nanmean(xs[:, regular], axis=0)
            outlier = np.nanargmax(np.abs(slot_mean - np.nanmean(xs)))
            print("  half {} {}: team mean x {:.2f} | most extreme regular slot: mean x {:.2f}, "
                  "std x {:.3f}, present {:.0%}".format(
                      half, name, np.nanmean(xs), slot_mean[outlier],
                      np.nanstd(xs[:, outlier]), slot_share[outlier]))

    section("2. ball (graded frames)")
    missing = np.isnan(ball[:, 0])
    for half in (1, 2):
        print("  missing in half {}: {:.1%}".format(half, missing[half_of == half].mean()))
    _, gaps = runs(missing & graded)
    missing_frames = max(gaps.sum(), 1)
    print("gaps: {} | length frames: {}".format(gaps.size, pct(gaps)))
    for low, high in ((1, 5), (6, 25), (26, 125), (126, 750), (751, 10 ** 9)):
        in_bucket = (gaps >= low) & (gaps <= high)
        print("  {:>4}-{:<6} frames: {:5d} gaps, {:.1%} of missing frames".format(
            low, high if high < 10 ** 9 else "+", int(in_bucket.sum()),
            gaps[in_bucket].sum() / missing_frames))
    seen = graded & ~missing
    print("ball speed:", pct(ball[seen, 2], (5, 50, 95, 99)),
          "| <0.5: {:.1%}".format(np.mean(ball[seen, 2] < 0.5)))
    print("ball outside 0-1: x {:.2%}  y {:.2%}".format(
        np.mean((ball[seen, 0] < 0) | (ball[seen, 0] > 1)),
        np.mean((ball[seen, 1] < 0) | (ball[seen, 1] > 1))))

    section("3. smart tagging")
    events = load_smart_tagging.load()
    tags = pd.DataFrame([dict(code=e["code"].lower(), start=e["start_frame"], end=e["end_frame"],
                              **{k: (v or "").lower() for k, v in e["tags"].items()})
                         for e in events])
    tags["dur_s"] = (tags.end - tags.start) / FPS
    print(tags.groupby("code").dur_s.describe()[["count", "min", "50%", "max"]].round(1))
    print(pd.crosstab(tags.code, tags.get("Team")))
    if "Type" in tags:
        print(pd.crosstab(tags.Type, tags.get("Team")))
    coverage = np.zeros(N, dtype=np.int16)
    for start, end in zip(tags.start, tags.end):
        coverage[start:end] += 1
    print("graded frames under >=1 tag: {:.1%}  >=2 tags: {:.1%}".format(
        np.mean(coverage[graded] >= 1), np.mean(coverage[graded] >= 2)))
    _, untagged = runs((coverage == 0) & graded)
    print("untagged stretches s:", pct(untagged / FPS), "| longest {:.0f} s".format(
        untagged.max() / FPS if untagged.size else 0))

    attack_team = np.full(N, -1, dtype=np.int8)
    defend_team = np.full(N, -1, dtype=np.int8)
    for row in tags.itertuples():
        side = TEAM_CODE.get(getattr(row, "Team", ""), -1)
        target = attack_team if row.code in ATTACK else defend_team if row.code in DEFEND else None
        if target is None or side < 0:
            continue
        span = target[row.start:row.end]
        span[(span >= 0) & (span != side)] = 2  # conflicting tags on the same frame
        span[span < 0] = side
    overlap = graded & np.isin(attack_team, (0, 1)) & np.isin(defend_team, (0, 1))
    print("frames with attack+defend tags: {} | same team: {:.1%}  other team: {:.1%}".format(
        int(overlap.sum()), np.mean(attack_team[overlap] == defend_team[overlap]),
        np.mean(attack_team[overlap] != defend_team[overlap])))
    print("frames with conflicting attack tags: {:.2%}".format(np.mean(attack_team[graded] == 2)))

    set_pieces = tags[tags.code == "set pieces"]
    for label, group in [("all", set_pieces)] + list(set_pieces.groupby(set_pieces.get("Type"))):
        before, after = [], []
        for start in group.start:
            before.append(slice(max(start - SET_PIECE_WINDOW, 0), start))
            after.append(slice(start, start + SET_PIECE_WINDOW))

        def stats(windows):
            frames = np.concatenate([np.arange(N)[w] for w in windows])
            return "ball missing {:.0%}, ball speed p50 {:.1f}, player speed p50 {:.1f}".format(
                missing[frames].mean(), np.nanmedian(ball[frames, 2]),
                np.nanmedian(players[frames, :, 2]))
        print("  set piece {} (n={}): 2 s before start: {} | 2 s after: {}".format(
            label, len(group), stats(before), stats(after)))
    print("kickoff time-base check:", load_smart_tagging.check_time_base(events))

    section("4. public 5 minutes (understanding the labels, not tuning)")
    labels = pd.read_csv(GT_PUBLIC).label.str.strip().str.upper().values
    for label in "HAD":
        frames = labels == label
        line = "  {}: {} frames | ball missing {:.0%} | ball speed p50 {:.1f} | player speed p50 {:.1f}".format(
            label, int(frames.sum()), missing[frames].mean(), np.nanmedian(ball[frames, 2]),
            np.nanmedian(players[frames, :, 2]))
        if label in "HA":
            own = "HA".index(label)
            line += " | attack tag own {:.0%} other {:.0%} none {:.0%}".format(
                np.mean(attack_team[frames] == own), np.mean(attack_team[frames] == 1 - own),
                np.mean(attack_team[frames] < 0))
        print(line)
    _, dead = runs(labels == "D")
    print("  D stretches: {} | length s: {}".format(dead.size, pct(dead / FPS)))


if __name__ == "__main__":
    main()
