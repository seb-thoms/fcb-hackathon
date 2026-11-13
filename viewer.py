"""Watch the match with the tracking on top: video + homography overlay, 2D pitch and a label strip.

Everything is drawn locally in an OpenCV window; nothing leaves the machine.

    python viewer.py                                   # interactive, from the first-half kickoff
    python viewer.py --start 113742                    # second-half kickoff
    python viewer.py --no-video                        # pitch and strip only, much faster
    python viewer.py --start 11185 --end 18684 --out public_window.mp4   # render a clip instead

Keys (click the window first):
    space  play / pause          , .  -/+ 1 frame          j l  -/+ 1 s        J L  -/+ 10 s
    n p    next / previous ground-truth change              [ ]  slower / faster
    o      toggle the video overlay                         q    quit

The strip at the bottom is +-15 s around the current frame: ground truth (public 5 minutes only,
X elsewhere), the smooth50 baseline, whether the ball is tracked, the Smart Tagging attacking team
and set pieces. Colours: H blue, A red, D grey, X/none dark, conflicting tags purple.
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from explore_data import ATTACK, TEAM_CODE, load_atd  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
N = GAMES[GAME]["n_video_frames"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
KICKOFFS = {1: 11185, 2: 113742}  # video frames, from load_smart_tagging.KICKOFF_VIDEO_FRAMES
SOURCE_W = 1920  # tactical video width in pixels; the homography maps into this
VIDEO_W, VIDEO_H = 960, 540
MARGIN = 12
PITCH_W = 560
PITCH_H = int(round(PITCH_W * WIDTH / LENGTH))
PANEL_W = PITCH_W + 2 * MARGIN
STRIP_SECONDS = 15
ROW_H = 20
LABEL_W = 110
BALL_GHOST_SECONDS = 10  # how long the last known ball position stays drawn after losing it
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BASELINE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
WINDOW = "MTAH viewer"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colours, BGR
HOME, AWAY = (215, 120, 30), (50, 50, 220)
DEAD, NONE, CONFLICT = (140, 140, 140), (45, 45, 45), (170, 60, 170)
BALL, SET_PIECE, SEEN = (0, 230, 255), (0, 200, 255), (90, 190, 90)
GRASS, LINE, BG, TEXT = (60, 120, 50), (235, 235, 235), (25, 25, 25), (230, 230, 230)
LABEL_COLOURS = np.array([HOME, AWAY, DEAD, NONE], dtype=np.uint8)  # H A D X
TAG_COLOURS = np.array([NONE, HOME, AWAY, CONFLICT], dtype=np.uint8)  # none H A conflict
LABEL_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def label_codes(path):
    labels = pd.read_csv(path).label.astype(str).str.strip().str.upper()
    return labels.map(LABEL_CODE).fillna(3).values.astype(np.int8)


class Match:
    """Every per-frame array the viewer draws from, indexed by video frame."""

    def __init__(self):
        paths = asset_paths()
        root = ET.parse(paths["atd_meta"]).getroot()
        self.team = team_of_each_track(paths["atd_meta"])
        self.goalkeeper = np.array([
            any(p.findtext("Name") == "position_type" and p.findtext("Value") == "Goalkeeper"
                for p in player.iter("ProviderParameter"))
            for player in root.iter("Player")])
        home_team = next(root.iter("Team"))
        direction = {p.findtext("Name"): p.findtext("Value")
                     for p in home_team.iter("ProviderParameter")}
        self.home_attacks_right_first = direction.get(
            "attack_direction_first_half") == "left_to_right"

        print("loading tracking ...")
        self.players, self.ball = load_atd(paths["atd_raw"])
        self.homography, _, self.valid = load_homography.load()

        self.half = np.zeros(N, dtype=np.int8)
        for half, first, last in half_video_blocks(GAME):
            self.half[first:last + 1] = half
        self.gt = label_codes(GT_PUBLIC)
        self.gt_changes = np.flatnonzero(np.diff(self.gt) != 0) + 1
        self.baseline = label_codes(BASELINE)

        self.ball_seen = ~np.isnan(self.ball[:, 0])
        seen_index = np.where(self.ball_seen, np.arange(N), -1)
        self.last_seen = np.maximum.accumulate(seen_index)

        self.tags = load_smart_tagging.load()
        self.attack_team = np.full(N, -1, dtype=np.int8)
        self.set_piece = np.zeros(N, dtype=bool)
        for event in self.tags:
            code = event["code"].lower()
            span = slice(event["start_frame"], event["end_frame"])
            if code == "set pieces":
                self.set_piece[span] = True
            side = TEAM_CODE.get((event["tags"].get("Team") or "").lower(), -1)
            if code in ATTACK and side >= 0:
                frames = self.attack_team[span]
                frames[(frames >= 0) & (frames != side)] = 2
                frames[frames < 0] = side

    def tags_at(self, frame):
        return [event for event in self.tags if event["start_frame"] <= frame < event["end_frame"]]

    def clock(self, frame):
        half = self.half[frame]
        if not half:
            return "not graded"
        seconds = (frame - KICKOFFS[half]) / FPS + (45 * 60 if half == 2 else 0)
        if frame < KICKOFFS[half]:
            return "half {} (before kickoff)".format(half)
        return "half {}  {:02d}:{:02d}".format(half, int(seconds // 60), int(seconds % 60))


class VideoReader:
    """Sequential reads while playing, a seek only when jumping."""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise IOError("cannot open video: {}".format(path))
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count != N:
            print("WARNING: video reports {} frames, expected {}".format(count, N))
        self.next_frame = 0
        self.cached = (None, None)

    def get(self, frame):
        if self.cached[0] == frame:
            return self.cached[1]
        if frame != self.next_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.capture.read()
        self.next_frame = frame + 1
        image = cv2.resize(image, (VIDEO_W, VIDEO_H)) if ok else None
        self.cached = (frame, image)
        return image


def text(image, string, x, y, colour=TEXT, scale=0.45, thickness=1):
    cv2.putText(image, string, (int(x), int(y)), FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_video(match, reader, frame, overlay):
    image = reader.get(frame) if reader else None
    if image is None:
        image = np.full((VIDEO_H, VIDEO_W, 3), BG, dtype=np.uint8)
        text(image, "video off" if reader is None else "no video frame", 20, 40)
        return image
    image = image.copy()
    if not overlay or not match.valid[frame]:
        return image

    matrix = match.homography[frame]
    scale = VIDEO_W / SOURCE_W
    corners = np.array(load_homography.pitch_corners(matrix)) * scale
    cv2.polylines(image, [corners.astype(np.int32)], True, LINE, 1, cv2.LINE_AA)

    xs, ys = load_homography.project(matrix, match.players[frame, :, 0], match.players[frame, :, 1])
    for slot in np.flatnonzero(np.isfinite(xs)):
        centre = (int(xs[slot] * scale), int(ys[slot] * scale))
        colour = HOME if match.team[slot] == 0 else AWAY
        cv2.circle(image, centre, 6, colour, -1, cv2.LINE_AA)
        cv2.circle(image, centre, 9 if match.goalkeeper[slot] else 6, LINE, 1, cv2.LINE_AA)
        text(image, str(slot + 1), centre[0] + 7, centre[1] - 6, LINE, 0.35)

    last = match.last_seen[frame]
    if last >= 0 and frame - last <= BALL_GHOST_SECONDS * FPS:
        bx, by = load_homography.project(matrix, match.ball[last, 0], match.ball[last, 1])
        centre = (int(bx * scale), int(by * scale))
        filled = last == frame
        cv2.circle(image, centre, 5, BALL, -1 if filled else 1, cv2.LINE_AA)
    return image


def pitch_point(x, y):
    """Normalised pitch coordinates -> panel pixels."""
    return int(MARGIN + x * PITCH_W), int(MARGIN + y * PITCH_H)


def draw_pitch(match, frame):
    image = np.full((PITCH_H + 2 * MARGIN, PANEL_W, 3), BG, dtype=np.uint8)
    cv2.rectangle(image, pitch_point(0, 0), pitch_point(1, 1), GRASS, -1)
    cv2.rectangle(image, pitch_point(0, 0), pitch_point(1, 1), LINE, 1)
    cv2.line(image, pitch_point(0.5, 0), pitch_point(0.5, 1), LINE, 1)
    cv2.circle(image, pitch_point(0.5, 0.5), int(9.15 / LENGTH * PITCH_W), LINE, 1, cv2.LINE_AA)
    for depth, wide in ((16.5, 40.32), (5.5, 18.32)):
        dx, top, bottom = depth / LENGTH, 0.5 - wide / 2 / WIDTH, 0.5 + wide / 2 / WIDTH
        cv2.rectangle(image, pitch_point(0, top), pitch_point(dx, bottom), LINE, 1)
        cv2.rectangle(image, pitch_point(1 - dx, top), pitch_point(1, bottom), LINE, 1)

    for slot in np.flatnonzero(np.isfinite(match.players[frame, :, 0])):
        centre = pitch_point(*match.players[frame, slot, :2])
        colour = HOME if match.team[slot] == 0 else AWAY
        cv2.circle(image, centre, 5, colour, -1, cv2.LINE_AA)
        if match.goalkeeper[slot]:
            cv2.circle(image, centre, 8, LINE, 1, cv2.LINE_AA)

    last = match.last_seen[frame]
    if last >= 0 and frame - last <= BALL_GHOST_SECONDS * FPS:
        filled = last == frame
        cv2.circle(image, pitch_point(*match.ball[last, :2]), 4, BALL, -1 if filled else 1,
                   cv2.LINE_AA)

    half = match.half[frame]
    if half:
        home_right = match.home_attacks_right_first == (half == 1)
        text(image, "H attacks " + ("->" if home_right else "<-"), MARGIN + 4, MARGIN + 14, HOME)
    return image


def draw_info(match, frame, height, playing, speed):
    image = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    names = "HADX"
    seen = match.ball_seen[frame]
    lost = "" if seen else " (lost {:.1f} s)".format(
        (frame - match.last_seen[frame]) / FPS if match.last_seen[frame] >= 0 else float("nan"))
    active = np.isfinite(match.players[frame, :, 0])
    lines = [
        "frame {}   {}".format(frame, match.clock(frame)),
        "ground truth: {}   baseline50: {}".format(names[match.gt[frame]],
                                                    names[match.baseline[frame]]),
        "ball: {}{}   players H/A: {}/{}".format("tracked" if seen else "missing", lost,
                                                int((active & (match.team == 0)).sum()),
                                                int((active & (match.team == 1)).sum())),
    ]
    for event in match.tags_at(frame)[:4]:
        extra = event["tags"].get("Type") or ""
        lines.append("tag: {} | {} {}".format(event["code"].lower(),
                                               (event["tags"].get("Team") or "-").lower(), extra))
    lines.append("{}  x{:.2g}".format("playing" if playing else "paused", speed))
    for row, line in enumerate(lines):
        text(image, line, MARGIN, 20 + row * 20)
    return image


def draw_strip(match, frame, width):
    rows = [
        ("ground truth", LABEL_COLOURS[match.gt]),
        ("baseline50", LABEL_COLOURS[match.baseline]),
        ("ball tracked", np.where(match.ball_seen[:, None], SEEN, NONE).astype(np.uint8)),
        ("attack tag", TAG_COLOURS[match.attack_team + 1]),
        ("set piece", np.where(match.set_piece[:, None], SET_PIECE, NONE).astype(np.uint8)),
    ]
    span = STRIP_SECONDS * FPS
    columns = width - LABEL_W - MARGIN
    offsets = np.round(np.linspace(-span, span, columns)).astype(int)
    frames = frame + offsets
    inside = (frames >= 0) & (frames < N)
    frames = np.clip(frames, 0, N - 1)

    image = np.full((len(rows) * ROW_H + 26, width, 3), BG, dtype=np.uint8)
    for index, (name, colours) in enumerate(rows):
        band = colours[frames].copy()
        band[~inside] = BG
        top = 6 + index * ROW_H
        image[top:top + ROW_H - 4, LABEL_W:LABEL_W + columns] = band[None]
        text(image, name, MARGIN, top + ROW_H - 8)
    centre = LABEL_W + columns // 2
    bottom = 6 + len(rows) * ROW_H
    cv2.line(image, (centre, 2), (centre, bottom), LINE, 1)
    text(image, "-{} s".format(STRIP_SECONDS), LABEL_W, bottom + 14)
    text(image, "now", centre - 12, bottom + 14)
    text(image, "+{} s".format(STRIP_SECONDS), LABEL_W + columns - 40, bottom + 14)
    return image


def render(match, reader, frame, overlay=True, playing=False, speed=1.0):
    video = draw_video(match, reader, frame, overlay)
    pitch = draw_pitch(match, frame)
    info = draw_info(match, frame, VIDEO_H - pitch.shape[0], playing, speed)
    top = np.hstack([video, np.vstack([pitch, info])])
    return np.vstack([top, draw_strip(match, frame, top.shape[1])])


def write_clip(match, reader, start, end, path):
    first = render(match, reader, start)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    for frame in range(start, end + 1):
        writer.write(first if frame == start else render(match, reader, frame))
        if (frame - start) % (10 * FPS) == 0:
            print("  {} / {}".format(frame - start, end - start))
    writer.release()
    print("wrote", path)


def interactive(match, reader, start, scale):
    frame, playing, speed, overlay, dirty = start, False, 1.0, True, True
    steps = {ord(","): -1, ord("."): 1, ord("j"): -FPS, ord("l"): FPS,
             ord("J"): -10 * FPS, ord("L"): 10 * FPS}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        started = time.time()
        if dirty or playing:
            canvas = render(match, reader, frame, overlay, playing, speed)
            if scale != 1:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, canvas)
            dirty = False
        budget = 1000 / (FPS * speed) - (time.time() - started) * 1000
        key = cv2.waitKey(max(1, int(budget)) if playing else 30) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            playing, dirty = not playing, True
        elif key in steps:
            frame, dirty = frame + steps[key], True
        elif key in (ord("n"), ord("p")):
            changes = match.gt_changes
            if key == ord("n"):
                later = changes[changes > frame]
                frame = int(later[0]) if later.size else frame
            else:
                earlier = changes[changes < frame]
                frame = int(earlier[-1]) if earlier.size else frame
            dirty = True
        elif key == ord("["):
            speed, dirty = max(0.25, speed / 2), True
        elif key == ord("]"):
            speed, dirty = min(8.0, speed * 2), True
        elif key == ord("o"):
            overlay, dirty = not overlay, True
        elif playing:
            frame += 1
        frame = int(np.clip(frame, 0, N - 1))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=KICKOFFS[1], help="first video frame")
    parser.add_argument("--end", type=int, default=None, help="last frame, with --out")
    parser.add_argument("--out", help="write an mp4 of start..end instead of opening a window")
    parser.add_argument("--no-video", action="store_true", help="skip decoding the video")
    parser.add_argument("--scale", type=float, default=1.0, help="window scale, e.g. 0.8")
    args = parser.parse_args()

    match = Match()
    reader = None if args.no_video else VideoReader(asset_paths()["video"])
    if args.out:
        end = args.end if args.end is not None else min(args.start + 60 * FPS, N - 1)
        write_clip(match, reader, args.start, end, args.out)
    else:
        interactive(match, reader, args.start, args.scale)


if __name__ == "__main__":
    main()
