"""External validation on REAL possession labels from other matches: Metrica Sports sample games.

RULES.md forbids tuning on BAR-ATH's ground truth but allows labels from other games. This module
derives per-frame H/A/D labels for the three public Metrica Sports sample games from their event
data, degrades their (corrected) tracking until it looks like BAR-ATH's automatic ATD, and scores
any labeller with the official evaluation/possession_metrics.py.
Data: Metrica Sports sample data, https://github.com/metrica-sports/sample-data (free to use;
acknowledge the source if anything built on it is made public). Raw files are downloaded to
data/external/metrica_sample/ and parsed arrays cached in data/external/cache/ (gitignored).

    import sys; sys.path.insert(0, "solution/validation")
    import metrica_sample as ms
    data = ms.load_game(1)                        # degraded (seed 0); degrade=False for clean
    ms.score(my_labels, data)                     # spellF1, macroF1, possession counts ...
    ms.evaluate(my_label_fn, seed=[0, 1, 2])      # games 1-3, prints per game and mean
    .venv/bin/python solution/validation/metrica_sample.py   # label / degradation / baseline stats

load_game() returns the solution/atd.py format: players (n,S,2) float32 metres, x along the
length in [0,105], y in [0,68], NaN when unseen; ball (n,2); team (S,) int8 0=home 1=away; gk (S,);
n_frames; pitch; fps; halves [(half, first, last)] inclusive (kick-off to the period's last
tracking row); home_attacks_positive_x_first_half; labels '<U1' (H/A/D in halves, X outside).
Frame index = 0-based tracking row (Metrica frame - 1). evaluate() hands label_fn only INPUT_KEYS,
never the labels. Degraded games carry 2 extra "ghost" slots per team (see DEGRADATION).

LABEL RULE (rule="control", default) -- events are processed in Start Frame order, per half
D, ball dead:
  * Hard stoppage -> D from its termination frame until the next restart:
      BALL OUT, SHOT with result OUT or GOAL: the first frame in [Start, End+25] at which the
        corrected ball centre is outside the 105x68 pitch (found for 194 of 203 ball-outs), else
        End Frame -- i.e. the stoppage starts exactly when the ball crosses the line;
      BALL LOST FORCED (play halted by the referee): End Frame;  PASS ...OFFSIDE: End Frame
        (the receiver is flagged on reception);  BALL LOST OFFSIDE / HAND BALL / END HALF /
        VOLUNTARY: Start Frame.  After END HALF: D to the end of the half.
  * Foul without a hard stoppage (CHALLENGE ...FAULT..., FAULT RECEIVED): if the free kick follows
    with no live action (PASS/SHOT/RECOVERY/BALL LOST/CARRY/plain CHALLENGE) after the last foul
    event, D from that last foul event; a live action after it means advantage was played.
  * Fallback (no stoppage event at all before a SET PIECE; 1 per game): D from the end of the
    last live event.
  * Restart = SET PIECE start frame, or a RECOVERY during dead ball (Metrica's marker for a goal
    kick / dropped ball) unless it is followed directly by a SET PIECE or END HALF. RETAKEN set
    pieces are ordinary restarts; their aborted play ends at the referee's FORCED.
Possession while live: the restarting team owns the ball from the restart frame. A RECOVERY by
  the other team opens a candidate at its frame. The candidate is CONFIRMED -- possession switches
  AT THE CANDIDATE FRAME, the new team's first touch -- by control evidence before the owner
  touches the ball again: a PASS (Metrica PASS = completed to a team-mate, incl. a header pass
  after an aerial duel), a SHOT, FAULT RECEIVED (fouled while in possession), a CARRY >= 13 frames
  (Game 3 only; 1-frame carries are artefacts), or the recovering team's own BALL LOST / BALL OUT
  >= HOLD_FRAMES=13 (~0.5 s) after the recovery (the player kept it). It is DROPPED (no change) when
  the owner touches the ball again, when the new team's BALL LOST / BALL OUT comes < 13 frames
  after its recovery (one-touch interception, clearance or header), or at a stoppage. A non-owner
  PASS/SHOT without a recorded RECOVERY switches at its own start. CHALLENGE (duel won/lost) never
  changes possession by itself. One-touch vs held is bimodal in the data (0-6 vs >= 14 frames).
rule="touch": Metrica's own possession definition -- any RECOVERY or other on-ball event by the
  non-owner switches at its frame (one-touch interceptions/clearances become short possessions).

Resolved from the event sequences: goal kicks are not SET PIECEs (RECOVERY + PASS/BALL LOST
"GOAL KICK"); a foul is often followed by the opponent's RECOVERY + BALL LOST FORCED after the
whistle, so FORCED decides the stoppage frame; aerial duels come as CHALLENGE AERIAL-WON plus
RECOVERY plus PASS HEAD at one frame (switch at that frame); a THEFT without an opposing RECOVERY
leaves possession unchanged; a goal kick followed only by an aerial foul still restarts play.

Label stats (graded frames; control rule / touch rule possessions):
    game 1: 226 possessions, mean 14.9 s (median 9.8 s), D 41.5% in 99 stretches  | touch 380, 8.9 s
    game 2: 220 possessions, mean 15.0 s (median 10.3 s), D 41.5% in 99 stretches | touch 354, 9.3 s
    game 3: 261 possessions, mean 14.9 s (median 10.2 s), D 31.9% in 102 stretches| touch 404, 9.6 s
(BAR-ATH's public 5 minutes, from DATA.md: 17 possessions, mean 14.2 s, D 19.6%.)
Check against Metrica's own corrected tracking, which blanks the ball while it is dead (never used
to build the labels): the ball is missing on 92/97/99% of our D frames and 0.8/1.3/0.4% of live
frames; it reappears exactly at our restart frame (median offset 0) and vanishes a median 2-4
frames after our stoppage start (Metrica blanks at BALL OUT End Frame + 1, End Frame being a
median 2-6 frames after the line crossing; at FORCED End + 1; mostly at the last foul event + 1).

DEGRADATION (degrade=True; deterministic in (seed, game); degrade={...} overrides DEGRADE)
  ball: short (< 3 s) gaps interpolated as the provider does; dead-ball visibility synthesized
  (ball keeps rolling 0.2-2 s after the stoppage in 50% of cases, lies at the restart spot for
  5-30% of the stoppage, at most 10 s, before 70% of restarts); live losses: 30% of stoppages
  lose the ball 1-6 s early, 30% of restarts lose it for 1-5 s, plus interior gaps (0.6 per live
  minute, lognormal median 5 s, 3-30 s); smooth noise 0.21 m per axis (0.30 m RMS).
  players: per-slot dropouts (0.5/min, median 2 s), common-mode bad windows (1.2/min, median 4 s,
  each slot lost with p=0.45), ghosts in 2 extra slots per team (1.5 episodes/min/team, median 3 s:
  75% team flips = a player seen as the other team, 25% duplicates ~1 m away), smooth noise
  0.10 m per axis (0.14 m RMS, max 0.03 m/frame change: no teleports).
  Achieved, seed 0, games 1/2/3 (targets): ball present 0.61/0.58/0.66 (0.59; game 3 has less
  dead time); missing on live 0.10/0.11/0.10 (0.09); on dead 0.81/0.86/0.86 (>= 2/3; 0.86 is what
  0.59 present and 0.09 live imply at a 41% D share); missing frames in gaps 2-10 s
  0.14/0.14/0.22 (0.15), 10-30 s 0.45/0.34/0.37 (0.41), > 30 s 0.41/0.53/0.41 (0.44); gaps < 3 s
  0/0/1; players median 22, mean 20.7-20.8 (20.5), < 18 on 8-9% (9%), a 12th object in a team
  5.7-6.1% (6.4%), > 22 objects 2.0-3.0% (2.4%); dropout chunks median 3 s.

BASELINE (starter/baseline.py replica: nearest player, D where no ball, +/-50 majority)
    spellF1 / macroF1     clean            degraded (seed 0)    BAR-ATH full match
    game 1                0.451 / 0.838    0.319 / 0.762        0.2433 / 0.6724
    game 2                0.443 / 0.856    0.329 / 0.773
    game 3                0.413 / 0.845    0.321 / 0.770
  Seeds 0-2 move a game's degraded score by about +/-0.015, so average seeds (evaluate(seed=[..])).

CAVEATS
  * Labels come from events, not from a corrected tracking "owner" channel. Switches sit on event
    touch frames (Metrica frames are tied to its tracking but carry a few frames of jitter);
    stoppages start at the line crossing, the referee's FORCED or the last foul event, and that
    can differ by up to ~1 s from where a tracking operator marks the ball dead.
  * The control rule follows CHALLENGE.md's wording (no change on a touch without control), and its
    mean possession (15 s) is close to BAR-ATH's public 14.2 s. A tracking owner that flips on every
    opponent touch would look like rule="touch" (1.6x more possessions). The published BAR-ATH
    baseline sweep is mixed evidence (degraded, games 1-3 x seeds 0-2, window 0/25/50/125):
    control 0.048/0.236/0.331/0.375, touch 0.083/0.270/0.328/0.289, BAR-ATH 0.049/0.200/0.243/0.224.
    The raw w0 score matches control; the w125 < w50 order matches touch. If a decision depends on
    it, score both rules.
  * The degraded baseline is still easier than BAR-ATH (spellF1 ~0.33 vs 0.24, macroF1 ~0.77 vs
    0.67). Raising dead-ball ball visibility to miss_dead ~0.58 reproduces BAR-ATH's baseline
    macroF1 sweep almost exactly but breaks the 59% ball-present target and barely moves spellF1,
    so the spellF1 gap comes from something unmodelled that splits possessions (tracklet identity
    switches, aerial-ball projection error, ...) or from the label convention. Use the harness to
    rank variants, not to predict the board score.
  * Not modelled: slot identity switches/fragmentation (Metrica slots are persistent players),
    false ball detections, Smart Tagging and video. Dead-ball ball positions are synthetic.
"""
import io
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the official scorer, reused as-is)

# ----------------------------------------------------------------------------- params
DATA_DIR = os.path.join(ROOT, "data", "external", "metrica_sample")  # data/ is gitignored
CACHE_DIR = os.path.join(ROOT, "data", "external", "cache")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
CACHE_VERSION = 1
FPS = 25
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
FILES = {
    1: ["data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["data/Sample_Game_3/Sample_Game_3_events.json",
        "data/Sample_Game_3/Sample_Game_3_metadata.xml",
        "data/Sample_Game_3/Sample_Game_3_tracking.txt"],
}

# Label rule (see module docstring). HOLD: a recovering player who keeps the ball at least this
# long before his own BALL LOST / BALL OUT was in control (he took a second touch / carried it).
HOLD_FRAMES = 13            # ~0.5 s
OUT_SEARCH_PAD = 25         # frames after a BALL OUT's End Frame searched for the line crossing
HARD_LOST = {"FORCED", "OFFSIDE", "HAND BALL", "HANDBALL", "END HALF", "VOLUNTARY", "GOAL"}
ON_BALL = {"recovery", "pass", "pass_end", "shot", "shot_end", "carry", "lost", "lost_end", "out",
           "fault_recv"}
CONTROL = {"pass", "pass_end", "shot", "shot_end", "fault_recv"}   # CARRY: only if >= HOLD long
TERMINAL = {"out", "shot_end", "pass_end", "lost_end"}
LIVE = {"recovery", "pass", "shot", "carry", "lost", "challenge"}


# ----------------------------------------------------------------------------- download / paths
def _path(rel):
    return os.path.join(DATA_DIR, rel)


def ensure_downloaded(game_id):
    """Fetch the raw files of one game from the public Metrica repository if they are missing."""
    for rel in FILES[game_id]:
        dst = _path(rel)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        print("downloading", rel, "...")
        urllib.request.urlretrieve(URL + rel, tmp)
        os.replace(tmp, dst)


# ----------------------------------------------------------------------------- tracking parsers
def _read_csv_team(path):
    """One Metrica CSV tracking file -> (period, frame, player names, xy (n,P,2) 0-1, ball (n,2))."""
    with open(path) as fh:
        header = [fh.readline() for _ in range(3)][2].rstrip("\r\n").split(",")
    names = [h for h in header[3:-2] if h]
    arr = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=np.float64)
    xy = arr[:, 3:3 + 2 * len(names)].reshape(len(arr), len(names), 2)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), names, xy, arr[:, -2:]


def _parse_csv_game(game_id):
    _, home_path, away_path = (_path(r) for r in FILES[game_id])
    per_h, fr_h, names_h, xy_h, ball_h = _read_csv_team(home_path)
    per_a, fr_a, names_a, xy_a, _ = _read_csv_team(away_path)
    if not (np.array_equal(fr_h, fr_a) and np.array_equal(fr_h, np.arange(1, len(fr_h) + 1))):
        raise ValueError("game {}: tracking frames are not 1..n in both files".format(game_id))
    players = np.concatenate([xy_h, xy_a], axis=1)
    team = np.array([0] * len(names_h) + [1] * len(names_a), dtype=np.int8)
    return players, ball_h, team, per_h, names_h + names_a


def _parse_epts_game(game_id):
    _, meta_path, raw_path = (_path(r) for r in FILES[game_id])
    root = ET.parse(meta_path).getroot()
    local = root.find(".//Score").get("idLocalTeam")
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    name_of = {p.get("id"): p.findtext("Name") for p in root.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [chan[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), refs[0::2]))
    order = []
    for _, _, pids in specs:
        order += [p for p in pids if p not in order]
    slots = sorted(order, key=lambda p: (team_of[p] != local, order.index(p)))  # home first
    col = {p: i for i, p in enumerate(slots)}

    with open(raw_path) as fh:
        text = fh.read().replace(":", ",").replace(";", ",")
    width = 1 + 2 * max(len(p) for _, _, p in specs) + 2
    arr = pd.read_csv(io.StringIO(text), header=None, names=range(width)).to_numpy(np.float64)
    frame = arr[:, 0].astype(np.int64)
    if not np.array_equal(frame, np.arange(1, len(frame) + 1)):
        raise ValueError("game {}: tracking frames are not 1..n".format(game_id))
    n = len(frame)
    players = np.full((n, len(slots), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for first, last, pids in specs:
        rows = np.flatnonzero((frame >= first) & (frame <= last))
        k = len(pids)
        players[rows[:, None], np.array([col[p] for p in pids])[None, :]] = \
            arr[rows, 1:1 + 2 * k].reshape(len(rows), k, 2)
        ball[rows] = arr[rows, 1 + 2 * k:3 + 2 * k]
    period = np.zeros(n, dtype=np.int64)
    for p, key in ((1, "first_half"), (2, "second_half")):
        a, b = params.get(key + "_start"), params.get(key + "_end")
        if a and b:
            period[(frame >= int(a)) & (frame <= int(b))] = p
    team = np.array([0 if team_of[p] == local else 1 for p in slots], dtype=np.int8)
    return players, ball, team, period, [name_of[p] for p in slots]


def _load_tracking(game_id, use_cache=True):
    """Clean tracking in metres, cached as npz: players (n,S,2), ball (n,2), team, period, names."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "metrica_game{}_v{}.npz".format(game_id, CACHE_VERSION))
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    ensure_downloaded(game_id)
    parse = _parse_epts_game if game_id == 3 else _parse_csv_game
    players, ball, team, period, names = parse(game_id)
    scale = np.array(PITCH)
    out = dict(players=(players * scale).astype(np.float32), ball=(ball * scale).astype(np.float32),
               team=team, period=period.astype(np.int8), names=np.array(names))
    np.savez(cache, **out)
    return out


# ----------------------------------------------------------------------------- events
def _tokens(subtype):
    if subtype is None or (isinstance(subtype, float) and np.isnan(subtype)):
        return frozenset()
    return frozenset(t.strip().upper() for t in str(subtype).split("-") if t.strip())


def load_events(game_id):
    """Events as a DataFrame on 0-based tracking rows: team (0 home, 1 away, -1 none), type,
    subtype ('-'-joined), toks (frozenset), period, start, end (>= start), sx, sy, ex, ey (0-1)."""
    ensure_downloaded(game_id)
    if game_id == 3:
        raw = json.load(open(_path(FILES[3][0])))["data"]
        root = ET.parse(_path(FILES[3][1])).getroot()
        local = root.find(".//Score").get("idLocalTeam")
        rows = []
        for e in raw:
            st = e.get("subtypes")
            if isinstance(st, dict):
                st = [st]
            sub = "-".join(s["name"] for s in st) if st else ""
            end = e.get("end") or {}
            rows.append(dict(
                team=-1 if not e.get("team") else (0 if e["team"]["id"] == local else 1),
                type=e["type"]["name"].upper(), subtype=sub, period=int(e["period"]),
                start=int(e["start"]["frame"]) - 1,
                end=int(end["frame"]) - 1 if end.get("frame") is not None else -1,
                sx=e["start"].get("x"), sy=e["start"].get("y"), ex=end.get("x"), ey=end.get("y")))
        ev = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(_path(FILES[game_id][0]))
        ev = pd.DataFrame(dict(
            team=np.where(raw["Team"].str.lower() == "home", 0,
                          np.where(raw["Team"].str.lower() == "away", 1, -1)),
            type=raw["Type"].str.upper(), subtype=raw["Subtype"].fillna(""),
            period=raw["Period"].astype(int), start=raw["Start Frame"].astype(int) - 1,
            end=raw["End Frame"].fillna(0).astype(int) - 1,
            sx=raw["Start X"], sy=raw["Start Y"], ex=raw["End X"], ey=raw["End Y"]))
    ev["end"] = np.maximum(ev["end"].to_numpy(), ev["start"].to_numpy())
    ev["toks"] = [_tokens(s) for s in ev["subtype"]]
    ev["kind"] = [_kind(t, k) for t, k in zip(ev["type"], ev["toks"])]
    # stable chronological order: by start frame, file order breaking ties
    ev = ev.reset_index().rename(columns={"index": "order"})
    return ev.sort_values(["start", "order"], kind="mergesort").reset_index(drop=True)


def _kind(etype, toks):
    if etype == "SET PIECE":
        return "restart"
    if etype == "BALL OUT":
        return "out"
    if etype == "SHOT":
        return "shot_end" if toks & {"OUT", "GOAL"} else "shot"
    if etype == "PASS":
        return "pass_end" if "OFFSIDE" in toks else "pass"
    if etype == "BALL LOST":
        return "lost_end" if toks & HARD_LOST else "lost"
    if etype == "RECOVERY":
        return "recovery"
    if etype == "FAULT RECEIVED":
        return "fault_recv"
    if etype == "CHALLENGE":
        return "fault" if "FAULT" in toks else "challenge"
    if etype == "CARRY":
        return "carry"
    return None  # CARD and anything unknown: no bearing on ball state or possession


# ----------------------------------------------------------------------------- labels from events
def _crossing(ball, first, last):
    """First frame in [first, last] where the (clean) ball centre is outside the pitch, or None."""
    seg = ball[first:last + 1]
    L, W = PITCH
    out = (seg[:, 0] < 0) | (seg[:, 0] > L) | (seg[:, 1] < 0) | (seg[:, 1] > W)
    idx = np.flatnonzero(out)
    return first + int(idx[0]) if len(idx) else None


def _termination_frame(e, ball, stats):
    k, toks = e["kind"], e["toks"]
    if k == "out" or (k in ("shot_end", "lost_end") and toks & {"OUT", "GOAL"}):
        hit = _crossing(ball, int(e["start"]), int(e["end"]) + OUT_SEARCH_PAD)
        stats["n_out"] += 1
        if hit is not None:
            stats["n_out_refined"] += 1
            return hit
        return int(e["end"])
    if k == "pass_end":
        return int(e["end"])          # the offside receiver is flagged on reception
    if k == "lost_end" and "FORCED" in toks:
        return int(e["end"])          # play halted by the referee: ball travel ends at the whistle
    return int(e["start"])            # offside / hand ball / end half / voluntary


def _recovery_restarts(evs, i):
    """A RECOVERY during dead ball is Metrica's marker for the ball being brought back into play
    other than by a SET PIECE (goal kick, dropped ball). It counts as a restart unless the next
    event is a SET PIECE or the end of the half (then it was only a pick-up during the stoppage)."""
    for j in range(i + 1, len(evs)):
        k = evs[j]["kind"]
        if k == "restart":
            return False
        if k == "lost_end" and "END HALF" in evs[j]["toks"]:
            return False
        if k in ON_BALL or k in ("fault", "challenge"):
            return True
    return False


def derive_labels(ev, n_frames, halves, ball, rule="control", hold=HOLD_FRAMES, trace=None):
    """Per-frame 'H'/'A'/'D' inside halves, 'X' outside, from the event stream. See docstring.

    rule='control' (default): possession changes at the new team's first touch only once that team
    shows control (a pass, shot, carry, a foul suffered in possession, or the recovering player
    keeping the ball >= `hold` frames before losing it). rule='touch': Metrica's own possession
    definition, every opponent RECOVERY / on-ball event switches immediately.
    Returns (labels, stats)."""
    labels = np.full(n_frames, "X", dtype="<U1")
    stats = dict(n_out=0, n_out_refined=0, n_stop_hard=0, n_stop_soft=0, n_stop_fallback=0,
                 n_recovery_restarts=0, n_cand=0, n_cand_confirmed=0, n_switch=0)
    recs = ev.to_dict("records")
    for half, f0, f1 in halves:
        evs = [e for e in recs if f0 <= e["start"] <= f1]
        cps = []                                   # change points (frame, 'H'|'A'|'D')
        state = dict(alive=False, owner=None, cand=None, soft=None, last_live=f0)

        def mark(frame, lab):
            frame = int(min(max(frame, f0), f1 + 1))
            if cps and frame < cps[-1][0]:
                frame = cps[-1][0]
            cps.append((frame, lab))

        def begin(frame, team):
            mark(frame, "HA"[team])
            state.update(alive=True, owner=team, cand=None, soft=None, last_live=frame)

        def switch(frame, team, why):
            if state["cand"] is not None:
                stats["n_cand_confirmed"] += 1
                if trace is not None:
                    trace.append((state["cand"], team, "confirmed:" + why, s - state["cand"]))
            mark(frame, "HA"[team])
            state.update(owner=team, cand=None)
            stats["n_switch"] += 1

        def drop(why):
            if state["cand"] is not None and trace is not None:
                trace.append((state["cand"], 1 - state["owner"], "dropped:" + why,
                              s - state["cand"]))
            state["cand"] = None

        for i, e in enumerate(evs):
            k, T, s = e["kind"], int(e["team"]), int(e["start"])
            if k is None:
                continue
            if not state["alive"]:
                if k == "restart" and T in (0, 1):
                    begin(s, T)
                elif k == "recovery" and T in (0, 1) and _recovery_restarts(evs, i):
                    stats["n_recovery_restarts"] += 1
                    begin(s, T)
                continue
            if k == "restart":                     # set piece with no explicit stoppage before it
                drop("restart")
                if state["soft"] is not None:
                    stats["n_stop_soft"] += 1
                    stop = state["soft"]
                else:
                    stats["n_stop_fallback"] += 1
                    stop = state["last_live"]
                mark(min(stop, s), "D")
                if T in (0, 1):
                    begin(s, T)
                continue
            if T in (0, 1) and k in ON_BALL:
                owner = state["owner"]
                if rule == "touch":
                    if T != owner:
                        switch(s, T, k)
                elif T == owner:
                    drop("owner_" + k)             # the owner touched it again: no change
                elif k == "recovery":
                    if state["cand"] is None:
                        state["cand"] = s
                        stats["n_cand"] += 1
                elif k in CONTROL:
                    switch(state["cand"] if state["cand"] is not None else s, T, k)
                elif k == "carry":                 # Game 3 only; 1-frame carries are artefacts
                    first = state["cand"] if state["cand"] is not None else s
                    if int(e["end"]) - first >= hold:
                        switch(first, T, "carry")
                else:                              # lost / lost_end / out by the non-owner
                    if state["cand"] is not None and s - state["cand"] >= hold:
                        switch(state["cand"], T, "hold_" + k)
                    else:
                        drop("onetouch_" + k)
            if k in TERMINAL:
                drop("stoppage")
                stats["n_stop_hard"] += 1
                mark(_termination_frame(e, ball, stats), "D")
                state.update(alive=False, owner=None, cand=None, soft=None)
                continue
            if k in ("fault", "fault_recv"):
                state["soft"] = s if state["soft"] is None else max(state["soft"], s)
            elif k in LIVE and state["soft"] is not None and s > state["soft"]:
                state["soft"] = None               # play went on (advantage)
            if k in LIVE or k == "fault_recv":
                state["last_live"] = max(state["last_live"], int(e["end"]))

        # paint the half from its change points
        seg = np.full(f1 - f0 + 1, "D", dtype="<U1")
        for j, (frame, lab) in enumerate(cps):
            stop = cps[j + 1][0] if j + 1 < len(cps) else f1 + 1
            seg[frame - f0:stop - f0] = lab
        labels[f0:f1 + 1] = seg
    return labels, stats


# ----------------------------------------------------------------------------- game metadata
def _runs(mask):
    """Half-open (starts, stops) of the True runs of a boolean array."""
    d = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _halves(ev, period):
    """[(half, first, last)] inclusive: from the half's first KICK OFF to its last tracking row."""
    out = []
    for p in (1, 2):
        ko = ev[(ev["kind"] == "restart") & ev["toks"].apply(lambda t: "KICK OFF" in t)
                & (ev["period"] == p)]
        rows = np.flatnonzero(period == p)
        if len(ko) and len(rows):
            out.append((p, int(ko["start"].min()), int(rows.max())))
    return out


def _gk_slots(players, team, halves):
    """Label-free goalkeepers: per half and team, the slot that is deepest towards its own goal on
    at least half of the frames it is tracked (and for at least a minute)."""
    gk = np.zeros(len(team), dtype=bool)
    for _, f0, f1 in halves:
        x = players[f0:f1 + 1, :, 0].astype(np.float64)
        for t in (0, 1):
            cols = np.flatnonzero(team == t)
            xt = x[:, cols]
            depth = xt if np.nanmean(xt) < PITCH[0] / 2 else PITCH[0] - xt
            depth = np.where(np.isnan(depth), np.inf, depth)
            ok = np.isfinite(depth).any(axis=1)
            counts = np.bincount(np.argmin(depth[ok], axis=1), minlength=len(cols))
            seen = np.isfinite(depth[ok]).sum(axis=0)
            gk[cols[(counts >= 0.5 * np.maximum(seen, 1)) & (counts >= 60 * FPS)]] = True
    return gk


# ----------------------------------------------------------------------------- degradation
# Targets (BAR-ATH automatic tracking, measured label-free): ball present on 59% of graded frames,
# missing on ~9% of live and >= ~2/3 of dead frames; missing frames 15% / 41% / 44% in gaps of
# 2-10 s / 10-30 s / > 30 s, no gaps < ~3 s; players median 22 tracked, < 18 on ~9% of frames, in
# chunks of seconds; smooth noise ~0.1-0.2 m (players), ~0.3 m (ball). Durations in seconds.
DEGRADE = dict(
    min_gap_s=3.0,                                   # provider interpolates shorter ball gaps
    tail_p=0.5, tail_s=(0.2, 2.0), tail_tau_s=0.8,   # ball still seen rolling on after the stop
    pre_p=0.7, pre_frac=(0.05, 0.3), pre_max_s=10.0,  # ball seen lying at the restart spot
    loss_before_p=0.3, loss_before_s=(1.0, 6.0),     # ball lost shortly before a stoppage
    loss_after_p=0.3, loss_after_s=(1.0, 5.0),       # ball lost right after a restart
    gap_per_min=0.6, gap_median_s=5.0, gap_sigma=0.6, gap_clip_s=(3.0, 30.0),  # live gaps
    ball_noise_m=0.21, player_noise_m=0.10, noise_sigma_frames=12,   # per-axis std, smoothing
    slot_drop_per_min=0.5, slot_drop_median_s=2.0, slot_drop_sigma=0.8, slot_drop_clip_s=(0.4, 20.0),
    bad_per_min=1.2, bad_median_s=4.0, bad_sigma=0.6, bad_clip_s=(1.0, 20.0), bad_q=0.45,
    bad_jitter_s=1.0,                                # common-mode dropouts of many slots at once
    ghost_slots_per_team=2, ghost_per_min=1.5, ghost_median_s=3.0, ghost_sigma=0.7,
    ghost_clip_s=(0.5, 30.0), ghost_flip_share=0.75, ghost_offset_m=1.0,  # duplicates / team flips
)


def _smooth_noise(rng, shape, std, sigma):
    """Gaussian-filtered white noise along axis 0, rescaled to the given per-axis std."""
    from scipy.ndimage import gaussian_filter1d
    w = gaussian_filter1d(rng.standard_normal(shape), sigma, axis=0, mode="reflect")
    return w * (std / w.std())


def _lognormal_frames(rng, size, median_s, sigma, clip_s):
    v = rng.lognormal(np.log(median_s * FPS), sigma, size)
    return np.clip(v, clip_s[0] * FPS, clip_s[1] * FPS).astype(np.int64)


def _interp_short_gaps(ball, halves, min_len):
    """Linear interpolation of NaN runs shorter than min_len with a valid frame on both sides."""
    out = ball.copy()
    for _, f0, f1 in halves:
        seg = out[f0:f1 + 1]
        starts, stops = _runs(np.isnan(seg[:, 0]))
        for a, b in zip(starts, stops):
            if b - a < min_len and a > 0 and b < len(seg):
                w = (np.arange(1, b - a + 1) / (b - a + 1.0))[:, None]
                seg[a:b] = seg[a - 1] * (1 - w) + seg[b] * w
    return out


def _degrade_ball(ball, labels, halves, rng, P):
    n = len(ball)
    clean = ball.astype(np.float64)
    out = clean.copy()
    live = (labels == "H") | (labels == "A")
    remove = np.zeros(n, dtype=bool)
    starts, stops = _runs(labels == "D")
    for a, b in zip(starts, stops):
        u = rng.random(5)                            # fixed number of draws per stoppage
        span = rng.random(4)
        after_live = a > 0 and live[a - 1]
        before_restart = b < n and live[b]
        miss = np.isnan(out[a:b, 0])
        first_nan = a + int(np.argmax(miss)) if miss.any() else b
        lost_before = after_live and u[0] < P["loss_before_p"]
        if lost_before:                              # lost a few s before the stop, never seen again
            m = int(np.interp(span[0], [0, 1], P["loss_before_s"]) * FPS)
            remove[max(a - m, 0):first_nan] = True
        elif after_live and u[1] < P["tail_p"] and miss.any():
            hist = clean[max(first_nan - 6, 0):first_nan]
            idx = np.flatnonzero(~np.isnan(hist[:, 0]))
            if len(idx) >= 2:                        # keep rolling with friction
                vel = (hist[idx[-1]] - hist[idx[0]]) / float(idx[-1] - idx[0])
                m = min(int(np.interp(span[1], [0, 1], P["tail_s"]) * FPS), b - first_nan)
                tau = P["tail_tau_s"] * FPS
                t = np.arange(1, m + 1, dtype=np.float64)[:, None]
                pos = hist[idx[-1]] + vel * tau * (1 - np.exp(-t / tau))
                pos[:, 0] = np.clip(pos[:, 0], -6, PITCH[0] + 6)
                pos[:, 1] = np.clip(pos[:, 1], -6, PITCH[1] + 6)
                fill = np.isnan(out[first_nan:first_nan + m, 0])
                out[first_nan:first_nan + m][fill] = pos[fill]
        if before_restart and u[2] < P["pre_p"]:     # ball lying at the restart spot
            q = b
            while q < min(n, b + FPS) and np.isnan(clean[q, 0]):
                q += 1
            if q < n and not np.isnan(clean[q, 0]):
                m = int(min(np.interp(span[2], [0, 1], P["pre_frac"]) * (b - a),
                            P["pre_max_s"] * FPS))
                fill = np.isnan(out[b - m:b, 0])
                out[b - m:b][fill] = clean[q]
        if before_restart and u[3] < P["loss_after_p"]:
            m = int(np.interp(span[3], [0, 1], P["loss_after_s"]) * FPS)
            remove[b:b + m] |= live[b:b + m]
    # interior live gaps (aerial balls, occlusions)
    live_idx = np.flatnonzero(live)
    n_gaps = rng.poisson(P["gap_per_min"] * len(live_idx) / (60.0 * FPS))
    gap_at = rng.choice(live_idx, size=n_gaps) if n_gaps else np.zeros(0, dtype=np.int64)
    gap_len = _lognormal_frames(rng, n_gaps, P["gap_median_s"], P["gap_sigma"], P["gap_clip_s"])
    for s0, m in zip(gap_at, gap_len):
        remove[s0:s0 + m] = True
    out[remove] = np.nan
    out = _interp_short_gaps(out, halves, int(P["min_gap_s"] * FPS))
    synth = ~np.isnan(out[:, 0]) & (np.isnan(clean[:, 0]) | remove)   # not a clean observation
    out += _smooth_noise(rng, out.shape, P["ball_noise_m"], P["noise_sigma_frames"])
    return out, synth


def _degrade_players(players, team, rng, P):
    """Dropouts, ghost slots (duplicates and team flips) and smooth noise.
    Returns (players (n, S + 2G, 2), team (S + 2G,)): G extra slots per team carry the ghosts."""
    n, S, _ = players.shape
    present = ~np.isnan(players[:, :, 0])
    drop = np.zeros((n, S), dtype=bool)
    for s in range(S):                               # independent per-slot dropouts
        rows = np.flatnonzero(present[:, s])
        k = rng.poisson(P["slot_drop_per_min"] * len(rows) / (60.0 * FPS)) if len(rows) else 0
        at = rng.choice(rows, size=k) if k else np.zeros(0, dtype=np.int64)
        for s0, m in zip(at, _lognormal_frames(rng, k, P["slot_drop_median_s"],
                                               P["slot_drop_sigma"], P["slot_drop_clip_s"])):
            drop[s0:s0 + m, s] = True
    k = rng.poisson(P["bad_per_min"] * n / (60.0 * FPS))  # common-mode bad windows
    at = rng.integers(0, n, size=k)
    lens = _lognormal_frames(rng, k, P["bad_median_s"], P["bad_sigma"], P["bad_clip_s"])
    jit = int(P["bad_jitter_s"] * FPS)
    for s0, m in zip(at, lens):
        hit = rng.random(S) < P["bad_q"]
        j = rng.integers(-jit, jit + 1, size=(S, 2))
        for s in np.flatnonzero(hit):
            drop[max(s0 + j[s, 0], 0):max(s0 + m + j[s, 1], 0), s] = True
    out = players.astype(np.float64)
    out[drop] = np.nan
    G = int(P["ghost_slots_per_team"])               # ghosts: BAR-ATH has a 12th object in a team
    ghost = np.full((n, 2 * G, 2), np.nan)           # on ~6% of frames and > 22 objects on ~2%
    for t in (0, 1):
        k = rng.poisson(P["ghost_per_min"] * n / (60.0 * FPS))
        at = np.sort(rng.integers(0, n, size=k))
        lens = _lognormal_frames(rng, k, P["ghost_median_s"], P["ghost_sigma"], P["ghost_clip_s"])
        flips = rng.random(k) < P["ghost_flip_share"]
        offsets = rng.normal(0.0, P["ghost_offset_m"], size=(k, 2))
        picks = rng.random(k)
        free_at = np.zeros(G, dtype=np.int64)
        for s0, m, flip, off, u in zip(at, lens, flips, offsets, picks):
            src = np.flatnonzero((team == (1 - t if flip else t)) & ~np.isnan(out[s0, :, 0]))
            slots = np.flatnonzero(free_at <= s0)
            if not len(src) or not len(slots):
                continue
            p, g = src[int(u * len(src))], t * G + slots[0]
            ghost[s0:s0 + m, g] = out[s0:s0 + m, p] + (0.0 if flip else off)
            if flip:                                 # the player is seen as the other team
                out[s0:s0 + m, p] = np.nan
            free_at[slots[0]] = s0 + m
    out = np.concatenate([out, ghost], axis=1)
    out += _smooth_noise(rng, out.shape, P["player_noise_m"], P["noise_sigma_frames"])
    return out, np.concatenate([team, np.repeat(np.array([0, 1], dtype=np.int8), G)])


DEGRADE_TARGETS = dict(ball_present=0.59, miss_live=0.09, miss_dead=0.67, gap_2_10=0.15,
                       gap_10_30=0.41, gap_30=0.44, players_median=22, players_lt18=0.09,
                       team_gt11=0.064, players_gt22=0.024, players_mean=20.5,
                       player_noise_m="0.1-0.2", ball_noise_m=0.3)


def degradation_stats(data, clean=None):
    """Input-quality statistics over graded frames, comparable with DEGRADE_TARGETS. `clean` (the
    degrade=False dict of the same game) adds the positional-noise figures."""
    lab = data["labels"]
    graded, dead = lab != "X", lab == "D"
    live = (lab == "H") | (lab == "A")
    miss = np.isnan(data["ball"][:, 0])
    lens = []
    for _, f0, f1 in data["halves"]:
        a, b = _runs(miss[f0:f1 + 1])
        lens += list(b - a)
    lens = np.array(lens, dtype=np.int64)
    tot = max(lens.sum(), 1)
    share = lambda lo, hi: float(lens[(lens >= lo * FPS) & (lens < hi * FPS)].sum()) / tot  # noqa
    seen = ~np.isnan(data["players"][:, :, 0])
    n_players = seen.sum(axis=1)[graded]
    team = np.asarray(data["team"])
    per_team = [seen[:, team == t].sum(axis=1)[graded] for t in (0, 1)]
    out = dict(ball_present=float(1 - miss[graded].mean()), miss_live=float(miss[live].mean()),
               miss_dead=float(miss[dead].mean()), gap_lt2=share(0, 2), gap_2_10=share(2, 10),
               gap_10_30=share(10, 30), gap_30=share(30, 1e9),
               n_gaps_lt3s=int((lens < 3 * FPS).sum()), n_gaps=int(len(lens)),
               players_median=float(np.median(n_players)), players_mean=float(n_players.mean()),
               players_lt18=float((n_players < 18).mean()),
               team_gt11=float(np.mean([(c > 11).mean() for c in per_team])),
               players_gt22=float((n_players > 22).mean()))
    if clean is not None:
        S = clean["players"].shape[1]                # ghost slots have no clean counterpart

        def noise(a, b):
            d = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1)
            step = np.abs(np.diff(d, axis=0))
            return float(np.sqrt(np.nanmean(d ** 2))), float(np.nanmax(step))
        both = ~np.isnan(data["ball"][:, 0]) & ~np.isnan(clean["ball"][:, 0])
        synth = data.get("ball_synthetic", np.zeros(len(both), dtype=bool))
        out["ball_noise_rms_m"], _ = noise(data["ball"][both & ~synth], clean["ball"][both & ~synth])
        out["ball_synthetic_share"] = float(synth[graded].sum()) / max(int((~miss[graded]).sum()), 1)
        out["ball_seen_dead_synthetic_share"] = float((synth & dead).sum()) / max(
            int((~miss & dead).sum()), 1)
        out["player_noise_rms_m"], out["player_noise_max_step_m"] = noise(data["players"][:, :S],
                                                                          clean["players"])
        drop = np.isnan(data["players"][:, :S, 0]) & ~np.isnan(clean["players"][:, :, 0])
        dl = []
        for s in range(drop.shape[1]):
            a, b = _runs(drop[:, s])
            dl += list(b - a)
        out["player_drop_median_s"] = float(np.median(dl)) / FPS if dl else 0.0
    return out


# ----------------------------------------------------------------------------- public API
INPUT_KEYS =("players", "ball", "team", "gk", "n_frames", "pitch", "halves",
              "home_attacks_positive_x_first_half", "fps", "game_id")


def load_game(game_id, degrade=True, seed=0, rule="control", use_cache=True):
    """One Metrica sample game in the in-memory format of solution/atd.py, plus labels.

    Returns dict: players (n,S,2) float32 m, ball (n,2) float32 m (NaN = missing), team (S,) int8
    0=home 1=away, gk (S,) bool, n_frames, pitch (105, 68), halves [(half, first, last)] inclusive,
    home_attacks_positive_x_first_half, fps, labels (n,) '<U1' H/A/D in halves and X outside,
    plus game_id, degraded, seed, rule, label_stats, ball_synthetic (diagnostics only). Only
    INPUT_KEYS are meant for a labeller. `degrade` may also be a dict overriding DEGRADE entries."""
    trk = _load_tracking(game_id, use_cache)
    ev = load_events(game_id)
    n = len(trk["ball"])
    halves = _halves(ev, trk["period"])
    labels, stats = derive_labels(ev, n, halves, trk["ball"], rule=rule)
    team = trk["team"].astype(np.int8)
    players, ball = trk["players"], trk["ball"]
    gk = _gk_slots(players, team, halves)
    kick_off = halves[0][1]
    home_x = np.nanmean(players[kick_off, team == 0, 0])
    synth = np.zeros(n, dtype=bool)
    if degrade:
        P = dict(DEGRADE, **degrade) if isinstance(degrade, dict) else DEGRADE
        rng = np.random.default_rng([int(seed), int(game_id)])
        ball, synth = _degrade_ball(ball, labels, halves, rng, P)
        players, team = _degrade_players(players, team, rng, P)
        gk = np.concatenate([gk, np.zeros(len(team) - len(gk), dtype=bool)])
    return dict(players=np.asarray(players, dtype=np.float32), ball=np.asarray(ball, dtype=np.float32),
                team=team, gk=gk, n_frames=n, pitch=PITCH, halves=halves,
                home_attacks_positive_x_first_half=bool(home_x < PITCH[0] / 2), fps=FPS,
                labels=labels, game_id=game_id, degraded=bool(degrade), seed=seed, rule=rule,
                label_stats=stats, ball_synthetic=synth)


def score(pred_labels, data):
    """Official metrics (evaluation/possession_metrics.py) of a prediction against data['labels'].
    X in the prediction is forward-filled within each half for the temporal metrics, as on the
    board."""
    pred = np.asarray(pred_labels)
    if len(pred) != data["n_frames"]:
        raise ValueError("prediction has {} frames, game has {}".format(len(pred), data["n_frames"]))
    r = pm.evaluate(pm.label_codes(data["labels"]), pm.label_codes(pred))
    return dict(spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"],
                spell_f1_iou50=r["spell_f1_at_headline"], n_pred_spells=r["n_pred_spells"],
                n_gt_spells=r["n_gt_spells"], mean_spell_s=r["mean_spell_duration_s"],
                gt_mean_spell_s=r["gt_mean_spell_duration_s"], alive_dead_f1=r["alive_dead_f1"],
                team_acc_both_alive=r["team_acc_both_alive"],
                f1_H=r["per_class"]["H"]["f1"], f1_A=r["per_class"]["A"]["f1"],
                f1_D=r["per_class"]["D"]["f1"], n_abstain=r["n_abstain_on_graded"])


SUMMARY_KEYS = ("spell_f1", "macro_f1", "spell_f1_iou50", "n_pred_spells", "n_gt_spells",
                "alive_dead_f1", "team_acc_both_alive")


def evaluate(label_fn, games=(1, 2, 3), degrade=True, seed=0, rule="control", verbose=True):
    """Run label_fn(data) -> labels (n_frames of 'H','A','D','X') on each game and score it.

    label_fn only sees INPUT_KEYS (never the labels). `seed` may be an int or a sequence of ints;
    with several seeds every game is scored once per degradation seed and the scores averaged.
    Returns {game: {metric: value}, ..., 'mean': {metric: mean over games}}."""
    seeds = [seed] if np.isscalar(seed) else list(seed)
    results = {}
    if verbose:
        print("{:<6} {:>8} {:>8} {:>9} {:>6} {:>6} {:>8} {:>8} {:>7}".format(
            "game", "spellF1", "macroF1", "spell@.5", "pred", "true", "aliveF1", "teamAcc", "sec"))
    for g in games:
        per_seed = []
        for sd in seeds:
            data = load_game(g, degrade=degrade, seed=sd, rule=rule)
            t0 = time.time()
            pred = label_fn({k: data[k] for k in INPUT_KEYS})
            res = score(pred, data)
            res["seconds"] = time.time() - t0
            per_seed.append(res)
        res = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
        results[g] = res
        if verbose:
            print("{:<6} {:>8.4f} {:>8.4f} {:>9.4f} {:>6.0f} {:>6.0f} {:>8.4f} {:>8.4f} {:>7.1f}".format(
                g, res["spell_f1"], res["macro_f1"], res["spell_f1_iou50"], res["n_pred_spells"],
                res["n_gt_spells"], res["alive_dead_f1"], res["team_acc_both_alive"],
                res["seconds"]))
    results["mean"] = {k: float(np.mean([results[g][k] for g in games])) for k in SUMMARY_KEYS}
    if verbose:
        m = results["mean"]
        print("{:<6} {:>8.4f} {:>8.4f} {:>9.4f} {:>6.0f} {:>6.0f} {:>8.4f} {:>8.4f}".format(
            "mean", m["spell_f1"], m["macro_f1"], m["spell_f1_iou50"], m["n_pred_spells"],
            m["n_gt_spells"], m["alive_dead_f1"], m["team_acc_both_alive"]))
    return results


# ----------------------------------------------------------------------------- baseline
def _majority(codes, window):
    """starter/baseline.py's smooth(): centred +/-window majority of H/A, D where no ball at all."""
    if window <= 0:
        return np.where(codes == 0, "H", np.where(codes == 1, "A", "D"))
    home = np.concatenate(([0], np.cumsum(codes == 0)))
    away = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    labels[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return labels


def baseline_labels(data, window=50):
    """Replica of starter/baseline.py: team of the nearest tracked player to the ball, D where the
    ball is missing, then a centred +/-window majority vote. X outside the halves."""
    players = np.asarray(data["players"], dtype=np.float64)
    ball = np.asarray(data["ball"], dtype=np.float64)
    d2 = ((players - ball[:, None, :]) ** 2).sum(axis=2)
    d2 = np.where(np.isnan(d2), np.inf, d2)
    best = np.argmin(d2, axis=1)
    seen = np.isfinite(d2[np.arange(len(d2)), best])
    codes = np.full(len(ball), -1, dtype=np.int8)
    codes[seen] = np.asarray(data["team"])[best[seen]]
    labels = _majority(codes, window)
    out = np.full(len(labels), "X", dtype="<U1")
    for _, f0, f1 in data["halves"]:
        out[f0:f1 + 1] = labels[f0:f1 + 1]
    return out


# ----------------------------------------------------------------------------- reports / CLI
def label_summary(data):
    """Possession statistics of data['labels'] (graded frames only)."""
    codes = pm.label_codes(data["labels"])
    runs = pm.valid_runs(codes)
    lengths = np.array([b - a for a, b, _ in pm.possession_spells(codes, runs)])
    graded = codes != pm.X
    alive = (codes == pm.H) | (codes == pm.A)
    return dict(n_possessions=int(len(lengths)), mean_possession_s=float(lengths.mean()) / FPS,
                median_possession_s=float(np.median(lengths)) / FPS,
                possessions_lt1s=float(np.mean(lengths < FPS)),
                d_share=float(np.mean(codes[graded] == pm.D)),
                n_dead_stretches=sum(1 for s in pm.segments(codes, runs) if s[2] == pm.D),
                home_share=float(np.sum(codes == pm.H)) / max(int(alive.sum()), 1),
                graded_frames=int(graded.sum()))


def dead_alignment(clean):
    """Event-derived D against Metrica's own corrected tracking, which blanks the ball while it is
    dead: share of D / live frames with the ball missing, and frame offsets of the blanking onset
    (vs our stoppage start) and of the ball's reappearance (vs our restart frame)."""
    lab = clean["labels"]
    miss = np.isnan(clean["ball"][:, 0])
    dead, live = lab == "D", (lab == "H") | (lab == "A")
    onset, back = [], []
    for a, b in zip(*_runs(dead)):
        idx = np.flatnonzero(miss[a:b])
        if not len(idx):
            continue
        onset.append(int(idx[0]))
        if b < len(lab) and live[b]:
            r = b
            if miss[b - 1]:
                while r < len(lab) and miss[r]:
                    r += 1
            else:
                r = a + int(idx[-1]) + 1
            back.append(r - b)
    return dict(dead_ball_missing=float(miss[dead].mean()), live_ball_missing=float(miss[live].mean()),
                onset_median=float(np.median(onset)), onset_p90=float(np.percentile(onset, 90)),
                reappear_median=float(np.median(back)), reappear_abs_p90=float(np.percentile(np.abs(back), 90)))


def _fmt(d):
    return "  ".join("{}={}".format(k, round(v, 3) if isinstance(v, float) else v) for k, v in d.items())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, nargs="+", default=list(GAMES))
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--rule", default="control", choices=("control", "touch"))
    args = ap.parse_args(argv)
    games = tuple(args.games)
    print("Metrica Sports public sample data (github.com/metrica-sports/sample-data), games", games)

    print("\n== labels from events (rule={}) ==".format(args.rule))
    for g in games:
        t0 = time.time()
        clean = load_game(g, degrade=False, rule=args.rule)
        load_s = time.time() - t0
        other = "touch" if args.rule == "control" else "control"
        alt = label_summary(load_game(g, degrade=False, rule=other))
        print("game {} (n={}, halves={}, load {:.1f}s)".format(g, clean["n_frames"], clean["halves"], load_s))
        print("   ", _fmt(label_summary(clean)))
        print("    rule={}: n_possessions={} mean_possession_s={:.1f}".format(
            other, alt["n_possessions"], alt["mean_possession_s"]))
        print("    derivation:", _fmt(clean["label_stats"]))
        print("    vs Metrica's own corrected ball blanking:", _fmt(dead_alignment(clean)))

    print("\n== degradation (seed {}) vs BAR-ATH targets ==".format(args.seed[0]))
    print("    targets:", _fmt(DEGRADE_TARGETS))
    for g in games:
        t0 = time.time()
        deg = load_game(g, degrade=True, seed=args.seed[0], rule=args.rule)
        load_s = time.time() - t0
        st = degradation_stats(deg, load_game(g, degrade=False, rule=args.rule))
        print("game {} (load {:.1f}s): {}".format(g, load_s, _fmt(st)))

    for name, degrade in (("clean", False), ("degraded", True)):
        print("\n== baseline (nearest player, D where no ball, +/-50 majority), {} ==".format(name))
        evaluate(baseline_labels, games=games, degrade=degrade,
                 seed=args.seed if degrade else 0, rule=args.rule)
    print("\n(BAR-ATH full match, same baseline: spellF1 0.2433, macroF1 0.6724)")


if __name__ == "__main__":
    main()
