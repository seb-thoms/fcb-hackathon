# Solution proposal — More than a Hack 2027, Phase 1: Ball Possession

> **Status:** draft for team review · **Deadline:** November 17, 2026 · **Repo:** <https://github.com/seb-thoms/fcb-hackathon>
>
> Edit freely. Open questions for the team are collected in [§10](#10-open-questions-for-the-team).
> Numbers quoted from our own exploration come from `python explore_data.py`, which prints
> aggregates only.

## Contents

1. [The problem in one page](#1-the-problem-in-one-page)
2. [What counts as possession](#2-what-counts-as-possession)
3. [The data](#3-the-data)
4. [How we are graded](#4-how-we-are-graded)
5. [Rules that constrain the design](#5-rules-that-constrain-the-design)
6. [Why the baseline scores 0.24 — and where the points are](#6-why-the-baseline-scores-024--and-where-the-points-are)
7. [Proposal A — rule-based possession state machine](#7-proposal-a--rule-based-possession-state-machine)
8. [Proposal B — self-trained ML, no external data](#8-proposal-b--self-trained-ml-no-external-data)
9. [Discussion and recommendation](#9-discussion-and-recommendation)
10. [Open questions for the team](#10-open-questions-for-the-team)

---

## 1. The problem in one page

**Organisers:** Metrica Sports, FC Barcelona (Barça Innovation Hub) and Mobile World Capital
Barcelona.

**Match:** FC Barcelona (home, `H`) vs Athletic Club (away, `A`), LaLiga 2024-25, matchday 2.

**Task:** for **every frame** of the tactical match video — 199,319 frames at 25 fps, about 2 h 13 min
of footage of which ~106 min is the actual match — output one label:

| label | meaning |
|---|---|
| `H` | Barcelona in possession, ball in play |
| `A` | Athletic in possession, ball in play |
| `D` | dead ball — not in play (throw-in setup, goal kick setup, foul, goal celebration…) |
| `X` | no answer (fine on ungraded frames; see §4 for why we should not use it on graded ones) |

**Deliverable that is scored:** one CSV, header `frame,label`, exactly 199,319 rows, `frame` = 0…199318.

```
frame,label
0,X
...
11185,H
11186,H
...
```

**How the winner is decided:** an automatic script compares our file to a **private** frame-by-frame
ground truth for the full match. Ranked on `spellF1` (are whole possessions recovered?), ties
broken by `macroF1` (per-frame correctness). No jury, no presentation.

**What we must beat:** the published baseline, `spellF1 = 0.2433` on the full match.

**Also required (not scored, but needed for the result to count):**

1. a **public code repository** that regenerates the *exact same* CSV from the provided data;
2. a **short technical document** (PDF or Markdown): approach, assets used/dropped, how missing
   tracking was handled, possession rules implemented, limitations, how to reproduce.

Submission is one Google Form per team: `<team_name>.csv`, repo URL, `<team_name>.pdf`.
There is **no live leaderboard** — we get no feedback before results.

**Why it is hard:** the tracking is automatic and uncorrected. The ball is missing on **41% of
graded frames**, players have no identities, and there is no "ball in play" or "who has the ball"
field in the data. We must infer both from movement.

---

## 2. What counts as possession

From `CHALLENGE.md` — this is what our rules must reproduce:

- **While the ball is live, exactly one team has possession.** There is no "contested" or
  "unknown" class in the ground truth.
- **Possession changes only when the other team gains *controlled* possession.** Not a change:
  a touch without control, a deflection that does not intercept the pass, an aerial duel won
  without subsequent control. Suggested proxies: a player keeps the ball a few seconds; a team
  completes more than two consecutive passes.
- **Loose / contested ball stays with the last team that had it.**
- **After every restart there is a clear possessor.**
- **A possession is a maximal run of one team with the ball alive.** `H H H A A` = 2 possessions.
  `H H D D H H` = **2 possessions** (the stoppage splits it), not one.
- **Dead ball is a stoppage, never a possession**, and we never say who "owns" a dead ball.
- **The ground truth was derived mechanically from *corrected* tracking data** (not video, not event
  data): `D` where corrected tracking marks the ball dead, otherwise the team it names as owner.
  No smoothing afterwards. So "dead" and "owning" mean whatever that corrected tracking means —
  the public 5 minutes show it.

---

## 3. The data

Everything lives in `data/` (gitignored, **never commit or share it**). We use the **tactical**
recording; it is the reference timeline (row *i* of the submission = tactical frame *i*).

### 3.1 Assets

| file | size | what it is |
|---|---|---|
| `tactical/BAR-ATH_tactical.mp4` | 5.2 GB | wide, high camera; 1920×1080, 25 fps, 199,319 frames, no cuts/replays |
| `tactical/BAR-ATH_tactical_atd.txt` | 196 MB | **automatic tracking data (ATD)** — players and ball, one line per frame |
| `tactical/BAR-ATH_tactical_atd.xml` | 59 KB | ATD metadata: teams, track slots, attack direction, units |
| `tactical/BAR-ATH_tactical_homography.csv` | 23 MB | one 3×3 pitch→pixel matrix per frame |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` | 261 KB | 638 automatic tactical phase / set-piece tags |
| `broadcast/*` | 14 GB video + ATD + homography | TV feed, **not synchronised** to tactical. Proposed: **not used** |
| `evaluation/gt/BAR-ATH_possession_gt_public.csv` | in repo | ground truth for the first 5 minutes only |

### 3.2 Tracking (ATD) — structure

Each line of the `.txt` is one frame:

```
<frame_id> : <slot1>;<slot2>;...;<slot46> : <ball>
```

- `frame_id` runs 1…199319, contiguous. **Submission row = `frame_id − 1`.**
- Each slot and the ball is `x,y,speed`; `NaN` when not observed. All 46 slots are present on every
  line, most of them `NaN` at any given moment.
- `x, y` are **normalised 0–1**. Multiply by **105.3 × 68.0 m** (the pitch size in `games.py`) for metres.
  `x` runs along the length of the pitch.
- **Player speed is in m/s** (the metadata says "kph" — it is wrong; it matches position deltas in m/s).
- **The ball speed channel is unusable** — filled on only 8.8% of graded frames and uncorrelated with
  the ball's motion. Derive ball velocity from positions.
- **No ball height, no ball-state flag, no ball-owner field.**

**Track slots are not players.** 46 slots (Barcelona 24, Athletic 22) for 22 players: the tracker opens a
new slot whenever it loses someone and cannot re-link them. Each slot belongs to one team. Two slots
are tagged `position_type = Goalkeeper`; the rest are `Unknown`.

| observation (graded frames) | value |
|---|---|
| active slots per frame | median 22; < 18 on 9.2% of frames; > 22 on 2.4% (duplicates) |
| tracklet length | median 35 s; 6.6% shorter than 2 s |
| teleports (> 1 m between frames) | 0.009% — positions are clean |
| player speed | median 1.4 m/s, p99 6.4, max 9.4 m/s |
| **ball present** | **59.3%** (40.7% missing in *each* half) |
| **ball gap length** | **median 268 frames ≈ 11 s**; 19 gaps > 30 s carry 44% of all missing frames |
| ball outside 0–1 | x 0.6%, y 3.6% of ball-present frames (ball out of play candidates) |

The most important finding: **ball gaps are long blackouts, not flickers.** We cannot interpolate our
way through them; for long stretches possession must be inferred from **team shape and movement**.

**Orientation.** Metadata says Barcelona attacks **left→right in the 1st half**; confirmed by the
data (Barcelona goalkeeper slot sits at mean x ≈ 0.10 in the 1st half, ≈ 0.89 in the 2nd).
**Coordinates are fixed to the pitch; teams swap ends at halftime.**

### 3.3 Smart Tagging — structure

SportsCode XML, 638 instances, each `{code, start, end, labels}`. `start/end` are seconds from video
frame 0, so `frame = round(seconds × 25)` (verified against both kickoffs). Loader:
`starter/load_smart_tagging.py`.

| code | n | median length | notes |
|---|---|---|---|
| players in the box | 97 | 28 s | |
| ball in final third | 93 | 27 s | |
| ball in the box | 72 | 17 s | |
| progression | 66 | 18 s | mirror of *defending in middle third* |
| defending in middle third | 66 | 18 s | |
| set pieces | 59 | 35 s (mostly fixed) | `Type`: throw-in 26, corner 12, goal kick 10, free kick 5, kick-off 6 |
| creating chances | 30 | **always 30 s** | mirror of *defending in defensive third* |
| defending in defensive third | 30 | 30 s | |
| long ball | 29 | 12 s | |
| attacking / defensive transition | 24 / 24 | 15 s | mirrors of each other |
| build up | 22 | 21 s | mirror of *defending in attacking third* |
| defending in attacking third | 22 | 21 s | |
| goals | 4 | 169 s | |

What we learned:

- **Defending tags are exact mirrors of attacking tags** (same counts, same intervals, team always
  the opponent). They carry no extra information.
- `Team` is `Barcelona`, `Atletic Club` or `N/A`. **All 26 throw-ins and all 6 kick-offs are `N/A`.**
- Tags cover 82% of graded frames; on 9% of frames attacking tags for *both* teams overlap.
- Some interval lengths are fixed (30 s, 35 s) — **tag edges are not possession boundaries**.
- **It is a weak prior:** on the public 5 minutes the attacking-tag team agreed with the truth on
  only ~53% of `H` frames and ~41% of `A` frames.
- Around set-piece starts the ball is mostly untracked (corners and kick-offs: 100% missing in ±2 s).

### 3.4 Homography

One row per frame: 9 matrix values, `reprojection_error`, `valid`. Maps normalised pitch (0–1) to
image pixels; valid on 100% of tactical frames. **Only needed for anything touching the video**
(e.g. our `viewer.py`, or our own ball detector). Note it maps the *ground plane*: an airborne ball
does not project correctly.

### 3.5 Frame alignment and graded frames

| source | index | → submission row |
|---|---|---|
| submission | 0…199318 | — |
| ATD | `frame_id` 1…199319 | `frame_id − 1` |
| Smart Tagging | seconds | `round(s × 25)` |
| homography | `frame` 0…199318 | same |

| span | video frames | graded? |
|---|---|---|
| before kickoff | 0 – 11,184 | no |
| **1st half** | **11,185 – 90,651** | **yes** (starts at kickoff) |
| halftime | 90,652 – 110,807 | no |
| **2nd half** | **110,808 – 190,197** | **yes** — but kickoff is at 113,742: the first 2,934 graded frames are all `D` |
| after final whistle | 190,198 – 199,318 | no |

`games.half_video_blocks("BAR-ATH")` returns the graded blocks. 158,857 graded frames in total.

### 3.6 Public ground truth

`evaluation/gt/BAR-ATH_possession_gt_public.csv` — 199,319 rows, real labels only on frames
**11,185–18,684** (verified: contiguous, nothing outside). `H` 3,454 · `A` 2,577 · `D` 1,469;
21 segments, **17 possessions, mean 14.2 s**; 4 dead-ball stretches of 7–20 s.
**For checking the pipeline only — not for training or tuning** (see §5).

### 3.7 Tools already in the repo

| tool | purpose |
|---|---|
| `python explore_data.py` | aggregate-only data profile (safe to share) |
| `python viewer.py` | local viewer: video + tracking overlay, 2D pitch, label/tag timeline |
| `python starter/baseline.py` | regenerate the four baselines |
| `python evaluation/run_evaluation.py` | score every CSV in `evaluation/results/` on the public window |
| `python evaluation/validate_submission.py <file>` | the official format checks — run on the exact file we upload |

---

## 4. How we are graded

The definitions live in `evaluation/possession_metrics.py`; that file wins over any doc.

### 4.1 Preprocessing done by the scorer

1. Only frames where the ground truth is not `X` are graded — the two halves. Everything temporal
   is computed **inside each half separately** (halftime cannot create a fake transition).
2. For the temporal metrics, every `X` in our file is **forward-filled with our last non-`X` label in
   that half** (back-filled at the start of a half).

### 4.2 `spellF1` — the ranking metric

1. A **possession spell** = a maximal run of `H` or of `A` inside a half. `D` runs are not spells,
   they only separate them.
2. Every predicted spell is compared with every true spell **of the same team**, by temporal IoU:
   `IoU = overlap / (union of the two intervals)`.
3. Greedy **one-to-one** matching by descending IoU: each true spell can be credited to at most one
   predicted spell and vice versa.
4. At each threshold t ∈ {0.3, 0.4, 0.5, 0.6, 0.7}: TP = matched pairs with IoU ≥ t,
   FP = unmatched predicted spells, FN = unmatched true spells → F1.
5. `spellF1` = mean of the five F1s. Range [0, 1]; any constant guess scores 0.

Sensitivity published by the organisers:

| perturbation of a perfect answer | spellF1 |
|---|---|
| 1 frame late | 1.0000 |
| 3 / 8 frames late | 0.9818 / 0.8933 |
| 1 s late | 0.6539 |

**What this means in practice:**

- **Every spurious split costs twice.** One true possession `HHHHHHHH` predicted as `HHHAHHHH`
  becomes three predicted spells: at best one of them matches (and maybe none reaches IoU 0.5),
  and the others are false positives. Flicker is catastrophic.
- **A spurious `D` also splits a possession.** `HHHDDHHH` is two `H` spells. So outputting `D` when
  the ball merely disappears (what the baseline does) destroys possessions.
- **Missing a real `D` merges two possessions** (e.g. Barcelona keeps the ball after a throw-in:
  `H…H D…D H…H` is two spells; predicting `H` throughout gives one long spell that matches neither
  well). Dead-ball boundaries matter as much as team changes.
- **Boundaries need to be within a few frames**, but IoU 0.3–0.7 tolerates some slack on long spells;
  short spells are fragile.
- Missing a short true possession costs one FN; inventing one costs one FP. Precision and recall are
  symmetric, so **the predicted number of spells should be close to the true one**.
- **Noise level:** differences below ~0.04 on the full match are noise.

### 4.3 `macroF1` — the tie-break

Per-frame F1 for each of `H`, `A`, `D`, averaged unweighted. `X` on a graded frame counts as wrong.
Floor ≈ 0.16 (constant guess). Differences below ~0.01 are noise.

### 4.4 Should we ever write `X`?

On graded frames, **no**. Under `spellF1`, `X` = "repeat my previous label", which we can just as
well write ourselves; under `macroF1` `X` is always wrong whereas a guess is sometimes right. So `X`
only ever loses. **Write `X` on ungraded frames only** (before kickoff, halftime, after the whistle)
— it costs nothing there.

### 4.5 The public window cannot pick between versions

Five minutes = 17 possessions; one possession ≈ 6 points of `spellF1`. The organisers show that
tuning even the baseline's smoothing window on it **picks the wrong value** for the full match:

| baseline | public spellF1 | full-match spellF1 |
|---|---|---|
| smooth50 | 0.3256 | **0.2433** ← better on the full match |
| smooth125 | **0.4242** ← looks better | 0.2242 |

We therefore use the public window **only for sanity** (alignment, label meaning, "is it badly
broken?"), with `--bootstrap 1000` when comparing. Choices between versions must come from
label-free checks (§9.3).

---

## 5. Rules that constrain the design

- **Labels must come from an algorithm.** No hand labelling, no hand-correcting output, no rules or
  constants tied to specific frames of this match.
- **No hand labels on this match for any purpose** — not to train, not to validate. Labels from
  other matches are allowed, but we have decided **not to use external data**.
- **Self-supervision / pseudo-labels are allowed** if produced by an algorithm with no person
  deciding any of them. This is what Proposal B relies on.
- **No private ground truth; no human-produced data about this match** (Opta/StatsBomb/Wyscout
  events, corrected tracking…).
- **Public 5 minutes: pipeline checks only, not training, not tuning.**
- **Watching the video is allowed** for understanding and debugging; turning what we see into labels
  or frame-specific rules is not.
- **No match data (video, files, extracts) to any cloud-hosted LLM.** Code and aggregate stats only.
- **Reproducibility:** our public repo must regenerate the *identical* CSV → fixed seeds, pinned
  dependencies, deterministic code, one command.
- **Data must not be in the repo** (`data/` and result CSVs are already gitignored).
- Team of 3–5 members.

---

## 6. Why the baseline scores 0.24 — and where the points are

The baseline (`starter/baseline.py`):

1. per frame, the team of the tracked player **nearest the ball**;
2. **`D` whenever the ball is not tracked**;
3. majority vote over ±50 frames.

Its failure modes map directly to the score:

| baseline behaviour | effect on spellF1 | our fix |
|---|---|---|
| `D` whenever the ball is missing (41% of frames, in ~11 s blocks) | real possessions split into pieces by fake stoppages; real `D` only by coincidence | dedicated **dead-ball detector**; through ball gaps, carry possession or infer it from team shape |
| nearest player = possessor | every pass in flight, deflection and 50/50 flips the label | possession changes only on **sustained control** by the other team |
| symmetric majority vote | boundaries smeared by up to the window size; short possessions erased | event-based boundaries (control onset, restart moment) + sticky temporal decoding |
| no notion of restarts | possession after a stoppage is a guess | restart logic: which team takes the throw-in / corner / goal kick |

Order of expected gains: **(1) stop emitting `D` for missing ball → (2) real dead-ball detection → (3)
control-based team changes → (4) team-shape inference inside ball blackouts → (5) restart team.**

---

## 7. Proposal A — rule-based possession state machine

**Idea:** encode the possession definition directly as football logic over the tracking, then decode
with a sticky three-state model so the output is a clean sequence of spells. No learning; every
parameter is a football-meaningful quantity chosen *a priori*.

### 7.1 Pipeline

```
ATD ──► 1. preprocess ──► 2. per-frame evidence ──► 3. events ──► 4. state machine / HMM ──► 5. post-process ──► CSV
           (metres,          (control, out-of-play,     (control onsets,   (H / A / D with         (X on ungraded,
            attack dir,       team shape, stillness,     restarts, gaps)    sticky transitions)     validation)
            ball velocity)    Smart Tagging priors)
```

### 7.2 Step 1 — preprocessing

- Convert to metres; derive ball velocity from positions (ignore the speed channel); light smoothing.
- **Interpolate only short ball gaps** (≤ ~0.4 s); long gaps stay missing and are handled explicitly.
- **Attack-normalised coordinates:** per half, flip x so each team attacks towards +x
  (from `attack_direction_first_half`, swapped in the 2nd half).
- Per frame, per team: centroid, length/width/spread, mean speed, forward velocity, defensive-line
  height, players per third, distance of each team's nearest player to the ball.

### 7.3 Step 2 — per-frame evidence

**Control (ball visible):** a player has control at frame t if within **r ≈ 1.0–1.5 m** of the ball,
the ball's velocity matches the player's (dribble / reception, not a pass flying past), and no
opponent is equally close. **Controlled possession** for a team = control for **≥ ~0.5–1 s**, or two
consecutive touches by the same team with no opponent touch between (≈ the "two passes" proxy).

**Dead ball candidates:**

- ball leaves the pitch (x or y outside [0, 1] beyond a small margin) → `D` from that frame;
- long stillness: ball static and most players slow (in the public window `D` frames have median
  player speed 1.2 m/s vs ~2 m/s in play);
- restart signatures: a player stationary at the touchline / corner arc / six-yard box with the ball,
  players arranged for a kick-off (each team in its own half);
- Smart Tagging `SET PIECES` and `GOALS` intervals as supporting evidence (the restart lies near the
  tag start; the stoppage just before it);
- 2nd half: `D` until the kick-off is detected (formation + ball at the centre spot / tag).

**Restart (D → live):** the ball starts moving again from a dead-ball location; the possessing team
is the taker: throw-in → the team of the player at the touchline; corner → attacking team; goal kick
→ goalkeeper's team; kick-off → the team at the centre spot; free kick → nearest player when it moves.

**Team-shape prior (ball invisible):** in possession a team stretches (larger width/length), pushes its
line up and moves forward; without it, the team compacts and the nearest defenders converge on the
ball area. Combined into a soft score `P(H | shape)`. Smart Tagging attacking-team flags add a weak
prior (≈ 50% agreement — low weight).

### 7.4 Step 3/4 — the possession state machine

States `H`, `A`, `D`. Rules applied in time order inside each half:

1. Start of half → `D` until kick-off; kick-off → kicking team.
2. Live, team T in possession → **stays T** through loose balls, deflections, aerial duels and ball
   blackouts, **unless**:
   - the other team achieves *controlled possession* → switch at the **onset** of that control
     (the first touch of the controlling sequence — boundaries need to be within a few frames);
   - dead-ball evidence fires → `D` from the out-of-play moment.
3. `D` → stays `D` until a restart is detected → restart team.
4. **Ball blackouts:** if the ball reappears with the same team in control → fill the gap with that
   team. If it reappears with the other team → place the switch at the most likely change point
   inside the gap using the team-shape score (not simply at the end of the gap). If stillness +
   set-piece evidence → `D` stretch inside the gap.

Implementation option (recommended): express the same logic as a **3-state HMM / Viterbi decoder**
whose emission scores come from the evidence above and whose transition matrix encodes the rules
(very sticky self-transitions; `H↔A` only allowed where control evidence exists; `D` entered only on
out-of-play evidence). Viterbi gives globally consistent boundaries instead of greedy ones.

### 7.5 Parameters (set from football knowledge, *not* tuned on the public window)

| parameter | proposed value | rationale |
|---|---|---|
| control radius | 1.0–1.5 m | typical ball-control distance in tracking literature |
| sustained control | 0.5–1.0 s | "keeps the ball", filters deflections |
| short-gap interpolation | ≤ 10 frames | bridge detector flicker only |
| out-of-bounds margin | 0.5–1 m | tracking noise near lines |
| HMM stay probability | from typical possession length ≈ 10–15 s | `p_stay = 1 − 1/(fps × mean duration)` |

### 7.6 Pros / cons

- ✅ Fast to build, explainable (easy technical document), deterministic, no training.
- ✅ Directly targets the largest baseline errors (fake `D`, flicker).
- ✅ Its confident outputs are exactly the pseudo-labels Proposal B needs.
- ❌ Weak inside long ball blackouts — hand-weighted team-shape score.
- ❌ Many thresholds; without a labelled validation set we cannot tune them, only set them sensibly.

---

## 8. Proposal B — self-trained ML, no external data

**Idea:** the ball is visible on 59% of frames. On those frames, simple rules tell us *with high
confidence* who has the ball (or that it is dead). Use them as **automatic pseudo-labels** to teach a
model **what a team in possession looks like without looking at the ball** — then apply that model
where the ball is missing, and decode the probabilities into clean spells. Allowed by the rules:
labels are produced by an algorithm, no human decides any of them, no external data.

### 8.1 Pipeline

```
ATD + tags ──► features (per frame, multi-scale windows)
                   │
Proposal A rules ──► high-confidence pseudo-labels (subset of frames: H / A / D)
                   │
                   ▼
   ball-blind classifier (gradient boosting)  ──► P(H), P(A), P(D) for every frame
                   │                                   ▲
                   │      ball-aware evidence (control, out-of-play) where ball visible
                   ▼                                   │
          HMM / Viterbi decoding (sticky, same as A) ──► spells ──► CSV
                   │
          (optional) self-training round: add new confident frames, retrain
```

### 8.2 Features (no ball position in the "ball-blind" set)

Per frame, and aggregated (mean / slope / std) over **±1 s, ±3 s, ±8 s** windows, in
attack-normalised coordinates:

- per team: centroid x/y, length, width, convex-hull area, defensive-line height, mean speed,
  forward velocity, players per third, players in the opponent box;
- relative: centroid difference, which team is higher up the pitch, spread ratio, pressing intensity
  (mean distance from each player to the nearest opponent), number of players sprinting per team;
- game context: time since/until ball last/next seen, global player stillness, half.
- Smart Tagging: attacking-team flag and set-piece flag as **features** (optionally excluded — see 8.3).

A second "ball-aware" feature set (nearest-player distances, ball speed, out-of-bounds) is used only
on frames with the ball, or folded into the decoder as extra evidence.

### 8.3 Pseudo-labels

Taken only where evidence is strong (a fraction of frames is enough):

- `H` / `A`: ball visible, one team's player in sustained control (≥ 1 s within ~1 m, no opponent
  within ~2 m), **and** the label is stable across a short window.
- `D`: ball out of the pitch, or the interval between out-of-play and a detected restart, or stillness
  inside a Smart Tagging set piece before the restart.
- Optionally: frames where the rule label and the Smart Tagging attacking team agree, for extra
  confidence.

If the tags are used to *select* labels, drop them from the *features* so the model does not just
learn to copy the tags.

### 8.4 Model and training

- **`sklearn.ensemble.HistGradientBoostingClassifier`** (already installed, deterministic with a fixed
  `random_state`, fast on ~10⁵ rows). Three classes, class-balanced weights.
- **Cross-half training:** train on 1st-half pseudo-labels → predict the 2nd half, and vice versa.
  Avoids fitting the model to the very frames it labels, and gives us a label-free check.
- **Calibrate** probabilities (isotonic / temperature on held-out pseudo-labels) before decoding.
- **Optional self-training:** add frames the model predicts with very high confidence, retrain once
  or twice; stop when the predicted spell count stabilises.

### 8.5 Decoding

Same 3-state sticky HMM as Proposal A, with emissions = calibrated model probabilities, overridden
by hard ball-aware evidence where it exists (control onsets, out-of-play moments, restarts). This
keeps the precise boundaries of the rules where the ball is visible and the model's judgement where
it is not.

### 8.6 Label-free validation — the "artificial blackout" test

We can measure how good the ball-blind model is **without any human label**: take frames where the
ball *is* visible and the pseudo-label is confident, **hide the ball**, run the pipeline, and check
how often the model recovers the pseudo-label and where it places switches. Report accuracy and
spell-level agreement by blackout length (1 s, 5 s, 15 s). This gives us a real decision signal
for choosing features, window sizes and decoder settings.

Caveat: pseudo-labelled frames are the "easy" ones; true blackouts happen disproportionately in
crowded, contested moments, so the test is optimistic.

### 8.7 Pros / cons

- ✅ Attacks the hardest 41% (ball blackouts) with something learned from this match's own data.
- ✅ Has a label-free validation loop (§8.6), unlike pure rules.
- ✅ Fully rule-compliant: algorithmic pseudo-labels, no external data.
- ❌ Depends on Proposal A for pseudo-labels — errors in the rules propagate.
- ❌ More moving parts; more to explain and to make bit-for-bit reproducible.
- ❌ Distribution shift: confident frames ≠ blackout frames.

---

## 9. Discussion and recommendation

### 9.1 Comparison

| | A — rules + HMM | B — self-trained ML + HMM |
|---|---|---|
| build time | 1–2 days | +1–2 days on top of A |
| handles ball blackouts | heuristic team-shape score | learned team-shape model |
| boundary precision (ball visible) | good (event-based) | same (inherits A's events) |
| dead-ball detection | rules | rules + learned `D` probability |
| validation without labels | sanity checks only | artificial-blackout test, cross-half agreement |
| explainability | high | medium |
| reproducibility risk | low | medium (seeds, library versions) |
| risk if it goes wrong | low | medium — can fall back to A |

### 9.2 Recommendation

**Build A first, then B on top — they are one pipeline, not two competing ones.**

B's pseudo-labels *are* A's confident outputs, and B's decoder *is* A's HMM. A is the safe
submission; B is a module that replaces the team-shape heuristic inside ball blackouts. If B does
not convincingly beat A on the label-free checks, we submit A.

Proposed schedule (4 days left):

| day | work | owner |
|---|---|---|
| Nov 13–14 | shared preprocessing + features module; dead-ball detector; control detection; state machine/HMM → **first valid submission (A)** | _tbd_ |
| Nov 14–15 | restart team logic; team-shape heuristic; label-free checks; viewer review of failure cases | _tbd_ |
| Nov 15–16 | Proposal B: pseudo-labels, ball-blind model, artificial-blackout test, integrate into decoder | _tbd_ |
| Nov 16 | freeze choice (A or A+B); reproducibility run from a clean clone; pin `requirements.txt` | _tbd_ |
| Nov 16–17 | technical document; validate exact file; submit via the form **before** the deadline | _tbd_ |

### 9.3 How we decide without labels

Because the public window cannot rank versions (§4.5), every decision uses checks that need no human
labels:

1. **Format:** `validate_submission.py` on every candidate file.
2. **Plausibility:** predicted spell count and mean duration vs football norms; no spells shorter than
   a few frames; home share plausible; `D` share plausible (restarts × typical stoppage length).
3. **Consistency with independent evidence:** every Smart Tagging set piece should be preceded by a
   predicted `D`; predicted out-of-play moments should line up with ball-out-of-bounds frames;
   agreement with attacking tags should improve, not collapse.
4. **Artificial-blackout test** (§8.6) for anything touching the ball-missing logic.
5. **Viewer review:** watch the worst disagreements between versions in `viewer.py`
   (watching to *debug* is allowed; never turn it into labels or frame-specific rules).
6. **Public window last, as a smoke test** with `--bootstrap 1000` — a big drop means a bug;
   a small gain means nothing.

### 9.4 Risks

| risk | mitigation |
|---|---|
| dead-ball semantics differ from ours (when does `D` start/end in the ground truth?) | inspect the 4 public `D` stretches in the viewer to understand the *convention*, then implement it as a general rule |
| over-segmentation (flicker) | sticky decoder; minimum-control requirement; count spells before every submission |
| non-reproducible output | fixed seeds, deterministic sklearn, pinned versions, clean-clone rerun |
| running out of time | A is submittable alone; submit an early valid file and overwrite later (the last form response counts) |
| accidentally breaking data rules | never commit `data/`; never paste data into chat tools; aggregates only |

---

## 10. Open questions for the team

1. **Owners** for: preprocessing/features, dead-ball detector, control + state machine, Proposal B,
   technical document.
2. **Team name** — needed for `<team_name>.csv` / `<team_name>.pdf`.
3. **Ground-truth conventions** (to observe in the viewer on the public window): does `D` start when
   the ball crosses the line or when play stops? Does a change of possession start at the first touch
   of the new team or later?
4. **Smart Tagging:** use as features, as pseudo-label filter, as decoder prior — or not at all?
5. **Video:** do we try our own ball detection to shrink the blackouts? (Proposed: no, unless A+B are
   done early.)
6. **Repo visibility:** keep private until submission day, then switch to public.

---

## 11. Implementation status — hand-off (Nov 16, evening)

> Written so anyone on the team can pick this up without the chat history. Everything below is in
> `solution/`; data stays in `data/` (gitignored). Numbers are aggregates only.

### 11.1 TL;DR

- Three versions were built. **v3 is the current code** (`python solution/pipeline.py`).
  - **v1:** rules plus a 3-state Viterbi.
  - **v2:** v1 plus a self-trained team-shape model and a tag restart prior (commit `eb3f9de`).
  - **v3:** a rebuild around *events*, with no ML. Team switches happen on touches, dead ball starts on line crossings or slow-downs, and dead ball ends on restarts.
- **Which file to submit is not decided yet.** That decision waits on the external validation in §11.5. Until then, either v2 or v3 is a valid, reproducible fallback.
- **Both versions are valid and deterministic.** v3 produces byte-identical files on rerun.
- **v3 needs only numpy and pandas** (`requirements-solution.txt`).

### 11.2 What we measured (this changed the plan)

| finding | consequence |
|---|---|
| The ball is missing on 41% of graded frames, but **85% of missing frames sit in gaps > 10 s**, and 83% of those frames look like stoppages (players walking). During live play the ball is visible on ~91% of frames (94% in the public window). | The ball blackouts that Proposal B targeted are mostly dead ball. We **dropped the ML shape model**. |
| Error attribution on the public window (diagnosis only, nothing fitted): with true team labels and our D, spellF1 0.48 → **0.83**. With true D and our teams, only → 0.53. In v2, team switches landed a median ~2.8 s off, stops ~5.6 s off and restarts ~7.6 s off. | The points are in **timing**: each possession switch on the right touch, each stoppage start and end on the right frame. |
| Labelling convention, seen on the public window: switches happen on **touch moments**, meaning a reception or the new team's first pass after winning a duel. A stoppage where the ball visibly crosses a line starts **on the crossing frame**. The restart team was the nearest player at the restart kick in 4 of 4 cases. | These became the rules in v3. |
| The **broadcast homography** is a strong dead-ball signal. The TV feed cuts to replays and close-ups when play stops. One offset per half is estimated in code (tactical = broadcast + 4061 in the 1st half, + 26943 in the 2nd). "Off" means no valid homography, or the view centres are > 15 m apart along the pitch. It is off on ~74–78% of dead frames against ~13–20% of live frames. | Used as soft dead-ball evidence in v3. |
| **Smart Tagging:** set pieces are useful (stoppage anchor, restart team when given, restart type). Phase tags describe ball location, not possession. | Set pieces used, phase tags dropped. |
| The **rules allow** pre-trained models, external data and labels from *other* games. | This enabled the external validation in §11.5. |

### 11.3 The v3 pipeline (`solution/`)

| file | what it does |
|---|---|
| `atd.py` | parses the tactical ATD into numpy arrays (metres) and caches them in `data/cache/` |
| `tracks.py` | track cleaning: carries a lost player's last position for 1.5 s (so an untracked receiver doesn't vanish), drops ghosts and duplicates |
| `events.py` | **touches** (ball-velocity change over ±5 frames with a player within 2.5 m; the ATD ball is smoothed, so ±2 frames misses most touches), **control** (ball within 1.5 m and moving with a player), **possession runs**, **line crossings**, **ball lost near a line**, **ball lost in open play**, **collective slow-down onsets** (fouls), **restarts** (ball slow, then kicked with a player at it, or ball reappearing in motion), team-shape velocities |
| `broadcast.py` | broadcast-off signal; estimates the per-half offsets itself |
| `pipeline.py` | `label_match(...)`, the **event-gated Viterbi** (see below), clean-up and blackout change points. Smart Tagging and broadcast are optional inputs, so the core also runs on other matches. |
| `diagnostics.py` | label-free plausibility checks (spell counts, D runs, set pieces covered) |
| `blackout_test.py` | hides the ball inside live possessions and checks that labels survive. Current result: 96% kept on 5 s blackouts with no false D, 88% on 15 s, 80% on 30 s |
| `tune_external.py`, `external_report.py` | score variants on the external matches (§11.5): parameter sweep, boundary latency per type, oracle ablations |
| `validation/metrica_sample.py` | **external validation harness (work in progress, see §11.5)** |

**How the decoder works.** It scores H, A and D on every frame, but the cost of *changing* state depends on events:
- **H↔A switch:** cheap (1.5) only at the first uncontested claim of the new team's *controlled* run, meaning ≥ 2 claims or ~0.8 s on the ball. Elsewhere it costs 12. A single touch, deflection or duel can't flip possession.
- **Entering D:** cheap at a line crossing (0.5), a ball lost near a line (2), a ball lost in open play (3) or a slow-down onset (3). Elsewhere it costs 10.
- **Leaving D:** cheap only at a detected restart (1 for the taker's team). Elsewhere it costs 20, so **ball movement during a stoppage doesn't end it**. The restart has to fit how the ball went out (touchline exit → touchline near the exit point; goal-line exit → corner, goal area, or centre after a goal) and the tagged set-piece type.
- **Two passes:** every decoded restart becomes a forced possession claim for the taker's team. The taker is the nearest player at the kick, or the tag's team when given. Then the half is decoded again.
- **Clean-up:** D runs under 2 s without a crossing are dropped, and possession blips under 0.5 s with no touch behind them are dropped.
- **Live blackouts:** if possession changes right after a ball gap, the switch moves to where both teams' shapes reverse.

Gating check: 159 of 165 decoded switches land exactly on a detected control-run touch. Dead-ball entries and exits land on a detected event about 55–65% of the time. The rest are placed by the soft evidence, which is the main thing to improve.

### 11.4 Checks so far (no ground truth involved, except the last row)

| | v1 | v2 | v3 |
|---|---|---|---|
| possessions (full match) | 278 | 283 | 259 |
| dead-ball runs | 129 | 117 | 98 |
| set pieces with a predicted D near the tag | 51/53 | 51/53 | 52/53 |
| restart within 30 s of a set-piece tag start | — | 51/59 | 56/59 |
| broadcast off: dead vs live frames | 0.72 / 0.18 | 0.74 / 0.20 | 0.78 / 0.13* |
| *public 5-min window, bug check only:* spellF1 / macroF1 | 0.480 / 0.790 | 0.485 / 0.755 | 0.412 / 0.826 |

\* v3 also uses broadcast-off as an input, so its agreement is partly circular. Without that input it is 0.72 / 0.18.
**The public window can't rank versions:** 17 possessions, with a bootstrap range of roughly ±0.16.

### 11.5 The two AI subagents that were still running at push time

At hand-off, two background AI agents (Claude Code subagents on Paul's machine) were still running:

1. **External validation harness on Metrica's public sample data.** This is three anonymised matches with synchronised tracking and event data, from github.com/metrica-sports/sample-data. It is free to use, and the source must be credited in the doc. The agent is building `solution/validation/metrica_sample.py`, which:
   - downloads the matches to `data/external/` (gitignored);
   - derives per-frame H/A/D labels from the *event data* (D from the ball going dead to the restart; possession changes only on controlled possession, placed at the new team's first controlled touch);
   - **degrades the tracking to look like our ATD** (ball missing mostly in stoppages, ~9% of live frames missing, player dropouts, noise);
   - exposes `load_game(game_id, degrade=True, seed=0)`, `score(labels, data)` and `evaluate(label_fn, ...)`.

   **Why it matters:** it is the only metric-correlated signal we're allowed to tune on. Labels from other games are explicitly fine; the BAR-ATH public window is not a tuning set.

   **Status at push:** the file is committed as work in progress. It compiles and its interface exists, but its labels and degradation haven't been checked yet.

   **To use it:**
   - `python solution/validation/metrica_sample.py` (sanity check plus baseline scores);
   - `python solution/external_report.py` (where v3 loses points on those games);
   - `python solution/tune_external.py --sweep` (one-at-a-time parameter sweep).

   **Caveats:**
   - The labels are event-derived, so their conventions approximate the ground truth's "corrected tracking owner".
   - There's no Smart Tagging or broadcast for those games, so only the tracking core is validated.
   - Try `--ball-smooth 2..3` to mimic the smoothed ATD ball.

2. **Independent code review** of `events.py`, `tracks.py`, `pipeline.py` and `broadcast.py`. The reviewer looks for indexing, NaN, Viterbi and determinism bugs, and doesn't edit files. **Status at push:** no findings yet.

If the session stays alive, review fixes and harness results will be pushed as follow-up commits to the **`possession-pipeline` branch**, with a PR, *not* merged into `main` automatically.

### 11.6 Next steps, in order

1. **Run the harness:** check its labels and degradation (e.g. the baseline scores), then run `external_report.py`.
2. **Choose the final version on external spellF1.** Compare v3 with the old logic: v1/v2 is commit `eb3f9de`, with an adapter that runs it on any data dict. Then do a coarse sweep of the main costs and thresholds. Don't use the public window for this.
3. **Apply the code-review findings.**
4. **Fill in `TECHNICAL_DOCUMENT.md`:** the external numbers and limitations, plus the Metrica sample data credit.
5. **Submit:**
   - generate the chosen file;
   - rename it to `<team_name>.csv` and the doc to `<team_name>.pdf` or `.md`;
   - run `python evaluation/validate_submission.py <team_name>.csv` on the exact file;
   - submit via the form before the deadline (the last response counts);
   - **make the repo public** and double-check that no data is committed.

**Regenerating each version:**
- v3: `pip install -r requirements-solution.txt && python solution/pipeline.py`.
- v2: check out `eb3f9de` in a separate worktree, `pip install scikit-learn==1.7.2`, then run `python solution/pipeline.py`.

**Still open:** the team name, and the final v2/v3 choice.
