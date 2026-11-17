# Ball possession from automatic tracking — technical document

**Challenge:** More than a Hack 2027, Phase 1 (BAR-ATH, frame-level H / A / D).
**Method in one line:** detect football events in the automatic tracking (touches, control, line
crossings, collective slow-downs, restarts) and decode H / A / D with a Viterbi whose transition costs
depend on those events, so possessions start and end on touches and stoppages start and end on the
moments the ball goes out and comes back into play. No learning, no labels from the provided game.

## 1. Why this approach

`spellF1` rewards recovering whole possessions with the right boundaries: a perfect answer shifted by
one second scores 0.65, and every false dead ball or flicker splits a possession in two. Measured on the
data (label-free), the ball is missing on 41% of graded frames but mostly during stoppages — during
live play it is visible on about nine frames in ten. So the points are in **timing**: putting each
possession change on the right touch, and each stoppage's start and end on the right frame. Filling
ball gaps matters much less. The design follows from that:

* team changes are allowed (cheaply) only at the first touch of the new team's *controlled* run;
* dead ball is entered (cheaply) only at a line crossing, a ball lost at a line, or the onset of a
  collective slow-down (the whistle while the ball is still on the pitch);
* dead ball is left (cheaply) only at a detected restart, so ball movement during a stoppage does not
  end it.

A per-frame classifier with smoothing (the baseline) cannot do this: smoothing moves every boundary.

## 2. Assets used and dropped

| asset | used for | |
|---|---|---|
| tactical ATD (players + ball) | everything: touches, control, crossings, slow-downs, restarts, team shape | used |
| Smart Tagging — set pieces | stoppage anchor around each tag start; restart team when `Team` is given; restart type → expected restart zone; second-half kick-off | used |
| Smart Tagging — phase tags | describe where the ball is, not who has it ("Ball in the box, Athletic" overlapped mostly Barcelona possession on inspection) | dropped |
| broadcast homography | **replay / close-up detector**: the TV feed leaves the live view when play stops. Aligned to the tactical timeline by one offset per half, *estimated in code* by sliding the two cameras' view-centre tracks; "off" = no valid homography or view centres > 15 m apart along the pitch | used (soft dead-ball evidence) |
| broadcast ATD, both videos | not downloaded; small ceiling for this design | not used |
| tactical homography | camera centre is only a weak ball proxy (≈12 m off on median) | dropped |
| ML team-shape model (self-trained on pseudo-labels) | tried; small effect, only in live ball blackouts | dropped |

## 3. Training / validation data

No labelled data from the provided game is used, for training or validation. The public 5-minute window
was used only to check the pipeline and to understand the labelling convention.

External validation (labels from other games): the **Metrica Sports public sample data**
(github.com/metrica-sports/sample-data — three anonymised matches with synchronised tracking and event
data, by Metrica Sports). Per-frame H/A/D labels are derived algorithmically from the event data, the
tracking is degraded to resemble the automatic BAR-ATH tracking, and pipeline variants are compared by
spellF1 on those games. _[numbers to be added]_

## 4. Possession rules implemented

* **Touch:** a sharp change of the ball's velocity (kick or reception) with a player within 2.5 m; the
  touch belongs to that player's team; "contested" if an opponent is about as close.
* **Control:** ball within 1.5 m of a player and moving with them, no opponent as close; a run of
  ≥ 8 such frames is a reception even without a velocity peak.
* **Possession change:** only when the other team gains control — two of its claims in a row, or one
  claim and ~0.8 s on the ball. The change is placed at the first uncontested claim of that run (a
  reception, or the first pass after winning a duel). Single touches, deflections and duels without
  control do not change possession.
* **Dead ball starts:** when the visible ball crosses a line (on the crossing frame); when the ball is
  last seen near a line moving out; when the ball is lost and stays missing; or at the onset of a
  collective slow-down of the players while the ball is still on the pitch (fouls, offside, injuries).
* **Dead ball ends:** only at a restart — the ball still for a moment, then kicked, with a player at
  it; or the ball reappearing in motion after a gap. The restart must fit how the ball went out
  (touchline exit → touchline near the exit point; goal-line exit → corner, goal area, or centre after
  a goal) and the set-piece type when tagged. **Restart team** = the team of the player at the ball
  when it is kicked (the tag's team overrides it when given); every restart starts a new possession.
* **Clean-up:** dead-ball runs < 2 s without a crossing are removed; possession blips < 0.5 s without a
  touch by that team are removed.
* **Live ball blackouts:** same team either side → that team throughout; if the decoded possession
  changes right after the gap, the change is moved to where both teams' shapes reverse (attack
  direction of the centroids).

## 5. Limitations

_[to be completed with measured numbers]_

## 6. Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
# data bundle unpacked in ./data (see DATA.md)
.venv/bin/python solution/pipeline.py          # -> evaluation/results/solution_final.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/solution_final.csv
```

Deterministic: numpy only, no randomness; two runs produce byte-identical files.
