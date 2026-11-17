# Fellowship — ball possession from automatic tracking (technical document)

**Team:** Fellowship · **Code:** <https://github.com/seb-thoms/fcb-hackathon> · **Submission:** `Fellowship.csv` (regenerate with `python solution/pipeline.py --out submission/Fellowship.csv`; MD5 `8aac452748347bc72c7cd66db97ba5ba`)

**Challenge:** More than a Hack 2027, Phase 1 (BAR-ATH, frame-level H / A / D).
**Method in one line:** detect football events in the automatic tracking (touches, control, line
crossings, collective slow-downs, restarts), then decode H / A / D with a Viterbi whose transition
costs depend on those events. Possessions start and end on touches; stoppages start when the ball goes
out and end when it comes back into play. There is no learning and no label from the provided game.

## 1. Why this approach

`spellF1` rewards recovering whole possessions with the right boundaries. A perfect answer shifted by
one second scores 0.65, and every false dead ball or flicker splits a possession in two.

On the data itself (label-free), the ball is missing on 41% of graded frames, but mostly during
stoppages:
- 85% of missing frames sit in gaps longer than 10 s;
- most of those gaps look like stoppages, with the players walking;
- during live play the ball is visible on about nine frames in ten.

So the points are in **timing**: placing each possession change on the right touch, and each
stoppage's start and end on the right frame. Filling ball gaps matters much less. The design follows
from that:

* a team change is cheap only at the first touch of the new team's *controlled* run;
* entering dead ball is cheap only at a line crossing, a ball lost at a line or in open play, or the
  onset of a collective slow-down (a whistle while the ball is still on the pitch);
* leaving dead ball is cheap only at a detected restart, so ball movement during a stoppage does not
  end it.

A per-frame classifier with smoothing, like the baseline, cannot do this: smoothing moves every
boundary.

## 2. Assets used and dropped

| asset | role | status |
|---|---|---|
| tactical ATD (players and ball) | everything: touches, control, crossings, slow-downs, restarts, team shape | used |
| Smart Tagging, set pieces | stoppage anchor around each tag start; restart team when `Team` is given; restart type → expected restart zone; second-half kick-off | used |
| Smart Tagging, phase tags | they describe where the ball is, not who has it | dropped |
| broadcast homography | **replay / close-up detector**: the TV feed leaves the live view when play stops. Aligned to the tactical timeline by one offset per half, *estimated in code* by sliding the two cameras' view-centre tracks against each other (+4061 and +26943 frames). "Off" means no valid homography, or view centres more than 15 m apart along the pitch. Off on ~83% of predicted dead frames against ~13% of live frames | used (soft dead-ball evidence) |
| broadcast ATD and both videos | not downloaded; small ceiling for this design (live play with the tactical ball missing is ~5% of graded frames) | not used |
| tactical homography | its view centre is only a weak ball proxy (~12 m off on median) | dropped |
| self-trained team-shape model (pseudo-labels) | tried; small effect, since it only acts in live ball blackouts | dropped |

## 3. Training and validation data

**No labelled data from the provided game** is used, for training or validation. The public 5-minute
window was used only to check the pipeline and to understand the labelling convention: switches on
touches, a stoppage starting on the line-crossing frame, the restart team being the player at the ball.

**External validation, with labels from other games:** the **Metrica Sports public sample data**
(github.com/metrica-sports/sample-data). These are three anonymised matches with synchronised tracking
and event data, published by Metrica Sports. We thank Metrica Sports for making them public.
The event data was produced by Metrica Sports and published with the sample; our team labelled
nothing by hand. Amount: 3 matches, about 700 labelled possessions.

- **Labels:** per-frame H/A/D, derived algorithmically from the event data (`solution/validation/`):
  - D runs from the ball going dead (the line-crossing frame, foul, offside, goal) to the restart;
  - possession changes only when the other team gains control, placed at its first controlled touch;
  - a second rule, where every opponent touch switches possession, is also scored for robustness.
- **Degraded input:** the clean tracking is degraded to resemble the BAR-ATH automatic tracking:
  - ball present on ~60% of frames, mostly missing in stoppages, ~10% missing in live play, with BAR-ATH's gap-length mix;
  - player dropouts, duplicates and team-flipped ghosts;
  - positional noise.

  The ball is optionally smoothed to match the BAR-ATH ATD ball's acceleration profile.
- **Parameter selection:** parameters were chosen by one-at-a-time sweeps and combined A/B tests on
  these games (`solution/tune_external.py`): 3 games × 3 degradation seeds, both label rules, two
  ball-smoothing levels. Changes were kept only if they helped under both label rules and both
  smoothing levels.
- **Label-free check:** the BAR-ATH blackout test arbitrated one trade-off. Hiding the ball inside
  live possessions must not create false dead balls.

| external spellF1 (mean of 9 game × seed runs) | control rule | touch rule | macroF1 |
|---|---|---|---|
| starter baseline (±50 majority vote) | 0.331 | 0.328 | 0.774 |
| first rules + Viterbi (v1 logic) | 0.592 | 0.460 | 0.844 |
| event-gated pipeline, initial parameters | 0.661 | 0.553 | 0.873 |
| event-gated pipeline, tuned parameters (v4), harness ball | 0.710 | 0.576 | 0.897 |
| v4, ball smoothed like the ATD | 0.693 | 0.541 | 0.886 |
| **final (v5: switch backdating + claim merging)**, harness ball | **0.729** | **0.577** | **0.905** |
| **final (v5)**, ball smoothed like the ATD | **0.710** | **0.547** | **0.892** |

We prioritise the "control" rule because it is the definition `CHALLENGE.md` gives for the ground truth:
possession changes only when the other team gains *controlled* possession. The "touch" rule guards
against large losses if the convention is looser. Stricter variants that gained more on the control rule
but lost ≥ 0.02 on the touch rule, and that under-switched (predicted / true possessions 0.89), were
rejected.

The harness is easier than BAR-ATH (the baseline scores 0.33 there against 0.24 on BAR-ATH), so these
numbers rank variants. They do not predict the BAR-ATH score.

## 4. Possession rules implemented

* **Touch:** a sharp change of the ball's velocity (measured over ±5 frames, since the ATD ball is
  smoothed) with a player within 2.5 m. The touch belongs to that player's team, and is "contested" if
  an opponent is about as close.
* **Control:** the ball within 1.0 m of a player, moving with them, with no opponent as close. A run
  of ≥ 8 such frames counts as a reception even without a velocity peak.
* **Possession change:** only when the other team gains control. That means two claims in a row, or
  one claim and ~0.8 s on the ball. Claims by the same player within 0.5 s count as one, since a single
  action often yields both a velocity peak and a control onset. Single touches, deflections and duels
  without control do not change possession.
* **Where the change is placed:** at the new team's *first touch*. Once control is confirmed, the switch
  is moved back to the new team's earliest claim since the old team's last controlled run. That search
  goes back at most 3 s and skips uncontrolled touches in between. The switch then moves further back,
  by up to 1 s, over the frames where a player of the new team was already at the ball. This follows
  the convention seen on the public window: the switch happens at the reception, or at the first pass
  after winning a duel.
* **Dead ball starts:**
  - when the visible ball crosses a line, on the crossing frame;
  - when the ball is last seen near a line moving out;
  - when the ball is lost and stays missing;
  - at the onset of a collective slow-down of the players while the ball is still on the pitch (fouls,
    offside, injuries).
* **Dead ball ends:** only at a restart.
  - **Detection:** the ball slow for a moment, then kicked with a player at it; or the ball reappearing in motion after a gap.
  - **Location:** the restart must fit how the ball went out. A touchline exit means a restart on the touchline near the exit point. A goal-line exit means a corner, the goal area, or the centre after a goal. When tagged, it must also fit the set-piece type.
  - **Restart team:** the team of the player at the ball when it is kicked; the tag's team overrides it when given.
  - **New possession:** every restart starts one, via a second decoding pass.
* **Clean-up:** dead-ball runs under 3 s without a line crossing are removed, and so are possession
  blips under 0.5 s without a touch by that team.
* **Live ball blackouts:** with the same team either side, that team holds the whole gap. If the decoded
  possession changes right after the gap, the change moves to where both teams' shapes reverse (the
  attack direction of their centroids).

## 5. Limitations

* **Team-switch timing is still the main error.** On the external games, perfect team labels on top of
  our dead-ball mask would raise spellF1 from ~0.69 to ~0.84. Perfect dead-ball labels on top of our
  teams would only raise it to ~0.73. Switch counts are now close to the truth, and switch errors show
  no bias (median signed error ~0 s). But only 40–50% of switches land within 0.5 s, and very short
  possessions (under ~3 s) are mostly missed.
* **Dead-ball boundaries without a visible crossing** (fouls, balls lost off-camera) rely on softer
  evidence. About 75% of BAR-ATH stoppage starts and ends coincide with a detected event; the rest are placed
  by player stillness, broadcast cuts and set-piece tags.
* **The ATD ball is the ground projection of what the camera sees,** so lofted balls are displaced. We
  do not model ball height.
* **The external harness is cleaner than BAR-ATH,** and its labels come from events rather than a
  corrected-tracking owner channel. It ranks variants; it does not predict the board score.
* **The public 5-minute window** (17 possessions) cannot rank versions: its bootstrap range is about
  ±0.16 spellF1. We did not tune on it.
* **With more time:** a learned touch/control detector trained on the external matches' event labels;
  ball recovery from the broadcast tracking or the video; and a duration-aware (semi-Markov) decoder.

## 6. Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-solution.txt   # numpy, pandas
# data bundle unpacked in ./data (see DATA.md)
.venv/bin/python solution/pipeline.py          # -> evaluation/results/solution_final.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/solution_final.csv
```

The output is deterministic (numpy only, no randomness): two runs produce byte-identical files.

To reproduce the parameter selection:
1. `python solution/validation/metrica_sample.py` downloads the public sample games to `data/external/`.
2. Run `python solution/tune_external.py --ab '{...}'` or `--sweep '{...}'`.
3. `python solution/external_report.py` reports boundary errors and oracle ablations.
4. `python solution/blackout_test.py` runs the label-free blackout check on BAR-ATH.
