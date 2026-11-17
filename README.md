# GTA VI Twitch CCU Forecaster — v3

Entry for the make.fun competition `gta6-twitch-ccu-forecast-01`:
forecast the Twitch concurrent viewers of the "Grand Theft Auto VI"
category for each hour 2–24 UTC of release day (19 Nov 2026), as five
quantiles (5%, 25%, 50%, 75%, 95%).

## How it works

`main.py` (Python 3.13, standard library only, offline, deterministic)
fits a separate size-weighted regression per hour on the case file's
own comparables (past big launches' hourly Twitch curves):

- the log-ratio `log(viewers_h / viewers_1)` is regressed on log
  hour-1 viewers, with a one-sided quadratic curvature term (the
  bend applies only above the mean size — the biggest launches
  flatten out early);
- a Thursday dummy plus a Thursday × size interaction (Thursday
  releases sit below the line late in the day, and the gap grows with
  launch size — GTA VI releases on a Thursday);
- when the target day is a Friday, Friday comparables count 4× in
  the training weights (Friday is the biggest weekday group);
- the regression median is shrunk 20% toward the pooled median and
  the spread is half the residual scale; slopes are smoothed across
  ±1 hour.

## Validation (local scorer, pinball loss on log scale)

| model | practice | LOO big | LOO Thu-big | LOO Thu-top | LOO all |
|---|---:|---:|---:|---:|---:|
| house baseline | 0.387 | — | — | — | — |
| v1 | 0.170 | 0.270 | 0.309 | 0.264 | 0.529 |
| v2-fixed | 0.167 | 0.245 | 0.169 | 0.175 | 0.479 |
| **v3 (this entry)** | **0.153** | **0.235** | **0.155** | **0.141** | **0.466** |

Practice answers were reconstructed from the same public SullyGnome
source the kit cites; all four hour-1 prints match the cases exactly.
Leave-one-out refits everything per fold. Deterministic (two full
runs byte-identical), ~0.1 s/case.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads `/input/<id>.in`, writes `/output/<id>.out` with 23 rows
(hours 2–24), five quantiles each.
