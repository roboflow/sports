# Local Notes

## Status

This fork is the active baseline as of August 24, 2026.

Reason:
- on `clip_1167_1226`, `PLAYER_TRACKING` was visually much better than the archived `soccer-analytics-app` pipeline
- IDs were stable enough to justify switching repos now instead of continuing to tune the old stack

## Local Changes In This Worktree

Files changed:
- `examples/soccer/main.py`
- `sports/configs/soccer.py`

What changed:
- patched radar and team-classification paths to tolerate missing team clusters, missing goalkeepers, and missing pitch keypoints
- patched tracking labels so merged detections do not crash when `tracker_id` is absent after merge
- updated pitch geometry to a 7v7 field in centimeters

## Current Field Geometry

Configured in `sports/configs/soccer.py`:
- length: `5029` cm
- width: `3200` cm
- penalty area: `1097 x 2195` cm
- goal area: `366 x 732` cm
- goal mouth: `198 x 564` cm

## Current Observations

From local visual review:
- `PLAYER_TRACKING` is the strongest mode so far
- radar overlay box renders, but pitch markings did not appear correctly yet
- team colors were not stable
- some players are still missed
- the ball remains weak on the tested clip

## Immediate Next Work

1. Fix radar overlay so projected pitch points/markings are visible and interpretable.
2. Improve player recall before chasing more advanced analytics.
3. Evaluate `BALL_DETECTION` separately from player tracking.
4. Only after that, decide whether to add data export and reconstruction-specific code around this repo.
