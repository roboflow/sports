# Soccer Match Analytics

Computer vision pipeline for turning raw soccer footage into an annotated match analysis video.
It detects and tracks players, goalkeepers, referees, the ball, and pitch keypoints, then estimates team possession and passes in perspective-corrected field coordinates.
I built the analytics layer on top of [`roboflow/sports`](https://github.com/roboflow/sports), adding temporal smoothing, short-gap ball tracking, and an in-video statistics overlay.

## Demo

https://github.com/user-attachments/assets/d44728f8-c230-4d3a-9bb7-93ba1588fb97

## Highlights

- Detects and tracks players, goalkeepers, referees, the ball, and pitch keypoints.
- Classifies players by team and projects detections onto real pitch coordinates.
- Estimates possession using the nearest player to the ball with three-frame smoothing.
- Counts likely passes from possession continuity and tracked-player changes.
- Extrapolates ball position across short detection gaps.
- Produces an annotated video with possession percentages, pass counts, and a possession-colored ball trail.

## How It Works

1. **Detect and track:** object-detection models identify match participants, the ball, and pitch keypoints; tracked IDs maintain player continuity across frames.
2. **Transform coordinates:** a homography maps image positions to the configured pitch coordinate system, preventing camera perspective from distorting distance calculations.
3. **Estimate possession:** possession is assigned to the team of the nearest player or goalkeeper within a configurable distance threshold.
4. **Stabilize predictions:** a three-frame window reduces rapid possession changes caused by noisy detections.
5. **Generate analytics:** changes between tracked players on the same possessing team provide a heuristic pass count, while overlays visualize the results.

## Technical Scope

| Capability | Upstream `roboflow/sports` | This project |
|:--|:--:|:--:|
| Player, goalkeeper, referee, and ball detection | Yes | Yes |
| Pitch-keypoint detection and field projection | Yes | Yes |
| Player tracking and team classification | Yes | Yes |
| Radar visualization | Yes | Yes |
| `MATCH_ANALYTICS` processing mode | No | Yes |
| Perspective-corrected possession estimation | No | Yes |
| Temporal possession smoothing | No | Yes |
| Heuristic pass counting | No | Yes |
| Possession trail and statistics overlay | No | Yes |
| Ball extrapolation during detection gaps | No | Yes |

## Installation

Python 3.8 or newer is required.

```bash
git clone https://github.com/KomisD/sports.git
cd sports/examples/soccer
pip install -r requirements.txt
./setup.sh
```

The setup script downloads the expected model weights into `examples/soccer/data/`:

- `football-player-detection.pt`
- `football-pitch-detection.pt`
- `football-ball-detection.pt`

The package can also be installed directly from GitHub:

```bash
pip install git+https://github.com/KomisD/sports.git
```

## Usage

Run the analytics pipeline from `examples/soccer`:

```bash
python main.py \
  --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-match-analytics.mp4 \
  --device cpu \
  --mode MATCH_ANALYTICS
```

Use `cuda`, `mps`, or another supported device instead of `cpu` when available. The command writes the annotated result to `--target_video_path`.

## Implementation

### `sports/common/possession.py`

- Introduces `PossessionTracker` for temporal smoothing and possession state.
- Projects the ball and player detections into pitch coordinates.
- Finds the closest eligible player and returns the estimated team, tracker ID, and distance.

### `sports/common/ball.py`

- Adds possession-aware trail colors, a current-ball marker, and team labels.
- Adds constant-velocity position extrapolation when a ball detection is briefly missing.

### `examples/soccer/main.py`

- Adds the `MATCH_ANALYTICS` command-line mode.
- Implements the two-pass analysis pipeline and video output.
- Renders possession percentages, current possession, and pass counts.

## Limitations

This is an experimental analytics pipeline, not a source of validated match statistics. Possession and pass estimates depend on detection quality, visible pitch keypoints, team classification, and stable tracker IDs.

- Processing is offline and uses two passes over the video, so it is not a real-time stream.
- The pass heuristic does not verify ball flight, receiver control, minimum possession time, or restarts.
- Occlusion and tracker-ID changes can produce false positives.
- Ball extrapolation uses a simple constant-velocity assumption.
- The analytics additions do not yet have an automated test suite.

## Upstream Project and License

This project builds on [`roboflow/sports`](https://github.com/roboflow/sports) and its soccer computer-vision examples. See [`LICENSE`](LICENSE) for licensing information; third-party libraries and model tooling retain their respective licenses.

Original upstream demo:
https://github.com/roboflow/sports/assets/26109316/7ad414dd-cc4e-476d-9af3-02dfdf029205
