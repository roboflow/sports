# Soccer AI ⚽

## 💻 install

We don't have a Python package yet. Install from source in a
[**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install git+https://github.com/roboflow/sports.git
cd examples/soccer
pip install -r requirements.txt
./setup.sh
```

## ⚽ datasets

Original data comes from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) 
Kaggle competition. This data has been processed to create new datasets, which can be 
downloaded from the [Roboflow Universe](https://universe.roboflow.com/).

| use case                        | dataset                                                                                                                                                          | train model                                                                                                                                                                                            |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| soccer player detection         | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_player_detector.ipynb)         |
| soccer ball detection           | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)    | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_ball_detector.ipynb)           |
| soccer pitch keypoint detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)   | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_pitch_keypoint_detector.ipynb) |

## 🤖 models

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Player Detection) - Detects 
players, goalkeepers, referees, and the ball in the video.
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Pitch Detection) - Identifies 
the soccer field boundaries and key points.
- [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) - Extracts 
features from image crops of players.
- [UMAP](https://umap-learn.readthedocs.io/en/latest/) - Reduces the dimensionality of 
the extracted features for easier clustering.
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - 
Clusters the reduced-dimension features to classify players into two teams.

## 🛠️ modes

- `PITCH_DETECTION` - Detects the soccer field boundaries and key points in the video. 
Useful for identifying and visualizing the layout of the soccer pitch.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-pitch-detection.mp4 \
  --device mps --mode PITCH_DETECTION
  ```

  https://github.com/user-attachments/assets/cf4df75a-89fe-4c6f-b3dc-e4d63a0ed211

- `PLAYER_DETECTION` - Detects players, goalkeepers, referees, and the ball in the 
video. Essential for identifying and tracking the presence of players and other 
entities on the field.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-detection.mp4 \
  --device mps --mode PLAYER_DETECTION
  ```

  https://github.com/user-attachments/assets/c36ea2c1-b03e-4ffe-81bd-27391260b187

- `BALL_DETECTION` - Detects the ball in the video frames and tracks its position. 
Useful for following ball movements throughout the match.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-ball-detection.mp4 \
  --device mps --mode BALL_DETECTION
  ```

  https://github.com/user-attachments/assets/2fd83678-7790-4f4d-a8c0-065ef38ca031

- `PLAYER_TRACKING` - Tracks players across video frames, maintaining consistent 
identification. Useful for following player movements and positions throughout the 
match.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-tracking.mp4 \
  --device mps --mode PLAYER_TRACKING
  ```
  
  https://github.com/user-attachments/assets/69be83ac-52ff-4879-b93d-33f016feb839

- `TEAM_CLASSIFICATION` - Classifies detected players into their respective teams based 
on their visual features. Helps differentiate between players of different teams for 
analysis and visualization.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-team-classification.mp4 \
  --device mps --mode TEAM_CLASSIFICATION
  ```

  https://github.com/user-attachments/assets/239c2960-5032-415c-b330-3ddd094d32c7

- `RADAR` - Combines pitch detection, player detection, tracking, and team 
classification to generate a radar-like visualization of player positions on the 
soccer field. Provides a comprehensive overview of player movements and team formations 
on the field.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-radar.mp4 \
  --device mps --mode RADAR
  ```

  https://github.com/user-attachments/assets/263b4cd0-2185-4ed3-9be2-cf4d8f5bfa67

### player-motion analytics

Four additional modes overlay player speed, direction, distance, and spotlight
tracking on the broadcast view. They share one tracking pass per run and reuse
the same detectors as the modes above (YOLO by default).

- `DIRECTION` — Team-colored ground ellipses with a velocity joystick dot on each
  player (centroid-based; no pitch homography).

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/renders/2e57b9_0-direction.mp4 \
  --device mps --mode DIRECTION
  ```

- `SPEED` — Same overlay with radial speed badges (m/s) from pitch-space Kalman
  velocity.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/renders/2e57b9_0-speed.mp4 \
  --device mps --mode SPEED
  ```

- `DISTANCE` — Cumulative distance traveled per player (m), shown on the overlay.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/renders/2e57b9_0-distance.mp4 \
  --device mps --mode DISTANCE
  ```

- `SPEED_AND_DISTANCE` — Per-player speed and cumulative distance overlay with
  radar traces. Spotlight one tracked player with `--track-id`, or omit it to
  annotate all players.

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/renders/2e57b9_0-speed-distance.mp4 \
  --device mps --mode SPEED_AND_DISTANCE --track-id 7
  ```

- `ALL` — Run-all orchestrator: computes shared tracking, homography, and
  kinematics once, then writes all five analytics renders (direction, speed,
  distance, speed-and-distance for all players, and a single-player spotlight).

  ```bash
  python main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/renders/2e57b9_0.mp4 \
  --device mps --mode ALL
  ```

#### analytics CLI flags

These flags apply to `DIRECTION`, `SPEED`, `DISTANCE`, `SPEED_AND_DISTANCE`, and
`ALL` (ignored by the original six modes):

| flag | default | purpose |
|:-----|:--------|:--------|
| `--tracker` | `botsort` | Tracker backend: `botsort`, `bytetrack`, or `botsort_nocmc` |
| `--player-detector` | `yolo` | Player detection: `yolo` or `inference` (Roboflow) |
| `--pitch-detector` | `yolo` | Pitch keypoints: `yolo` or `inference` (Roboflow) |
| `--track-id` | *(none)* | `SPEED_AND_DISTANCE` / `ALL`: spotlight this tracker id |
| `--show-track-ids` | off | `SPEED`: show tracker ID chips on players (with speed badges) |
| `--player-model-path` | *(bundled YOLO)* | Override YOLO player `.pt` path |
| `--pitch-model-path` | *(bundled YOLO)* | Override YOLO pitch `.pt` path |
| `--player-model-id` | Roboflow id | Inference player model id |
| `--pitch-model-id` | Roboflow id | Inference pitch model id |
| `--api-key` | env `ROBOFLOW_API_KEY` | Roboflow API key for `--*-detector inference` |
| `--max-frames` | all frames | Cap frames processed (debug / smoke tests) |
| `--cache` / `--no-cache` | cache on | Reuse on-disk detection cache |
| `--cache-dir` | `data/cache` | Cache directory |

Install `inference` from `requirements.txt` only when using the Roboflow
Inference backends.

#### caveats

- **Team colors** — Team assignment uses UMAP + clustering (`TeamClassifier`)
  without a fixed random seed, so jersey colors may swap between runs even when
  tracking ids stay stable.

## 🗺️ roadmap

- [ ] Add smoothing to eliminate flickering in RADAR mode.
- [ ] Add a notebook demonstrating how to save data and perform offline data analysis.

## © license

This demo integrates two main components, each with its own licensing:

- ultralytics: The object detection model used in this demo, YOLOv8, is distributed 
under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- sports: The analytics code that powers the sports analysis in this demo is based on 
the [Supervision](https://github.com/roboflow/supervision) library, which is licensed 
under the [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). 
This makes the sports part of the code fully open source and freely usable in your 
projects.
