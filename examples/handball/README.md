# Handball AI

## install

Install from source in a [Python>=3.8](https://www.python.org/) environment.

```bash
pip install git+https://github.com/roboflow/sports.git
cd examples/handball
pip install -r requirements.txt
./setup.sh
```

If a notebook fails with `RuntimeError: Numpy is not available`, restart the
runtime after the install cell finishes, then run the notebook from the top.
This is caused by PyTorch importing before the notebook pins a NumPy version.

## datasets

| use case                         | dataset                                                                                                                                             | train model                                                                                                                                            |
|:---------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| handball player detection        | [Handball_Player_Detection](https://universe.roboflow.com/project-k1mun/handball_player_detection)                                                  | [train_player_detector.ipynb](notebooks/train_player_detector.ipynb)                                                                                   |
| handball ball detection          | [handball-ballon](https://universe.roboflow.com/handballdetection/handball-ballon)                                                                  | [train_ball_detector.ipynb](notebooks/train_ball_detector.ipynb)                                                                                       |
| handball ball detection          | [ball-only-detection](https://universe.roboflow.com/handball-detection/ball-only-detection)                                                        | [train_ball_detector.ipynb](notebooks/train_ball_detector.ipynb)                                                                                       |
| handball court keypoint detection | [Handball Keypoint Detection](https://universe.roboflow.com/handball-wckh8/handball-keypoint-detection-ndvsl)                                      | [train_court_keypoint_detector.ipynb](notebooks/train_court_keypoint_detector.ipynb)                                                                   |

After training, place the exported model weights in:

```text
examples/handball/data/handball-player-detection.pt
examples/handball/data/handball-ball-detection.pt
examples/handball/data/handball-court-keypoint-detection.pt
```

## modes

- `COURT_RENDERING` - Renders an IHF-standard handball court with the 4 m
  throw-off area, 6 m goal-area lines, 9 m free-throw lines, 7 m lines,
  4 m goalkeeper restraining lines, and substitution marks.

  ```bash
  python main.py --target_dir data --mode COURT_RENDERING
  ```

- `POINT_RENDERING` - Draws sample player and ball positions on the handball
  court.

  ```bash
  python main.py --target_dir data --mode POINT_RENDERING
  ```

- `PATH_RENDERING` - Draws sample movement paths on the handball court.

  ```bash
  python main.py --target_dir data --mode PATH_RENDERING
  ```

- `COURT_DETECTION` - Detects handball court keypoints in a video.

  ```bash
  python main.py --source_video_path data/input.mp4 \
  --target_video_path data/court-detection.mp4 \
  --device cpu --mode COURT_DETECTION
  ```

- `PLAYER_DETECTION` - Detects players, referees, and goalkeepers in a video.

  ```bash
  python main.py --source_video_path data/input.mp4 \
  --target_video_path data/player-detection.mp4 \
  --device cpu --mode PLAYER_DETECTION
  ```

- `BALL_DETECTION` - Detects the handball in a video.

  ```bash
  python main.py --source_video_path data/input.mp4 \
  --target_video_path data/ball-detection.mp4 \
  --device cpu --mode BALL_DETECTION
  ```

- `RADAR` - Combines court keypoints and player detections to draw detected
  player locations on a rendered handball court.

  ```bash
  python main.py --source_video_path data/input.mp4 \
  --target_video_path data/radar.mp4 \
  --device cpu --mode RADAR
  ```

- `ALL_RENDERINGS` - Runs all static rendering modes.

  ```bash
  python main.py --target_dir data --mode ALL_RENDERINGS
  ```

Static rendering outputs are written to `examples/handball/data`.

## notebooks

| use case                         | notebook                                                        |
|:---------------------------------|:----------------------------------------------------------------|
| handball player detection        | [train_player_detector.ipynb](notebooks/train_player_detector.ipynb) |
| handball ball detection          | [train_ball_detector.ipynb](notebooks/train_ball_detector.ipynb) |
| handball court keypoint detection | [train_court_keypoint_detector.ipynb](notebooks/train_court_keypoint_detector.ipynb) |
| handball court render            | [render_court.ipynb](notebooks/render_court.ipynb)              |

## license

This example uses the `sports` package and Supervision-based rendering utilities,
which are distributed under the MIT license.
