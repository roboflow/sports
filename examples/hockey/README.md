# Hockey AI

## Install

Install from source in a [**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install git+https://github.com/roboflow/sports.git
cd examples/hockey
pip install -r requirements.txt
```

## Data

Raw video data should be placed in the `data/` directory (gitignored). Current datasets:

- `data/olympic_2022/` - 2022 Winter Olympics hockey matches

## Scripts

### Sample Frames

Extract a single frame from each video at a specified timestamp:

```bash
python sample_frames.py \
    --source_dir data/olympic_2022 \
    --target_dir data/olympic_2022_frames \
    --timestamp 30
```

Arguments:
- `--source_dir`: Directory containing source video files
- `--target_dir`: Directory to save extracted frames
- `--timestamp`: Time in seconds to extract frame (default: 30)
