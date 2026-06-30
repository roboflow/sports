import argparse
import os
from pathlib import Path

import cv2


def sample_frame(video_path: str, timestamp_sec: float) -> cv2.typing.MatLike | None:
    """Extract a single frame from a video at the specified timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame at {timestamp_sec}s from {video_path}")
        return None

    return frame


def process_videos(source_dir: str, target_dir: str, timestamp_sec: float) -> None:
    """Process all videos in source directory and save frames to target directory."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    target_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {source_dir}")
        return

    print(f"Found {len(video_files)} video files")

    for video_file in sorted(video_files):
        print(f"Processing: {video_file.name}")

        frame = sample_frame(str(video_file), timestamp_sec)
        if frame is not None:
            output_filename = f"{video_file.stem}.jpg"
            output_path = target_path / output_filename
            cv2.imwrite(str(output_path), frame)
            print(f"  Saved: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample a single frame from each video at a specified timestamp."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing source video files",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=30.0,
        help="Timestamp in seconds to extract frame (default: 30)",
    )

    args = parser.parse_args()
    process_videos(args.source_dir, args.target_dir, args.timestamp)
    print("Done!")


if __name__ == "__main__":
    main()
