import argparse
import os
from pathlib import Path

import cv2


def process_video(
    video_path: str,
    output_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bool:
    """Process a video and draw a black rectangle to hide the box score."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 0, 0),
            -1,
        )

        out.write(frame)
        frame_count += 1

        if frame_count % 500 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    cap.release()
    out.release()
    return True


def process_videos(
    source_dir: str,
    target_dir: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """Process all videos in source directory."""
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
    print(f"Black rectangle: x={x}, y={y}, w={width}, h={height}")

    for i, video_file in enumerate(sorted(video_files), 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")

        output_path = target_path / video_file.name
        success = process_video(str(video_file), str(output_path), x, y, width, height)

        if success:
            print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Hide box score by drawing a black rectangle on videos."
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
        help="Directory to save processed videos",
    )
    parser.add_argument("--x", type=int, default=170, help="X position of rectangle")
    parser.add_argument("--y", type=int, default=40, help="Y position of rectangle")
    parser.add_argument("--width", type=int, default=220, help="Width of rectangle")
    parser.add_argument("--height", type=int, default=180, help="Height of rectangle")

    args = parser.parse_args()
    process_videos(
        args.source_dir,
        args.target_dir,
        args.x,
        args.y,
        args.width,
        args.height,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
