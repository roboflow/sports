import hashlib
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "soccer" / "data" / "cache"
)


def _video_identity(video_path: str) -> str:
    """Return a stable identity string for a video file."""
    st = os.stat(video_path)
    return f"{os.path.realpath(video_path)}|{st.st_size}|{st.st_mtime_ns}"


def _key_hash(*parts) -> str:
    """Return a short stable hash of cache-key parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class FrameCache:
    """Load and save per-frame player detections and pitch keypoints for one clip."""

    def __init__(
        self,
        video_path: str,
        cache_dir=None,
        enabled: bool = True,
        player_backend=None,
        player_model_id=None,
        pitch_backend=None,
        pitch_model_id=None,
        ball_model_path=None,
    ):
        self.video_path = video_path
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._identity = _video_identity(video_path) if enabled else ""
        self._player_meta = {"backend": player_backend, "model_id": player_model_id}
        self._pitch_meta = {"backend": pitch_backend, "model_id": pitch_model_id}
        self._ball_meta = {"model_path": ball_model_path}
        self._det_key = _key_hash(
            CACHE_VERSION, self._identity, "det", player_backend, player_model_id
        )
        self._kp_key = _key_hash(
            CACHE_VERSION, self._identity, "kp", pitch_backend, pitch_model_id
        )
        self._ball_key = _key_hash(
            CACHE_VERSION, self._identity, "ball", ball_model_path
        )

    def _det_stem(self) -> Path:
        return self.cache_dir / f"detections-{self._det_key}"

    def _kp_stem(self) -> Path:
        return self.cache_dir / f"keypoints-{self._kp_key}"

    def _ball_stem(self) -> Path:
        return self.cache_dir / f"ball-{self._ball_key}"

    def _read(self, stem: Path):
        path = stem.with_suffix(".pkl")
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, OSError):
            return None

    def _write(self, stem: Path, payload: dict, manifest: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(stem.with_suffix(".pkl"), "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(stem.with_suffix(".json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    def _usable(self, data, max_frames) -> bool:
        if not data:
            return False
        if data.get("complete"):
            return True
        if max_frames is None:
            return False
        return int(data.get("frame_count", 0)) >= int(max_frames)

    def load_detections(self, max_frames):
        """Return cached detections or None when missing or incomplete."""
        if not self.enabled:
            return None
        data = self._read(self._det_stem())
        if not self._usable(data, max_frames):
            return None
        out = {}
        for fi, rec in data["frames"].items():
            fi = int(fi)
            if max_frames is not None and fi > max_frames:
                continue
            out[fi] = sv.Detections(
                xyxy=rec["xyxy"].astype(np.float32),
                confidence=rec["confidence"].astype(np.float32),
                class_id=rec["class_id"].astype(int),
            )
        return out

    def save_detections(self, det_by_frame, complete: bool) -> None:
        """Write detections to disk."""
        if not self.enabled:
            return
        frames = {}
        for fi, dets in det_by_frame.items():
            n = len(dets)
            frames[int(fi)] = {
                "xyxy": np.asarray(dets.xyxy, dtype=np.float32).reshape(n, 4),
                "confidence": (
                    np.asarray(dets.confidence, dtype=np.float32)
                    if dets.confidence is not None
                    else np.ones(n, dtype=np.float32)
                ),
                "class_id": (
                    np.asarray(dets.class_id, dtype=int)
                    if dets.class_id is not None
                    else np.zeros(n, dtype=int)
                ),
            }
        payload = {"complete": complete, "frame_count": len(frames), "frames": frames}
        manifest = {
            "kind": "detections",
            "version": CACHE_VERSION,
            "video_identity": self._identity,
            "detector": self._player_meta,
            "complete": complete,
            "frame_count": len(frames),
        }
        self._write(self._det_stem(), payload, manifest)

    def load_ball_detections(self, max_frames):
        """Return cached ball detections or None when missing or incomplete."""
        if not self.enabled:
            return None
        data = self._read(self._ball_stem())
        if not self._usable(data, max_frames):
            return None
        out = {}
        for fi, rec in data["frames"].items():
            fi = int(fi)
            if max_frames is not None and fi > max_frames:
                continue
            out[fi] = sv.Detections(
                xyxy=rec["xyxy"].astype(np.float32),
                confidence=rec["confidence"].astype(np.float32),
                class_id=rec["class_id"].astype(int),
            )
        return out

    def save_ball_detections(self, ball_by_frame, complete: bool) -> None:
        """Write ball detections to disk."""
        if not self.enabled:
            return
        frames = {}
        for fi, dets in ball_by_frame.items():
            n = len(dets)
            frames[int(fi)] = {
                "xyxy": np.asarray(dets.xyxy, dtype=np.float32).reshape(n, 4),
                "confidence": (
                    np.asarray(dets.confidence, dtype=np.float32)
                    if dets.confidence is not None
                    else np.ones(n, dtype=np.float32)
                ),
                "class_id": (
                    np.asarray(dets.class_id, dtype=int)
                    if dets.class_id is not None
                    else np.zeros(n, dtype=int)
                ),
            }
        payload = {"complete": complete, "frame_count": len(frames), "frames": frames}
        manifest = {
            "kind": "ball",
            "version": CACHE_VERSION,
            "video_identity": self._identity,
            "detector": self._ball_meta,
            "complete": complete,
            "frame_count": len(frames),
        }
        self._write(self._ball_stem(), payload, manifest)

    def load_keypoints(self, max_frames):
        """Return cached keypoints or None when missing or incomplete."""
        if not self.enabled:
            return None
        data = self._read(self._kp_stem())
        if not self._usable(data, max_frames):
            return None
        out = {}
        for fi, rec in data["frames"].items():
            fi = int(fi)
            if max_frames is not None and fi > max_frames:
                continue
            xy = rec["xy"].astype(np.float32)
            conf = rec["confidence"].astype(np.float32)
            out[fi] = (
                sv.KeyPoints.empty()
                if xy.size == 0
                else sv.KeyPoints(xy=xy, confidence=conf)
            )
        return out

    def save_keypoints(self, kp_by_frame, complete: bool) -> None:
        """Write pitch keypoints to disk."""
        if not self.enabled:
            return
        frames = {}
        for fi, kps in kp_by_frame.items():
            if kps is None or kps.xy.shape[0] == 0:
                frames[int(fi)] = {
                    "xy": np.zeros((0, 0, 2), dtype=np.float32),
                    "confidence": np.zeros((0, 0), dtype=np.float32),
                }
                continue
            conf = (
                kps.confidence
                if kps.confidence is not None
                else np.ones(kps.xy.shape[:2], dtype=np.float32)
            )
            frames[int(fi)] = {
                "xy": np.asarray(kps.xy, dtype=np.float32),
                "confidence": np.asarray(conf, dtype=np.float32),
            }
        payload = {"complete": complete, "frame_count": len(frames), "frames": frames}
        manifest = {
            "kind": "keypoints",
            "version": CACHE_VERSION,
            "video_identity": self._identity,
            "detector": self._pitch_meta,
            "complete": complete,
            "frame_count": len(frames),
        }
        self._write(self._kp_stem(), payload, manifest)


def build_or_load_detections(
    source_video_path: str,
    detector_factory,
    cache: FrameCache,
    max_frames=None,
):
    """Return frame-indexed player detections, using the cache when possible."""
    cached = cache.load_detections(max_frames)
    if cached is not None:
        print(f"Loaded player detections from cache ({len(cached)} frames).")
        return cached

    print("Computing player detections (cache miss)...")
    player_detector_fn = detector_factory()
    det_by_frame = {}
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            det_by_frame[frame_idx] = player_detector_fn(frame)
    finally:
        cap.release()
    cache.save_detections(det_by_frame, complete=max_frames is None)
    return det_by_frame


def build_or_load_ball_detections(
    source_video_path: str,
    detector_factory,
    cache: FrameCache,
    max_frames=None,
):
    """Return frame-indexed ball detections, using the cache when possible."""
    cached = cache.load_ball_detections(max_frames)
    if cached is not None:
        print(f"Loaded ball detections from cache ({len(cached)} frames).")
        return cached

    print("Computing ball detections (cache miss)...")
    ball_detector_fn = detector_factory()
    ball_by_frame = {}
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            ball_by_frame[frame_idx] = ball_detector_fn(frame)
    finally:
        cap.release()
    cache.save_ball_detections(ball_by_frame, complete=max_frames is None)
    return ball_by_frame


def build_or_load_keypoints(
    source_video_path: str,
    detector_factory,
    cache: FrameCache,
    max_frames=None,
):
    """Return frame-indexed pitch keypoints, using the cache when possible."""
    cached = cache.load_keypoints(max_frames)
    if cached is not None:
        print(f"Loaded pitch keypoints from cache ({len(cached)} frames).")
        return cached

    print("Computing pitch keypoints (cache miss)...")
    pitch_detector_fn = detector_factory()
    kp_by_frame = {}
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            kp_by_frame[frame_idx] = pitch_detector_fn(frame)
    finally:
        cap.release()
    cache.save_keypoints(kp_by_frame, complete=max_frames is None)
    return kp_by_frame
