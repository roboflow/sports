"""analytics/cache.py — on-disk cache for the per-frame neural-network outputs.

Player detections (raw detector output, before tracking) and pitch keypoints are
the slowest part of a run, and they depend only on the clip and the detector model
— not on tracking, homography, or annotation choices. Caching them on disk lets the
first run pay the inference cost once and every later run reuse it.

Homographies (rebuilt cheaply from keypoints) and tracking (re-run from the cached
detections) are intentionally *not* cached. Re-decoding video frames is cheap next
to the detectors, so frames are not cached either.

Cache entries are keyed by the clip identity (path + size + mtime) plus the detector
backend and model id, so different clips or detectors never collide. Each entry is a
pickle of plain numpy arrays (robust across supervision versions) next to a small
JSON manifest describing the key and frame count.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import supervision as sv

# Bump when the on-disk layout changes so stale entries are ignored.
CACHE_VERSION = 1

# Default location: a gitignored cache dir under the example's data folder.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _video_identity(video_path: str) -> str:
    """Identity string for a clip: real path + byte size + modification time."""
    st = os.stat(video_path)
    return f"{os.path.realpath(video_path)}|{st.st_size}|{st.st_mtime_ns}"


def _key_hash(*parts: object) -> str:
    """Short, stable hash of the cache-key parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class FrameCache:
    """Load/save per-frame player detections and pitch keypoints for one clip+model.

    A single instance covers both the detection cache (keyed by the player detector)
    and the keypoint cache (keyed by the pitch detector). When ``enabled`` is False all
    load/save calls become no-ops so callers can share one code path.
    """

    def __init__(
        self,
        video_path: str,
        *,
        cache_dir: str | os.PathLike | None = None,
        enabled: bool = True,
        player_backend: str | None = None,
        player_model_id: str | None = None,
        pitch_backend: str | None = None,
        pitch_model_id: str | None = None,
    ) -> None:
        self.video_path = video_path
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._identity = _video_identity(video_path) if enabled else ""
        self._player_meta = {"backend": player_backend, "model_id": player_model_id}
        self._pitch_meta = {"backend": pitch_backend, "model_id": pitch_model_id}
        self._det_key = _key_hash(
            CACHE_VERSION, self._identity, "det", player_backend, player_model_id
        )
        self._kp_key = _key_hash(
            CACHE_VERSION, self._identity, "kp", pitch_backend, pitch_model_id
        )

    # -- paths ----------------------------------------------------------------
    def _det_stem(self) -> Path:
        return self.cache_dir / f"detections-{self._det_key}"

    def _kp_stem(self) -> Path:
        return self.cache_dir / f"keypoints-{self._kp_key}"

    # -- generic read/write ---------------------------------------------------
    def _read(self, stem: Path) -> dict | None:
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

    def _usable(self, data: dict | None, max_frames: int | None) -> bool:
        """A cache is usable if it is complete, or already covers the frames asked for."""
        if not data:
            return False
        if data.get("complete"):
            return True
        if max_frames is None:
            return False
        return int(data.get("frame_count", 0)) >= int(max_frames)

    # -- detections -----------------------------------------------------------
    def load_detections(self, max_frames: int | None) -> dict[int, sv.Detections] | None:
        if not self.enabled:
            return None
        data = self._read(self._det_stem())
        if not self._usable(data, max_frames):
            return None
        out: dict[int, sv.Detections] = {}
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

    def save_detections(
        self, det_by_frame: dict[int, sv.Detections], *, complete: bool
    ) -> None:
        if not self.enabled:
            return
        frames: dict[int, dict] = {}
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

    # -- keypoints ------------------------------------------------------------
    def load_keypoints(self, max_frames: int | None) -> dict[int, sv.KeyPoints] | None:
        if not self.enabled:
            return None
        data = self._read(self._kp_stem())
        if not self._usable(data, max_frames):
            return None
        out: dict[int, sv.KeyPoints] = {}
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

    def save_keypoints(
        self, kp_by_frame: dict[int, sv.KeyPoints], *, complete: bool
    ) -> None:
        if not self.enabled:
            return
        frames: dict[int, dict] = {}
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


# ---------------------------------------------------------------------------
# Build-or-load helpers (the public entry points used by the analytics features)
# ---------------------------------------------------------------------------

def build_or_load_detections(
    source_video_path: str,
    player_detector_fn,
    cache: FrameCache,
    *,
    max_frames: int | None = None,
) -> dict[int, sv.Detections]:
    """Return ``{frame_idx: detections}`` for the whole clip, using the cache when possible.

    On a cache hit the detector is not run at all; on a miss the detector runs once over
    the clip and the result is written to the cache for later runs.
    """
    import cv2

    cached = cache.load_detections(max_frames)
    if cached is not None:
        print(f"Loaded player detections from cache ({len(cached)} frames).")
        return cached

    print("Computing player detections (cache miss)…")
    det_by_frame: dict[int, sv.Detections] = {}
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


def build_or_load_keypoints(
    source_video_path: str,
    pitch_detector_fn,
    cache: FrameCache,
    *,
    max_frames: int | None = None,
) -> dict[int, sv.KeyPoints]:
    """Return ``{frame_idx: keypoints}`` for the whole clip, using the cache when possible."""
    import cv2

    cached = cache.load_keypoints(max_frames)
    if cached is not None:
        print(f"Loaded pitch keypoints from cache ({len(cached)} frames).")
        return cached

    print("Computing pitch keypoints (cache miss)…")
    kp_by_frame: dict[int, sv.KeyPoints] = {}
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
