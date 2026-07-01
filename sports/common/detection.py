import os
from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    from inference import get_model
except ImportError:
    get_model = None

from sports.common.homography import keypoints_from_inference_field

PITCH_DETECTION_MODEL_PATH = str(
    Path(__file__).resolve().parents[2]
    / "examples"
    / "soccer"
    / "data"
    / "football-pitch-detection.pt"
)

DEFAULT_PITCH_MODEL_ID = "football-field-detection-f07vi/15"


def create_pitch_keypoint_detector(
    *,
    backend: str = "yolo",
    model_path: str | None = None,
    model_id: str = DEFAULT_PITCH_MODEL_ID,
    device: str = "cpu",
    api_key: str | None = None,
) -> Callable[[np.ndarray], sv.KeyPoints]:
    """Return a callable(frame_bgr) -> sv.KeyPoints for pitch keypoint detection."""
    if backend == "yolo":
        path = model_path or PITCH_DETECTION_MODEL_PATH
        model = YOLO(str(path)).to(device=device)

        def _kp_yolo(frame: np.ndarray) -> sv.KeyPoints:
            result = model.predict(frame, conf=0.3, verbose=False, device=device)[0]
            return sv.KeyPoints.from_ultralytics(result)

        return _kp_yolo

    if backend == "inference":
        if get_model is None:
            raise RuntimeError(
                "Install the 'inference' package for inference pitch keypoints."
            )
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError("Set ROBOFLOW_API_KEY for inference pitch keypoints.")
        model = get_model(model_id=model_id, api_key=key)

        def _kp_inf(frame: np.ndarray) -> sv.KeyPoints:
            result = model.infer(frame, confidence=0.3)[0]
            return keypoints_from_inference_field(result)

        return _kp_inf

    raise ValueError(f"Unknown pitch detector backend: {backend!r}")
