from __future__ import annotations

from typing import Generator, Iterable, List, TypeVar
from dataclasses import dataclass, field

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

from sports.configs.soccer import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    TEAM_NONE,
)

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


@dataclass
class TeamLocks:
    """Clip-level team and goalkeeper locks."""

    team_lock: dict
    gk_lock: dict = field(default_factory=dict)
    locked_goal_defenders: tuple[int, int] | None = None


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        # Deterministic + single-threaded: avoids OpenMP/numba segfaults after MPS
        # SigLIP and keeps pink/cyan assignment stable across demo renders.
        self.reducer = umap.UMAP(n_components=3, n_jobs=1, random_state=0)
        self.cluster_model = KMeans(n_clusters=2, n_init=10, random_state=0)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        # UMAP/numba can SIGSEGV while the MPS runtime is still active after
        # SigLIP; park the vision weights on CPU for the reduce+cluster step.
        data = np.ascontiguousarray(data, dtype=np.float64)
        was_mps = str(self.device).startswith("mps")
        if was_mps:
            self.features_model.to("cpu")
            try:
                import torch

                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        labels = np.asarray(self.cluster_model.labels_, dtype=int)
        counts = np.bincount(labels, minlength=2)
        # Rare collapse: almost every crop lands in one cluster. Retry once with a
        # different UMAP seed so demos don't paint the whole pitch one colour.
        if len(labels) >= 20 and int(counts.min()) < max(5, int(0.12 * len(labels))):
            self.reducer = umap.UMAP(n_components=3, n_jobs=1, random_state=1)
            projections = self.reducer.fit_transform(data)
            self.cluster_model = KMeans(n_clusters=2, n_init=10, random_state=1)
            self.cluster_model.fit(projections)
        if was_mps:
            self.features_model.to(self.device)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


def lock_teams_by_tracklet_majority(
    frames: list[tuple[int, sv.Detections]],
) -> dict[int, int]:
    """Lock one team per tracker id using a majority shirt-colour vote.

    Only outfield (player-class) rows contribute votes so goalkeeper rows and
    position-based GK fills cannot flip outfield tracklet colours.

    Args:
        frames: Sequence of ``(frame_idx, detections)`` pairs from a clip.

    Returns:
        Mapping of ``tracker_id`` to locked team id (0 or 1).
    """
    votes: dict[int, list[int]] = {}
    for _, dets in frames:
        if dets.tracker_id is None or dets.data is None:
            continue
        team = np.asarray(
            dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int
        )
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid < 0 or int(dets.class_id[i]) != PLAYER_CLASS_ID:
                continue
            raw = int(team[i])
            if raw in (0, 1):
                votes.setdefault(tid, []).append(raw)

    locked: dict[int, int] = {}
    for tid, vals in votes.items():
        counts = np.bincount(np.asarray(vals, dtype=int), minlength=2)
        locked[tid] = int(np.argmax(counts))
    return locked


def apply_team_lock(
    team_arr: np.ndarray,
    tracker_id: np.ndarray | None,
    team_lock: dict[int, int],
) -> np.ndarray:
    """Override team ids with a clip-level majority lock keyed on tracker id.

    Args:
        team_arr (np.ndarray): Per-detection team ids (modified in place).
        tracker_id (np.ndarray | None): Per-detection tracker ids.
        team_lock (dict[int, int]): Clip-level {tracker_id: team} mapping.

    Returns:
        np.ndarray: The updated team_arr.
    """
    if tracker_id is None or not team_lock:
        return team_arr
    for i, tid in enumerate(tracker_id):
        tid = int(tid)
        if tid in team_lock:
            team_arr[i] = team_lock[tid]
    return team_arr


def relock_detection_teams(
    dets: sv.Detections,
    team_lock: dict[int, int],
    *,
    gk_lock: dict[int, int] | None = None,
) -> sv.Detections:
    """Return detections with ``data['team']`` re-locked to the clip-level mapping.

    Outfield rows use ``team_lock``; goalkeeper rows prefer ``gk_lock`` when set
    so per-frame goal-distance assignment is not overwritten by shirt-colour votes.

    Args:
        dets: Input detections.
        team_lock: Clip-level ``{tracker_id: team}`` mapping.
        gk_lock: Optional clip-level goalkeeper ``{tracker_id: team}`` mapping.

    Returns:
        New detections with updated team data, or ``dets`` unchanged when empty.
    """
    if dets.tracker_id is None or len(dets) == 0:
        return dets
    if not team_lock and not gk_lock:
        return dets
    team = np.asarray(
        dets.data.get("team", np.full(len(dets), TEAM_NONE))
        if dets.data
        else np.full(len(dets), TEAM_NONE),
        dtype=int,
    )
    if team_lock:
        team = apply_team_lock(team, dets.tracker_id, team_lock)
    if gk_lock and dets.tracker_id is not None:
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if int(dets.class_id[i]) == GOALKEEPER_CLASS_ID and tid in gk_lock:
                team[i] = gk_lock[tid]
    data = dict(dets.data) if dets.data else {}
    data["team"] = team
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )


def clone_team_frames(
    frames: list[tuple[int, sv.Detections]],
) -> list[tuple[int, sv.Detections]]:
    """Copy tracked frames with fresh ``data`` dicts for lock derivation."""
    cloned: list[tuple[int, sv.Detections]] = []
    for frame_idx, dets in frames:
        data = dict(dets.data) if dets.data else {}
        if "team" in data:
            data["team"] = np.array(data["team"], dtype=int)
        cloned.append(
            (
                frame_idx,
                sv.Detections(
                    xyxy=dets.xyxy,
                    class_id=dets.class_id,
                    tracker_id=dets.tracker_id,
                    confidence=dets.confidence,
                    data=data,
                ),
            )
        )
    return cloned

