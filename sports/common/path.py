import numpy as np
from scipy.signal import savgol_filter

def _mad(x: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation (MAD) with Gaussian scaling.

    The MAD is a robust measure of statistical dispersion. It represents
    the median of absolute deviations from the data’s median and is scaled
    by 1.4826 to be consistent with the standard deviation for normally
    distributed data.

    Args:
        x (np.ndarray):
            Input array of numeric values. Can contain NaNs, which are ignored
            by `np.median` if the array is pre-filtered.

    Returns:
        float:
            Robust estimate of variability, similar in scale to standard deviation.
    """
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _runs(bool_arr: np.ndarray) -> list[tuple[int, int]]:
    """
    Identify consecutive True regions in a 1D boolean array.

    Returns a list of `(start, end)` index pairs (inclusive) corresponding
    to contiguous runs where the array is True. Useful for grouping
    consecutive detections or events.

    Args:
        bool_arr (np.ndarray):
            1D boolean array representing a binary condition over time or index.
        ```
    """
    idx = np.flatnonzero(bool_arr)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(g[0], g[-1]) for g in groups]


def _linear_interp_1d(y: np.ndarray) -> np.ndarray:
    """
    Perform 1D linear interpolation over NaN gaps.

    Missing (NaN) values inside the array are replaced by linear
    interpolation between neighboring valid points. Leading and trailing
    NaNs are replaced by the nearest valid edge value.

    Args:
        y (np.ndarray):
            1D numeric array that may contain NaN values.

    Returns:
        np.ndarray:
            Array of the same shape with all NaN values filled.
        ```
    """
    n = y.shape[0]
    out = y.copy()
    isnan = np.isnan(out)
    if isnan.all():
        return out
    x = np.arange(n)
    first = np.flatnonzero(~isnan)
    if first.size:
        first_idx = first[0]
        last_idx = first[-1]
        out[:first_idx] = out[first_idx]
        out[last_idx + 1 :] = out[last_idx]
    isnan = np.isnan(out)
    if isnan.any():
        out[isnan] = np.interp(x[isnan], x[~isnan], out[~isnan])
    return out


def _savgol_safe(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """
    Apply Savitzky–Golay smoothing with an automatic fallback.

    When the array is too short for the chosen window and polynomial order,
    the function switches to a simple moving average. This ensures stability
    across short sequences and avoids numerical errors.

    Args:
        y (np.ndarray):
            1D numeric array representing a signal or coordinate sequence.
        window (int):
            Window length for the Savitzky–Golay filter. Must be odd and
            smaller than or equal to the sequence length.
        poly (int):
            Polynomial order for the filter. Defines how flexible the fit is.

    Returns:
        np.ndarray:
            Smoothed version of the input array, same shape as input.
    """
    y = y.astype(float)
    n = y.shape[0]
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2:
        k = min(5, n)
        if k < 2:
            return y
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(k) / k
        return np.convolve(ypad, kernel, mode="valid")
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def clean_paths(
    video_xy: np.ndarray,
    jump_sigma: float = 5.0,
    min_jump_dist: float = 0.7,
    max_jump_run: int = 6,
    pad_around_runs: int = 1,
    smooth_window: int = 9,
    smooth_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean 2D player trajectories by removing teleport-like outliers,
    interpolating missing frames, and smoothing paths.

    The function processes each player's path independently.
    It detects unrealistically large frame-to-frame jumps, replaces
    those sections with linear interpolation, and smooths the
    resulting signal using a Savitzky–Golay filter.

    Args:
        video_xy (np.ndarray):
            Array of shape `(T, P, 2)` containing player coordinates.
            `T` is the frame count, `P` is the number of tracked players,
            and each coordinate is in court space (x, y).

        jump_sigma (float, default=5.0):
            Controls sensitivity to abnormal jumps relative to the player's
            typical movement speed. Lower values make the function more
            aggressive in detecting jumps.
            Example:
            - `jump_sigma=3.0` catches smaller irregularities.
            - `jump_sigma=7.0` only removes the most extreme ones.

        min_jump_dist (float, default=0.7):
            Absolute movement threshold. A jump must exceed this distance
            (in the same unit as your court coordinates) to be treated
            as a teleport, regardless of relative speed.
            - Increase it if normal movements are fast (e.g., full-court sprint).
            - Decrease it if the model often outputs small erratic shifts.

        max_jump_run (int, default=6):
            Maximum number of consecutive frames treated as a short
            "teleport run" to remove. Longer runs are preserved, assuming
            they reflect real movement or tracking loss.
            - Increase to also clean longer glitches.
            - Decrease to only handle quick spikes.

        pad_around_runs (int, default=1):
            Number of frames to also drop before and after each detected
            teleport run. This prevents edge artifacts.
            - Increase if residual jumps appear near corrected regions.
            - Decrease to minimize data removal.

        smooth_window (int, default=9):
            Window length for Savitzky–Golay smoothing. Must be odd.
            Larger values yield smoother, slower-changing paths but can
            blur fast direction changes.
            - Try 5–7 for short, fast clips.
            - Use 9–15 for longer, noisy trajectories.

        smooth_poly (int, default=2):
            Polynomial order for smoothing.
            - `1` for simpler moving-average–like smoothing.
            - `2–3` keeps local curvature (more natural motion arcs).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - `cleaned_xy`: Same shape as input `(T, P, 2)`, cleaned and smoothed.
            - `edited_mask`: Boolean mask `(T, P)` marking frames that were
              interpolated or modified.
    """
    T, P, _ = video_xy.shape
    cleaned = video_xy.astype(float).copy()
    edited = np.zeros((T, P), dtype=bool)

    for p in range(P):
        traj = cleaned[:, p, :]  # (T, 2)
        diffs = np.diff(traj, axis=0)
        speed = np.linalg.norm(diffs, axis=1)

        if np.all(~np.isfinite(speed)) or speed.size == 0:
            continue

        med = np.median(speed[np.isfinite(speed)])
        scale = _mad(speed[np.isfinite(speed)])
        scale = max(scale, 1e-6)

        jump_by_sigma = speed > (med + jump_sigma * scale)
        jump_by_abs = speed > min_jump_dist
        jump_mask = jump_by_sigma & jump_by_abs

        remove = np.zeros(T, dtype=bool)
        for s, e in _runs(jump_mask):
            length = e - s + 1
            if length <= max_jump_run:
                rs = max(0, s - pad_around_runs)
                re = min(T - 1, e + 1 + pad_around_runs)
                remove[rs : re + 1] = True

        if not remove.any():
            for d in range(2):
                traj[:, d] = _savgol_safe(traj[:, d], smooth_window, smooth_poly)
            cleaned[:, p, :] = traj
            continue

        edited[:, p] |= remove
        traj_nan = traj.copy()
        traj_nan[remove, :] = np.nan

        for d in range(2):
            traj_nan[:, d] = _linear_interp_1d(traj_nan[:, d])
            traj_nan[:, d] = _savgol_safe(traj_nan[:, d], smooth_window, smooth_poly)

        cleaned[:, p, :] = traj_nan

    return cleaned, edited
