"""Sparse video frame reading and H.264 video writing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)

# Track which paths have already emitted a fallback warning, so a long training
# loop with one bad container does not spam logs.
_FALLBACK_WARNED: Set[str] = set()


def _validate_frame_ids(frame_ids: Sequence[int]) -> List[int]:
    """Coerce to ``list[int]`` after rejecting empty / non-monotonic / negative."""
    if frame_ids is None:
        raise ValueError("frame_ids is None")
    fids = [int(f) for f in frame_ids]
    if not fids:
        raise ValueError("frame_ids is empty")
    if any(f < 0 for f in fids):
        raise ValueError(f"frame_ids must be >= 0, got {fids}")
    for i in range(len(fids) - 1):
        if fids[i + 1] <= fids[i]:
            raise ValueError(
                f"frame_ids must be strictly increasing, "
                f"got fids[{i}]={fids[i]} >= fids[{i + 1}]={fids[i + 1]}"
            )
    return fids


def _resolve_avg_rate(stream) -> float:
    """Pick the most reliable frame rate hint from a PyAV stream."""
    rate = getattr(stream, "average_rate", None) or getattr(stream, "base_rate", None)
    if rate is None:
        return 10.0  # LeRobot default; only used as a last-resort fallback
    return float(rate)


def _read_sparse(path: Path, fids: List[int]) -> Dict[int, np.ndarray]:
    """Fast path: pts seek + early stop. Returns ``{fid: rgb_frame}``."""
    import av  # local import — keep PyAV optional for non-data tests

    fid_set = set(fids)
    fid_max = fids[-1]
    collected: Dict[int, np.ndarray] = {}

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        avg_rate = _resolve_avg_rate(stream)
        time_base = stream.time_base
        if time_base is None or float(time_base) <= 0:
            # Cannot compute pts target → defer to caller's fallback.
            return collected

        target_pts = int(round(fids[0] / avg_rate / float(time_base)))
        try:
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        except (av.error.InvalidDataError, av.error.ValueError, OSError):
            # Seek failed; signal fallback by returning empty.
            return collected

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            cur_idx = int(round(float(frame.pts * time_base) * avg_rate))
            if cur_idx in fid_set and cur_idx not in collected:
                collected[cur_idx] = frame.to_ndarray(format="rgb24")
                if len(collected) == len(fid_set):
                    break
            if cur_idx > fid_max:
                break

    return collected


def _read_full_with_counter(path: Path, fids: List[int]) -> Dict[int, np.ndarray]:
    """Slow path: decode from frame 0 with a manual counter. Robust to broken
    pts metadata since it never inspects ``frame.pts``.
    """
    import av

    fid_set = set(fids)
    fid_max = fids[-1]
    collected: Dict[int, np.ndarray] = {}

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in fid_set:
                collected[i] = frame.to_ndarray(format="rgb24")
                if len(collected) == len(fid_set):
                    break
            if i > fid_max:
                break

    return collected


def read_video_frames(
    path: Path | str,
    frame_ids: Sequence[int],
    *,
    allow_fallback: bool = True,
) -> np.ndarray:
    """Read an arbitrary subset of frames from an mp4 without loading the whole
    file into memory.

    Args:
        path: video file path. Must exist; otherwise ``FileNotFoundError``.
        frame_ids: strictly increasing list of frame indices (relative to the
            first frame of the file). Empty / non-monotonic / negative inputs
            raise ``ValueError``.
        allow_fallback: when ``True`` (default), if the fast (pts seek) path
            returns fewer frames than requested, retry once with the slow
            (decode-from-zero with counter) path. When ``False``, raise
            ``IOError`` immediately on a partial read.

    Returns:
        ``np.ndarray`` of shape ``[len(frame_ids), H, W, 3]``, dtype ``uint8``,
        with rows ordered exactly like ``frame_ids``.

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ValueError: when ``frame_ids`` is malformed.
        IOError: when the video cannot be opened, or even after fallback fewer
            frames than requested were decoded (e.g. ``frame_ids`` exceed video
            length, or container is corrupt).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"video file not found: {p}")

    fids = _validate_frame_ids(frame_ids)

    try:
        collected = _read_sparse(p, fids)
    except Exception as exc:  # noqa: BLE001 — capture any PyAV / I/O surprise
        collected = {}
        if not allow_fallback:
            raise IOError(f"sparse decode failed for {p}: {exc}") from exc

    if len(collected) != len(fids) and allow_fallback:
        if str(p) not in _FALLBACK_WARNED:
            logger.warning(
                "video_io: sparse seek under-collected (%d/%d) for %s; "
                "retrying with full-decode fallback (suppressed for further calls)",
                len(collected),
                len(fids),
                p,
            )
            _FALLBACK_WARNED.add(str(p))
        try:
            collected = _read_full_with_counter(p, fids)
        except Exception as exc:  # noqa: BLE001
            raise IOError(f"fallback decode failed for {p}: {exc}") from exc

    if len(collected) != len(fids):
        missing = [f for f in fids if f not in collected]
        raise IOError(
            f"video {p}: decoded {len(collected)}/{len(fids)} requested frames "
            f"(missing first 5: {missing[:5]})"
        )

    return np.stack([collected[f] for f in fids], axis=0)


__all__ = ["read_video_frames"]


def write_h264_mp4(path, frames, *, fps=10, crf=18):
    """Write uint8 [T,H,W,3] RGB frames to an H.264 MP4."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be [T,H,W,3]")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for rgb in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(rgb, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
