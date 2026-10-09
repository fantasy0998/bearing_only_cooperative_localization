"""Exact longest-valid-prefix segmentation of measured 3D bearing directions."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np

DISTANCE2_TOLERANCE = 1e-12  # Floating point error only, not a noise allowance.


@dataclass(frozen=True)
class BearingSegmentationConfig:
    """Angular diameter in degrees and observer candidate-pool lower bound."""
    max_angle_deg: float = 3.0
    representatives_per_segment: int = 1
    candidate_multiplier: float = 2.0
    max_time_gap_s: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_angle_deg) or not 0 < self.max_angle_deg < 180:
            raise ValueError("max_angle_deg must be finite and between 0 and 180.")
        q = self.representatives_per_segment
        if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or q not in (1, 2, 3):
            raise ValueError("representatives_per_segment must be 1, 2 or 3.")
        if not np.isfinite(self.candidate_multiplier) or self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be finite and >= 1.")
        if self.max_time_gap_s is not None and (
            not np.isfinite(self.max_time_gap_s) or self.max_time_gap_s <= 0
        ):
            raise ValueError("max_time_gap_s must be finite and positive.")


@dataclass(frozen=True)
class BearingSegmentationResult:
    """Absolute indices and reason for each segment's starting boundary."""
    segments: tuple[np.ndarray, ...]
    split_reasons: tuple[str, ...]
    fast_check_count: int = 0
    fast_accept_count: int = 0
    exact_pair_comparisons: int = 0


def segment_bearings_buchin(
    bearings_3xm: np.ndarray, absolute_indices: np.ndarray, timestamps_s: np.ndarray,
    *, max_angle_rad: float, max_time_gap_s: float | None = None,
    use_aabb: bool = True,
) -> BearingSegmentationResult:
    """Cover aligned samples by the fewest contiguous angular-diameter segments.

    The AABB is only a safe acceptance bound; failed bounds use exact distances.
    Worst-case time is quadratic, storage linear. No interpolated samples or GT.
    """
    u = np.asarray(bearings_3xm, dtype=float)
    idx = np.asarray(absolute_indices)
    ts = np.asarray(timestamps_s, dtype=float)
    if idx.ndim != 1 or idx.dtype.kind not in 'iu' or np.any(idx < 0):
        raise ValueError("absolute_indices must be a nonnegative integer vector.")
    if u.shape != (3, idx.size) or ts.shape != (idx.size,):
        raise ValueError("Expected aligned (3,M) bearings and (M,) indices/timestamps.")
    if np.any(np.diff(idx.astype(np.int64)) <= 0) or np.any(np.diff(ts) <= 0):
        raise ValueError("Indices and timestamps must be strictly increasing.")
    if not np.isfinite(max_angle_rad) or not 0 < max_angle_rad < np.pi:
        raise ValueError("max_angle_rad must lie strictly between 0 and pi.")
    if max_time_gap_s is not None and (not np.isfinite(max_time_gap_s) or max_time_gap_s <= 0):
        raise ValueError("max_time_gap_s must be finite and positive.")
    norms = np.linalg.norm(u, axis=0)
    if not np.all(np.isfinite(ts)) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("Bearings/timestamps must be finite and bearings nonzero.")
    if not idx.size:
        return BearingSegmentationResult((), ())
    u = u / norms
    bound = (2 * np.sin(max_angle_rad / 2)) ** 2 + DISTANCE2_TOLERANCE
    segments, reasons = [], ['start']
    start = fast_checks = fast_accepts = pairs = 0
    lo = hi = u[:, 0].copy()
    for k in range(1, idx.size):
        reason = None
        if max_time_gap_s is not None and ts[k] - ts[k-1] > max_time_gap_s:
            reason = 'time_gap'
        else:
            accepted = False
            if use_aabb:
                fast_checks += 1
                accepted = np.maximum((u[:, k] - lo)**2, (u[:, k] - hi)**2).sum() <= bound
                fast_accepts += int(accepted)
            if not accepted:
                distances = np.sum((u[:, start:k] - u[:, k, None]) ** 2, axis=0)
                pairs += k - start
                if np.any(distances > bound):
                    reason = 'angular_diameter'
        if reason is not None:
            segments.append(idx[start:k].copy())
            reasons.append(reason)
            start = k
            lo = hi = u[:, k].copy()
        else:
            lo = np.minimum(lo, u[:, k])
            hi = np.maximum(hi, u[:, k])
    segments.append(idx[start:].copy())
    return BearingSegmentationResult(tuple(segments), tuple(reasons), fast_checks, fast_accepts, pairs)


def choose_segment_representatives(
    segments: tuple[np.ndarray, ...], *, per_segment: int,
    timestamps: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Choose original time-midpoint/end samples; absent times use lower midpoint."""
    if isinstance(per_segment, (bool, np.bool_)) or per_segment not in (1, 2, 3):
        raise ValueError("per_segment must be 1, 2 or 3.")
    result = {}
    for sid, segment in enumerate(segments):
        segment = np.asarray(segment, dtype=int)
        if not len(segment):
            continue
        if len(segment) <= per_segment:
            result[sid] = segment.copy()
            continue
        eligible = segment[1:-1] if per_segment == 3 else segment
        if timestamps is None:
            middle = eligible[(len(eligible) - 1) // 2]
        else:
            ts = np.asarray(timestamps)[segment]
            if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
                raise ValueError("Segment timestamps must be finite and increasing.")
            midpoint = (ts[0] + ts[-1]) / 2
            middle = min(eligible, key=lambda i: (abs(timestamps[i] - midpoint), timestamps[i], i))
        keep = [middle] if per_segment == 1 else [segment[0], segment[-1]]
        if per_segment == 3:
            keep.append(middle)
        result[sid] = np.array(sorted(set(keep)), dtype=int)
    return result


def supplement_segment_candidates(
    segments_by_edge: dict[str, tuple[np.ndarray, ...]], *, per_segment: int, minimum_count: int,
    timestamps: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], dict[str, list[int]]]:
    """Fill one observer's largest uncovered sample-position gaps using a heap."""
    if isinstance(minimum_count, bool) or not isinstance(minimum_count, (int, np.integer)) or minimum_count < 0:
        raise ValueError("minimum_count must be a nonnegative integer.")
    reps = {edge: choose_segment_representatives(segs, per_segment=per_segment, timestamps=timestamps)
            for edge, segs in segments_by_edge.items()}
    keep = {edge: set(int(i) for values in mapping.values() for i in values) for edge, mapping in reps.items()}
    added = {edge: [] for edge in segments_by_edge}
    heap = []
    def push(edge, sid, left, right):
        if right - left > 1:
            seg = segments_by_edge[edge][sid]
            middle = (left + right) // 2
            heapq.heappush(heap, (-(right-left-1), edge, int(seg[middle]), sid, left, right))
    for edge, segs in segments_by_edge.items():
        for sid, seg in enumerate(segs):
            positions = np.searchsorted(seg, reps[edge][sid])
            # Virtual boundary positions also cover the gaps beside a single midpoint.
            positions = np.r_[-1, positions, len(seg)]
            for left, right in zip(positions[:-1], positions[1:]):
                push(edge, sid, int(left), int(right))
    count = sum(map(len, keep.values()))
    while count < minimum_count and heap:
        _, edge, index, sid, left, right = heapq.heappop(heap)
        middle = (left + right) // 2
        keep[edge].add(index)
        added[edge].append(index)
        count += 1
        push(edge, sid, left, middle)
        push(edge, sid, middle, right)
    return {edge: np.array(sorted(values), dtype=int) for edge, values in keep.items()}, reps, added
