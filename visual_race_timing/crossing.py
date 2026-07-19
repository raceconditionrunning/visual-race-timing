"""Crossing-point guessing: zero-crossing of a signed distance from stored detections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from timecode import Timecode

from visual_race_timing.geometry import side_of_line, point_to_line_distance, line_segment_to_box_distance
from visual_race_timing.race_config import get_finish_line

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_COAST_SECONDS = 0.5
DEFAULT_CAP_SECONDS = 12.0
DEFAULT_HYSTERESIS_FRAC = 0.0075  # of frame height
DEFAULT_SEGMENT_MARGIN = 0.05     # fraction of segment length, past each endpoint


@dataclass
class TrackQuality:
    """Signals from one association run, for gating/confidence"""
    n_matched: int = 0
    n_coasted: int = 0
    max_coast_run: int = 0
    n_ambiguous: int = 0          # frames with >1 candidate above the IoU threshold
    min_iou_margin: float = float("inf")  # best - second-best IoU, over matched frames
    lost: bool = False            # gave up: coast budget exceeded before a match

    @property
    def n_frames(self) -> int:
        return self.n_matched + self.n_coasted

    @property
    def coast_frac(self) -> float:
        n = self.n_frames
        return self.n_coasted / n if n else 0.0


@dataclass
class GuessResult:
    frame: Optional[int]
    sub_frame: Optional[float]    # interpolated fractional frame of the flip
    confidence: float
    anchor_track: np.ndarray      # (N, 3): frame, x, y (bottom-center anchor)
    quality: TrackQuality
    reason: Optional[str] = None  # set when frame is None: why no guess was made


def _bottom_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    return np.array([(x1 + x2) / 2.0, y2])


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """box: (4,) xyxy. others: (M,4) xyxy. Returns (M,) IoU."""
    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = others[:, 0], others[:, 1], others[:, 2], others[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def _association_steps(store, detection_source, start_box, start_frame, direction, frame_dims, iou_threshold):
    """Generator: one IoU-association step per frame, indefinitely.

    Shared stepping logic used by both ``track_boxes`` (walk to loss/cap) and
    ``guess_crossing_frame`` (walk until a crossing is confirmed, so it never
    has to keep going -- and risk "losing" the track -- past the point where
    it already found the answer). No pixel data is touched; caller decides
    when to stop.

    Yields ``(query_frame, matched_box_or_None, ambiguous, margin)`` where
    ``matched_box`` is xyxy, ``ambiguous`` is True if >1 candidate cleared the
    IoU threshold, and ``margin`` is best-IoU minus second-best (or best IoU
    itself, if there was only one candidate).
    """
    # `last_*` is the most recent *matched* point (never updated while coasting);
    # `prev_*` is the one before that. Together they give a constant-velocity
    # prediction that keeps extrapolating further out the longer we coast.
    last_box = np.asarray(start_box[:4], dtype=float)
    last_frame = int(start_frame)
    prev_box, prev_frame = None, None
    query_frame = last_frame

    while True:
        query_frame += direction
        steps = query_frame - last_frame

        vel = (last_box - prev_box) / (last_frame - prev_frame) if prev_box is not None else np.zeros(4)
        predicted = last_box + vel * steps

        det_boxes, _, _, _ = store.get_frame_annotation(query_frame, frame_dims, source=detection_source)

        matched_box, ambiguous, margin = None, False, None
        if det_boxes is not None and len(det_boxes) > 0:
            ious = _iou(predicted[:4], det_boxes[:, :4])
            above = np.where(ious >= iou_threshold)[0]
            if len(above) > 0:
                order = above[np.argsort(-ious[above])]
                matched_box = det_boxes[order[0], :4]
                ambiguous = len(order) > 1
                margin = ious[order[0]] - (ious[order[1]] if ambiguous else 0.0)

        if matched_box is not None:
            prev_box, prev_frame = last_box, last_frame
            last_box, last_frame = matched_box, query_frame

        yield query_frame, matched_box, ambiguous, margin


def track_boxes(
    store,
    detection_source: str,
    start_frame: int,
    start_box: np.ndarray,
    direction: int,
    frame_dims: Tuple[int, int],   # (h, w)
    cap_frames: int,
    max_coast: int,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Tuple[np.ndarray, TrackQuality]:
    """Single-target IoU association over stored per-frame detections.

    No pixel data is touched -- pure box association against
    ``store.get_frame_annotation``. Walks ``direction`` (+1 forward, -1
    backward) from ``start_frame``/``start_box`` up to ``cap_frames`` steps,
    coasting on a constant-velocity prediction for up to ``max_coast``
    consecutive missed frames before giving up.

    Returns ``(anchor_track, quality)``. ``anchor_track`` is ``(N, 3)`` =
    ``[frame, x, y]`` of the bottom-center anchor for every *matched* frame
    (the seed box is row 0); it does not include coasted-but-unmatched
    positions.
    """
    assert direction in (1, -1)
    start_box = np.asarray(start_box[:4], dtype=float)
    anchor_track = [np.array([int(start_frame), *_bottom_center(start_box)])]
    quality = TrackQuality()
    coast_run = 0

    steps = _association_steps(store, detection_source, start_box, start_frame, direction, frame_dims, iou_threshold)
    for _ in range(cap_frames):
        query_frame, matched_box, ambiguous, margin = next(steps)
        if matched_box is not None:
            coast_run = 0
            quality.n_matched += 1
            if ambiguous:
                quality.n_ambiguous += 1
            quality.min_iou_margin = min(quality.min_iou_margin, margin)
            anchor_track.append(np.array([query_frame, *_bottom_center(matched_box)]))
        else:
            coast_run += 1
            quality.n_coasted += 1
            quality.max_coast_run = max(quality.max_coast_run, coast_run)
            if coast_run > max_coast:
                quality.lost = True
                break

    if quality.min_iou_margin == float("inf"):
        quality.min_iou_margin = 1.0  # never had a single match to compare against
    return np.array(anchor_track), quality


def _segment_projection_fraction(point, p1, p2) -> float:
    point, p1, p2 = np.asarray(point, dtype=float), np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    line_vec = p2 - p1
    line_len2 = float(np.dot(line_vec, line_vec))
    if line_len2 == 0:
        return 0.0
    return float(np.dot(point - p1, line_vec) / line_len2)


def _signed_distance_one(f, x, y, race_config: dict, fps, frame_dims: Tuple[int, int]) -> float:
    """Signed distance of one anchor point to the (per-frame) finish line.

    Depends only on this row's frame + point -- never on neighbouring rows --
    which is exactly what lets ``CrossingDetector`` accumulate ``d`` one sample
    at a time instead of recomputing the whole track each step.
    """
    frame_h, frame_w = frame_dims
    p0, p1 = get_finish_line(race_config, Timecode(fps, frames=int(f)),
                             frame_width=frame_w, frame_height=frame_h)
    pt = [[x, y]]
    side = side_of_line(p0, p1, pt)[0]
    dist = point_to_line_distance(pt, np.atleast_2d(p0), np.atleast_2d(p1))[0]
    return (1 if side else -1) * dist


class _Run:
    """One maximal same-sign run of d(t); ``max_abs`` is the hysteresis
    evidence for that side of a flip. Grows in place while the run is the open
    (current) one, then is frozen when a sign boundary closes it."""
    __slots__ = ("max_abs", "closed")

    def __init__(self, abs_d: float):
        self.max_abs = abs_d
        self.closed = False


class _Candidate:
    __slots__ = ("before_run", "after_run", "frame_int", "sub_frame")

    def __init__(self, before_run, after_run, frame_int, sub_frame):
        self.before_run = before_run
        self.after_run = after_run
        self.frame_int = frame_int
        self.sub_frame = sub_frame


class CrossingDetector:
    """Streaming form of :func:`find_zero_crossing`.

    Anchor points are pushed one at a time in *walk order* (monotonic in time
    -- ascending for a forward walk, descending for a backward one). Each
    ``push`` is O(1): a point contributes exactly one new signed-distance
    sample and at most one new sign flip at the tail, and earlier samples never
    change (``d`` for a row depends only on that row's own frame + position --
    see :func:`_signed_distance_one`). So the whole per-frame rescan the batch
    form used to do -- recomputing the signed distance over the growing track
    every step, O(n^2) over a walk -- collapses to O(n) total.

    The one non-obvious piece: ``sub_frame`` and the interpolated anchor point
    are *symmetric* in the two bracketing samples
    (``(f_a*|d_b| + f_b*|d_a|)/(|d_a|+|d_b|)``), so feeding a flip's two samples
    in either time order yields the same crossing -- which is what lets the same
    detector serve a backward walk (direction=-1) without reversing the track.
    """

    def __init__(self, race_config: dict, fps, frame_dims: Tuple[int, int],
                 hysteresis_frac: float = DEFAULT_HYSTERESIS_FRAC,
                 segment_margin: float = DEFAULT_SEGMENT_MARGIN,
                 direction: int = 1):
        self.race_config = race_config
        self.fps = fps
        self.frame_dims = frame_dims
        self.eps = hysteresis_frac * frame_dims[0]
        self.segment_margin = segment_margin
        # Points are pushed in walk order; for a backward walk (direction == -1)
        # the previously-pushed sample is the later-in-time one. The batch code
        # only treats a sign change as a flip when the *later* sample is off the
        # line (``d[i] != 0``), so which sample that rule applies to flips with
        # walk direction -- see push().
        assert direction in (1, -1)
        self.direction = direction

        self._prev = None            # (d, sign, frame, x, y) of the last pushed point
        self._cur_run: Optional[_Run] = None
        self._pending: List[_Candidate] = []
        self._n = 0
        self.found_sign_change = False
        self.done = False
        self.result: Optional[Tuple[int, float]] = None

    @staticmethod
    def _sign(d: float) -> float:
        # Matches np.sign: 0.0 stays 0.0 (a sample exactly on the line).
        return 0.0 if d == 0 else (1.0 if d > 0 else -1.0)

    def _register_candidate(self, before_run, after_run, d_p, d_c, f_p, f_c, x_p, x_c, y_p, y_c):
        """A sign flip between the previous and current sample. Record it as a
        commit-pending candidate unless it lands off the physical segment."""
        self.found_sign_change = True
        a_p, a_c = abs(d_p), abs(d_c)
        frac = a_p / (a_p + a_c)      # a_p + a_c > 0: a candidate requires d_c != 0
        sub_frame = f_p + frac * (f_c - f_p)
        anchor_x = x_p + frac * (x_c - x_p)
        anchor_y = y_p + frac * (y_c - y_p)
        p0, p1 = get_finish_line(self.race_config, Timecode(self.fps, frames=int(round(sub_frame))),
                                 frame_width=self.frame_dims[1], frame_height=self.frame_dims[0])
        t = _segment_projection_fraction((anchor_x, anchor_y), p0, p1)
        if not (-self.segment_margin <= t <= 1 + self.segment_margin):
            return  # crossed the infinite line, but off the physical segment
        self._pending.append(_Candidate(before_run, after_run, int(round(sub_frame)), sub_frame))

    def _prune(self):
        """Drop candidates that can never commit: a closed run whose evidence
        never cleared eps can't retroactively grow."""
        eps = self.eps
        self._pending = [
            c for c in self._pending
            if not (c.before_run.closed and c.before_run.max_abs <= eps)
            and not (c.after_run.closed and c.after_run.max_abs <= eps)
        ]

    def push(self, frame, x, y) -> Optional[Tuple[int, float]]:
        """Add one anchor point. Returns ``(frame, sub_frame)`` the first time a
        crossing is confirmed, then ``None`` forever after."""
        if self.done:
            return None

        d = _signed_distance_one(frame, x, y, self.race_config, self.fps, self.frame_dims)
        s = self._sign(d)
        a = abs(d)
        self._n += 1

        if self._prev is None:
            self._cur_run = _Run(a)
            self._prev = (d, s, frame, x, y)
            return None

        d_p, s_p, f_p, x_p, y_p = self._prev
        # A flip requires the later-in-time sample to be off the line, matching
        # the batch rule ``d[i] != 0`` (i = later index). cur is later on a
        # forward walk, prev is later on a backward one.
        later_off_line = (d != 0) if self.direction == 1 else (d_p != 0)
        is_candidate = (s != s_p) and later_off_line
        is_boundary = (s != s_p) and (s != 0) and (s_p != 0)

        if is_boundary:
            before_run = self._cur_run
            before_run.closed = True
            after_run = _Run(a)
            self._register_candidate(before_run, after_run, d_p, d, f_p, frame, x_p, x, y_p, y)
            self._cur_run = after_run
            self._prune()
        else:
            self._cur_run.max_abs = max(self._cur_run.max_abs, a)
            if is_candidate:
                # Previous sample sat exactly on the line: no run boundary, so
                # the "before" and "after" evidence is the same open run.
                self._register_candidate(self._cur_run, self._cur_run, d_p, d, f_p, frame, x_p, x, y_p, y)

        self._prev = (d, s, frame, x, y)

        # find_zero_crossing scans flips in ascending-time order and returns the
        # first that is confirmed, at the shortest track length that confirms
        # any. So: confirm at the first push where a candidate has both sides
        # cleared eps, and among all candidates confirmed at that push pick the
        # earliest in time (smallest sub_frame) -- which is exactly the batch's
        # ascending-index tiebreak, and holds for a backward walk too.
        eps = self.eps
        best = None
        for cand in self._pending:
            if cand.before_run.max_abs > eps and cand.after_run.max_abs > eps:
                if best is None or cand.sub_frame < best.sub_frame:
                    best = cand
        if best is not None:
            self.done = True
            self.result = (best.frame_int, best.sub_frame)
            return self.result
        return None

    def reason(self) -> Optional[str]:
        """Batch-compatible reason string for the sequence pushed so far,
        assuming no crossing was confirmed."""
        if self._n < 2:
            return "track_too_short"
        return "flip_outside_segment_or_jitter" if self.found_sign_change else "no_sign_change"


def find_zero_crossing(
    anchor_track: np.ndarray,
    race_config: dict,
    fps,
    frame_dims: Tuple[int, int],
    hysteresis_frac: float = DEFAULT_HYSTERESIS_FRAC,
    segment_margin: float = DEFAULT_SEGMENT_MARGIN,
) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """Find the first hysteresis-confirmed, in-segment sign flip of d(t).

    ``anchor_track`` must be in ascending time order (frame increasing).
    Returns ``(frame, sub_frame, reason)``; ``reason`` is set (and frame is
    ``None``) when no valid flip was found.

    Thin batch wrapper over :class:`CrossingDetector` -- one implementation,
    fed the whole track at once here and incrementally in
    :func:`guess_crossing_frame`.
    """
    if len(anchor_track) < 2:
        return None, None, "track_too_short"
    det = CrossingDetector(race_config, fps, frame_dims,
                           hysteresis_frac=hysteresis_frac, segment_margin=segment_margin)
    result = None
    for row in anchor_track:
        r = det.push(row[0], row[1], row[2])
        if r is not None:
            result = r
            break
    if result is not None:
        return int(result[0]), result[1], None
    return None, None, det.reason()


def guess_crossing_frame(
    store,
    race_config: dict,
    fps,
    frame_dims: Tuple[int, int],
    start_frame: int,
    start_box: np.ndarray,
    detection_source: str,
    direction: int = 1,
    cap_seconds: float = DEFAULT_CAP_SECONDS,
    coast_seconds: float = DEFAULT_COAST_SECONDS,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    hysteresis_frac: float = DEFAULT_HYSTERESIS_FRAC,
    segment_margin: float = DEFAULT_SEGMENT_MARGIN,
) -> GuessResult:
    """Track from a clicked/seed detection and find where it crosses the line.

    Baseline (build order step 1): bbox bottom-center anchor, no ReID/pose.

    Unlike ``track_boxes``, this checks for a confirmed crossing after every
    new matched frame and stops the walk the moment one is found -- it does
    not keep walking to loss/cap once it already has the answer. That keeps
    ``quality.lost`` meaningful as a trust signal: it's only True when the
    walk genuinely ran out of track *before* finding anything, not whenever
    a (possibly already-answered) walk happens to run off the edge of
    available detections, which -- since only near-line boxes are ever
    detected/stored (see ``detect.py``'s on-line filter) -- it eventually
    always would.
    """
    fps_f = float(fps)
    cap_frames = max(1, round(cap_seconds * fps_f))
    max_coast = max(1, round(coast_seconds * fps_f))

    start_box_arr = np.asarray(start_box[:4], dtype=float)
    start_anchor = _bottom_center(start_box_arr)
    anchor_track = [np.array([int(start_frame), *start_anchor])]
    quality = TrackQuality()
    coast_run = 0
    last_reason = "no_sign_change"

    # The detector is fed anchor points in walk (`direction`) order as they're
    # matched -- no per-frame rescan of the growing track. It is symmetric in
    # time, so a backward walk needs no reversal (see CrossingDetector).
    detector = CrossingDetector(race_config, fps_f, frame_dims,
                                hysteresis_frac=hysteresis_frac, segment_margin=segment_margin,
                                direction=direction)
    detector.push(int(start_frame), start_anchor[0], start_anchor[1])

    steps = _association_steps(store, detection_source, start_box_arr, start_frame, direction, frame_dims, iou_threshold)
    for _ in range(cap_frames):
        query_frame, matched_box, ambiguous, margin = next(steps)
        if matched_box is not None:
            coast_run = 0
            quality.n_matched += 1
            if ambiguous:
                quality.n_ambiguous += 1
            quality.min_iou_margin = min(quality.min_iou_margin, margin)
            anchor = _bottom_center(matched_box)
            anchor_track.append(np.array([query_frame, *anchor]))

            crossing = detector.push(query_frame, anchor[0], anchor[1])
            if crossing is not None:
                frame, sub_frame = crossing
                if quality.min_iou_margin == float("inf"):
                    quality.min_iou_margin = 1.0
                confidence = max(0.0, 1.0 - quality.coast_frac)
                return GuessResult(int(frame), sub_frame, confidence, np.array(anchor_track), quality, reason=None)
            last_reason = detector.reason()
        else:
            coast_run += 1
            quality.n_coasted += 1
            quality.max_coast_run = max(quality.max_coast_run, coast_run)
            if coast_run > max_coast:
                quality.lost = True
                break

    if quality.min_iou_margin == float("inf"):
        quality.min_iou_margin = 1.0
    anchor_track = np.array(anchor_track)
    if quality.lost and len(anchor_track) < 2:
        return GuessResult(None, None, 0.0, anchor_track, quality, reason="lost_immediately")

    # Only blame track loss when we never even saw a sign change to evaluate --
    # if find_zero_crossing rejected a real candidate (jitter/off-segment),
    # that's the more specific, correct diagnostic.
    reason = "lost_track" if (quality.lost and last_reason == "no_sign_change") else last_reason
    return GuessResult(None, None, 0.0, anchor_track, quality, reason=reason)


DEFAULT_LINE_DIST_PX = 10.0
DEFAULT_MAX_SCAN_SECONDS = 120.0
# ~2x the median near-line window (see NEAREST_CROSSING_SCAN_PLAN.md); a
# pragmatic (not provably-safe) bound on how far past the current best to
# keep scanning for a backward-from-a-later-seed candidate that beats it.
_TERMINATION_MARGIN_FRAMES = 15


@dataclass
class CrossingCandidate:
    crossing_frame: int
    sub_frame: Optional[float]
    seed_frame: int
    seed_box: np.ndarray          # xyxy, the near-line detection that produced this
    result: GuessResult


@dataclass
class ScanResult:
    best: Optional[CrossingCandidate]
    candidates: List[CrossingCandidate] = field(default_factory=list)  # all confirmed, sorted by distance from playhead
    seeds_tried: int = 0
    frames_scanned: int = 0        # frame span actually covered by the scan
    first_seed_frame: Optional[int] = None  # first near-line detection frame seen, for the no-confirmed-crossing fallback
    reason: Optional[str] = None   # "no_detections_in_range" | "no_confirmed_crossing" | None


class _ScanCache:
    """Memoizes ``get_frame_annotation(frame, source=...)`` for one scan.

    ``guess_crossing_frame``'s bidirectional per-seed walks overlap heavily
    in frame range in a pack (many runners near the line at once), and
    ``get_frame_annotation`` opens a fresh sqlite connection per call --
    on a crowded frame that can mean tens of seeds each coasting to their
    12s cap before giving up, which is the difference between a sub-second
    scan and a multi-second one. ``frame_dims`` is constant within a scan so
    it's not part of the cache key. Delegates everything else (notably
    ``scan_to_annotation``) straight through to the wrapped store.
    """

    def __init__(self, store):
        self._store = store
        self._cache: Dict[Tuple[int, str], tuple] = {}

    def get_frame_annotation(self, frame_number, img_shape=None, source=None):
        key = (frame_number, source)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._store.get_frame_annotation(frame_number, img_shape, source=source)
            self._cache[key] = cached
        return cached

    def __getattr__(self, name):
        return getattr(self._store, name)


def scan_to_near_line_detection(
    store,
    race_config: dict,
    fps,
    frame_dims: Tuple[int, int],   # (h, w)
    playhead: int,
    detection_source: str,
    direction: int,                # +1 forward, -1 backward
    max_scan_seconds: float = DEFAULT_MAX_SCAN_SECONDS,
    line_dist_px: float = DEFAULT_LINE_DIST_PX,
) -> Optional[int]:
    """Nearest frame to ``playhead`` (in ``direction``) with a detection box
    within ``line_dist_px`` of the finish line. Unlike ``find_nearest_crossing``,
    this does no crossing inference -- it just steps to the next on-line
    detection. Returns the frame number, or None if none is found in range."""
    assert direction in (1, -1)
    fps_f = float(fps)
    max_scan_frames = max(1, round(max_scan_seconds * fps_f))

    pos = playhead
    while True:
        next_frame = store.scan_to_annotation(pos, previous=(direction < 0), source=detection_source)
        if next_frame is None:
            return None
        pos = next_frame
        if abs(pos - playhead) > max_scan_frames:
            return None

        det_boxes, _, _, _ = store.get_frame_annotation(pos, frame_dims, source=detection_source)
        if det_boxes is None or len(det_boxes) == 0:
            continue
        p0, p1 = get_finish_line(race_config, Timecode(fps_f, frames=pos),
                                 frame_width=frame_dims[1], frame_height=frame_dims[0])
        if np.any(line_segment_to_box_distance(p0, p1, det_boxes[:, :4]) < line_dist_px):
            return pos


def find_nearest_crossing(
    store,
    race_config: dict,
    fps,
    frame_dims: Tuple[int, int],   # (h, w)
    playhead: int,
    detection_source: str,
    direction: int,                # +1 forward, -1 backward
    max_scan_seconds: float = DEFAULT_MAX_SCAN_SECONDS,
    line_dist_px: float = DEFAULT_LINE_DIST_PX,
    **guess_kwargs,                # forwarded to guess_crossing_frame
) -> ScanResult:
    """Find the confirmed crossing (any runner) nearest to ``playhead``.

    Walks outward from ``playhead`` in ``direction`` using
    ``store.scan_to_annotation`` to jump frame-to-frame between *any*
    detections (skipping empty gaps for free). At each such frame, every
    near-line box not already covered by a prior seed's walked track (see
    dedup below) is used to seed ``guess_crossing_frame`` in *both*
    directions -- a runner spotted near the line may have already crossed,
    so a backward search from that seed is what would find it.

    Dedup is by anchor-track coverage, not proximity-to-seed: a runner sits
    near the line for many consecutive frames, and naively seeding every one
    would redundantly re-run the same search. Each ``guess_crossing_frame``
    call's ``anchor_track`` records the exact positions it walked through;
    a new near-line box is skipped only if its own anchor lands within
    ``0.25 * box_height`` of an already-covered anchor *at that exact
    frame* -- so two different runners near the line in the same frame
    (a pack) are never confused for one another.

    Termination is a pragmatic bound, not a proof: scanning stops once the
    outer position passes ``best.crossing_frame + direction * W`` where
    ``W = ceil(coast_seconds * fps) + 15``, or once ``max_scan_seconds`` is
    exhausted. See NEAREST_CROSSING_SCAN_PLAN.md for why a fully-correct
    bound (scanning the full cap_seconds window past every best candidate)
    isn't worth its cost here.
    """
    assert direction in (1, -1)
    fps_f = float(fps)
    coast_seconds = guess_kwargs.get("coast_seconds", DEFAULT_COAST_SECONDS)
    W = math.ceil(coast_seconds * fps_f) + _TERMINATION_MARGIN_FRAMES
    max_scan_frames = max(1, round(max_scan_seconds * fps_f))

    cache = _ScanCache(store)

    covered: Dict[int, List[Tuple[float, float]]] = {}
    candidates: List[CrossingCandidate] = []
    best: Optional[CrossingCandidate] = None
    seeds_tried = 0
    first_seed_frame: Optional[int] = None

    pos = playhead
    while True:
        next_frame = store.scan_to_annotation(pos, previous=(direction < 0), source=detection_source)
        if next_frame is None:
            break
        pos = next_frame

        if abs(pos - playhead) > max_scan_frames:
            break
        if best is not None and direction * (pos - best.crossing_frame) > W:
            break

        det_boxes, _, _, _ = cache.get_frame_annotation(pos, frame_dims, source=detection_source)
        if det_boxes is None or len(det_boxes) == 0:
            continue

        p0, p1 = get_finish_line(race_config, Timecode(fps_f, frames=pos),
                                 frame_width=frame_dims[1], frame_height=frame_dims[0])
        near_mask = line_segment_to_box_distance(p0, p1, det_boxes[:, :4]) < line_dist_px
        near_boxes = det_boxes[near_mask]

        for box in near_boxes:
            box = box[:4].astype(float)
            anchor = _bottom_center(box)
            box_h = box[3] - box[1]
            thresh = 0.25 * box_h
            already_covered = any(
                math.hypot(anchor[0] - cx, anchor[1] - cy) < thresh
                for cx, cy in covered.get(pos, [])
            )
            if already_covered:
                continue

            if first_seed_frame is None:
                first_seed_frame = pos
            seeds_tried += 1

            for seed_dir in (1, -1):
                result = guess_crossing_frame(
                    cache, race_config, fps_f, frame_dims, pos, box, detection_source,
                    direction=seed_dir, **guess_kwargs,
                )
                for f, x, y in result.anchor_track:
                    covered.setdefault(int(f), []).append((float(x), float(y)))
                if result.frame is None:
                    continue

                candidate = CrossingCandidate(
                    crossing_frame=result.frame, sub_frame=result.sub_frame,
                    seed_frame=pos, seed_box=box.copy(), result=result,
                )
                candidates.append(candidate)

                on_scan_side = direction * (candidate.crossing_frame - playhead) > 0
                if on_scan_side and (
                    best is None or abs(candidate.crossing_frame - playhead) < abs(best.crossing_frame - playhead)
                ):
                    best = candidate

    candidates.sort(key=lambda c: abs(c.crossing_frame - playhead))
    frames_scanned = abs(pos - playhead) if pos != playhead else 0

    if best is not None:
        reason = None
    elif seeds_tried > 0:
        reason = "no_confirmed_crossing"
    else:
        reason = "no_detections_in_range"

    return ScanResult(best=best, candidates=candidates, seeds_tried=seeds_tried,
                      frames_scanned=frames_scanned, first_seed_frame=first_seed_frame, reason=reason)
