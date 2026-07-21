"""Standalone, tracker-independent ReID feature bank backed by boxmot's ReID.

Design notes
------------
* Embeddings come from boxmot's :class:`boxmot.reid.core.ReID`. We reuse its
  ``__call__(img, boxes=xyxy)`` path so cropping + preprocessing (default
  ``"resize"``) + L2-normalization match the tracker's embedding space exactly.
  We deliberately do NOT reimplement ``get_crops``.
* The EMA blend mirrors boxmot ``STrack.update_features`` exactly:
  ``smooth = normalize(alpha * prev + (1 - alpha) * curr)`` with ``alpha = 0.9``
  (see ``boxmot/trackers/common/track_models/botsort.py`` and
  ``boxmot/trackers/common/appearance/__init__.py``).
* Persistence is a plain ``.npz`` of ``{ids, feats}`` (regenerable state).
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Callable, List, Optional, Tuple

import numpy as np

from boxmot.reid.core import ReID

from visual_race_timing.logging import get_logger

logger = get_logger(__name__)

# Mirrors boxmot STrack.update_features (botsort.py: self.alpha = 0.9).
DEFAULT_EMA_ALPHA = 0.9

# The osnet_ain_x1_0 MSMT17 weights that ship in the repo's data/ directory.
# boxmot infers the architecture from the filename substring "osnet_ain_x1_0".
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_REID_WEIGHTS = (
    _REPO_ROOT
    / "data"
    / "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)


def available_reid_models() -> List[str]:
    """Names boxmot recognizes and can auto-download (see ``--reid-model``)."""
    from boxmot.reid.core.config import TRAINED_URLS
    return sorted(TRAINED_URLS.keys())


def build_extractor(weights=None, device: str = "cpu", half: bool = False) -> ReID:
    """Construct a single boxmot ReID runtime.

    Parameters
    ----------
    weights : path-like or None
        ReID checkpoint. Defaults to the repo's osnet_ain_x1_0 MSMT17 weights.
    device : str
        Torch device string ("cpu", "mps", "0", ...).
    half : bool
        Use fp16. Keep False on CPU/MPS.
    """
    if weights is None:
        weights = DEFAULT_REID_WEIGHTS
    weights = pathlib.Path(weights)
    if not weights.is_file():
        logger.warning(
            "ReID weights %s not found; boxmot will try to resolve/download by name.",
            weights,
        )
    else:
        weights = _ensure_pt_suffix(weights)
    return ReID(weights=str(weights), device=device, half=half)


def _ensure_pt_suffix(weights: pathlib.Path) -> pathlib.Path:
    """boxmot only accepts a ``.pt`` PyTorch suffix; torchreid ships ``.pth``.

    Create a sibling ``.pt`` alias (symlink, or copy as a fallback) preserving
    the filename stem so boxmot's substring-based architecture inference
    (e.g. ``osnet_ain_x1_0``) still resolves. Returns the ``.pt`` path.
    """
    if weights.suffix.lower() == ".pt":
        return weights
    alias = weights.with_suffix(".pt")
    if not alias.exists():
        try:
            alias.symlink_to(weights.name)
        except OSError:
            shutil.copy2(weights, alias)
        logger.info("Aliased ReID weights %s -> %s for boxmot.", weights.name, alias.name)
    return alias


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return np.zeros_like(vec)
    return vec / norm


def _ema_update(prev: Optional[np.ndarray], curr: np.ndarray,
                alpha: float = DEFAULT_EMA_ALPHA) -> np.ndarray:
    """boxmot-identical EMA: normalize(alpha*prev + (1-alpha)*curr)."""
    curr = _l2_normalize(curr)
    if prev is None:
        return curr
    blended = alpha * np.asarray(prev, dtype=np.float32) + (1.0 - alpha) * curr
    return _l2_normalize(blended)


class ReIDBank:
    """A bank of L2-normalized per-runner appearance features.

    Holds ``ids: (M,)`` int64 and ``feats: (M, D)`` float32 (unit rows). It is
    fully independent of any tracker: it only needs the shared ReID extractor.
    """

    def __init__(self, extractor: ReID, ids: Optional[np.ndarray] = None,
                 feats: Optional[np.ndarray] = None,
                 alpha: float = DEFAULT_EMA_ALPHA):
        self.extractor = extractor
        self.alpha = float(alpha)
        if ids is None or feats is None or len(ids) == 0:
            self.ids = np.empty((0,), dtype=np.int64)
            self.feats = np.empty((0, 0), dtype=np.float32)
        else:
            self.ids = np.asarray(ids, dtype=np.int64).reshape(-1)
            self.feats = np.asarray(feats, dtype=np.float32)

    # -- feature extraction ------------------------------------------------
    def extract(self, img: np.ndarray, box) -> np.ndarray:
        """Extract a single L2-normalized embedding for one xyxy box.

        ``box`` may be shape (4,), (1,4) or (1, >=4); only the first 4 columns
        (x1,y1,x2,y2 in pixels) are used.
        """
        box = np.atleast_2d(np.asarray(box, dtype=np.float32))[:1, :4]
        feats = self.extractor(img, boxes=box)  # (1, D), already L2-normalized
        feats = np.asarray(feats, dtype=np.float32)
        if feats.size == 0:
            raise ValueError("ReID extractor returned no features for the box.")
        return _l2_normalize(feats[0])

    # -- queries -----------------------------------------------------------
    def guess(self, img: np.ndarray, box) -> Tuple[List[int], List[float]]:
        """Return (sorted_ids, sorted_cosine_dists) ranked closest-first.

        Cosine distance = 1 - cosine_similarity, clipped to >= 0 (rows are
        unit-normalized, matching boxmot's cosine embedding distance).
        """
        if len(self.ids) == 0:
            return [], []
        query = self.extract(img, box)  # (D,)
        sims = self.feats @ query  # (M,)
        dists = np.maximum(0.0, 1.0 - sims)
        order = np.argsort(dists)
        sorted_ids = [int(i) for i in self.ids[order]]
        sorted_dists = [float(d) for d in dists[order]]
        return sorted_ids, sorted_dists

    def guess_batch(self, img: np.ndarray, boxes) -> List[Tuple[List[int], List[float]]]:
        """Full closest-first ranking for each of N xyxy boxes in one extractor call.

        Returns a length-N list (input order) of ``(sorted_ids, sorted_dists)``,
        each mirroring :meth:`guess`. Entries are ``([], [])`` when the bank is
        empty. One extractor forward pass covers all boxes.
        """
        boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float32))[:, :4]
        n = len(boxes)
        if n == 0:
            return []
        if len(self.ids) == 0:
            return [([], []) for _ in range(n)]
        feats = np.asarray(self.extractor(img, boxes=boxes), dtype=np.float32)  # (N, D)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        feats = feats / norms
        dists = np.maximum(0.0, 1.0 - feats @ self.feats.T)  # (N, M)
        order = np.argsort(dists, axis=1)  # (N, M) closest-first
        out = []
        for i in range(n):
            row = order[i]
            out.append(([int(self.ids[j]) for j in row],
                        [float(dists[i, j]) for j in row]))
        return out

    # -- mutation ----------------------------------------------------------
    def update(self, img: np.ndarray, box, runner_id: int) -> None:
        """EMA-blend the crop's embedding into ``runner_id``'s row.

        Inserts a new row if the runner is not yet in the bank.
        """
        runner_id = int(runner_id)
        feat = self.extract(img, box)
        idx = np.where(self.ids == runner_id)[0]
        if idx.size:
            row = idx[0]
            self.feats[row] = _ema_update(self.feats[row], feat, self.alpha)
        else:
            feat = _l2_normalize(feat).reshape(1, -1)
            if self.feats.size == 0:
                self.feats = feat.astype(np.float32)
            else:
                if self.feats.shape[1] != feat.shape[1]:
                    raise ValueError(
                        f"Feature dim mismatch: bank has {self.feats.shape[1]}, "
                        f"new is {feat.shape[1]}."
                    )
                self.feats = np.vstack([self.feats, feat]).astype(np.float32)
            self.ids = np.append(self.ids, runner_id).astype(np.int64)

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=self.ids, feats=self.feats)
        logger.info("Saved ReIDBank (%d runners) to %s", len(self.ids), path)

    @classmethod
    def load(cls, path, extractor: ReID, alpha: float = DEFAULT_EMA_ALPHA) -> "ReIDBank":
        """Load a bank from ``.npz``; returns an empty bank if the file is absent."""
        path = pathlib.Path(path)
        if not path.exists():
            logger.warning("No reid bank at %s, initializing empty bank.", path)
            return cls(extractor, alpha=alpha)
        data = np.load(path, allow_pickle=False)
        return cls(extractor, ids=data["ids"], feats=data["feats"], alpha=alpha)

    # -- durability: regenerate from human annotations ---------------------
    @classmethod
    def rebuild_from_annotations(
        cls,
        store,
        frame_getter: Callable[[int], Optional[np.ndarray]],
        extractor: ReID,
        min_size: int = 24,
        alpha: float = DEFAULT_EMA_ALPHA,
    ) -> "ReIDBank":
        """Rebuild the bank by extracting features from every human annotation.

        Parameters
        ----------
        store : SQLiteAnnotationStore
            Provides ``load_all_annotations(source="human")``. Boxes come back as
            ``(N, 7)`` = ``[xc, yc, w, h, runner_id, conf, cls]`` normalized to
            [0, 1] when ``img_shape`` is not supplied.
        frame_getter : callable(frame_number) -> BGR image or None
            Returns the full-frame image for a given frame number (e.g. wrapping
            a Loader's ``seek_frame`` + ``__next__``). Return None to skip.
        min_size : int
            Skip crops smaller than this in either dimension (pixels).
        """
        bank = cls(extractor, alpha=alpha)
        annotations = store.load_all_annotations(img_shape=None, source="human")
        n_frames = 0
        n_crops = 0
        for frame_num in sorted(annotations):
            boxes = annotations[frame_num]["boxes"]
            if boxes is None or boxes.size == 0:
                continue
            img = frame_getter(frame_num)
            if img is None:
                continue
            n_frames += 1
            h, w = img.shape[:2]
            xc, yc, bw, bh = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
            ids = boxes[:, 4].astype(int)
            for bx1, by1, bx2, by2, rid in zip(x1, y1, x2, y2, ids):
                if rid < 0:
                    continue
                if (bx2 - bx1) < min_size or (by2 - by1) < min_size:
                    continue
                bank.update(img, np.array([[bx1, by1, bx2, by2]], dtype=np.float32), int(rid))
                n_crops += 1
        logger.info(
            "Rebuilt ReIDBank from %d annotated frames, %d crops, %d runners.",
            n_frames, n_crops, len(bank.ids),
        )
        return bank
