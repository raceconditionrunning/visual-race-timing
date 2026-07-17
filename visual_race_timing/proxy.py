import pathlib
from typing import Optional, Tuple

from visual_race_timing.video import get_video_height_width

Crop = Tuple[int, int, int, int]  # w, h, x, y — ffmpeg-style, original pixel space


def canonical_crop(source: pathlib.Path, crop: Optional[Crop]) -> Crop:
    """Resolve crop to explicit (w,h,x,y) in original pixel space, defaulting to full frame."""
    if crop is not None:
        return tuple(crop)
    height, width = get_video_height_width(source)
    return width, height, 0, 0


def proxy_path_for(source: pathlib.Path, crop: Optional[Crop], scale: float,
                    proxy_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Deterministic path for the proxy matching (source, crop, scale). Pure naming, no I/O to check existence."""
    w, h, x, y = canonical_crop(source, crop)
    proxy_dir = proxy_dir or source.parent / "proxies"
    filename = f"{source.stem}__crop{w}x{h}+{x}+{y}__scale{scale:.2f}{source.suffix}"
    return proxy_dir / filename


def find_proxy(source: pathlib.Path, crop: Optional[Crop], scale: float,
               proxy_dir: Optional[pathlib.Path] = None) -> Optional[pathlib.Path]:
    """Return the proxy path if it exists on disk and isn't older than the source, else None.

    Naming is content-blind, so a stale proxy from a re-exported source with the
    same filename would otherwise be served silently — the mtime check catches
    the common case (source re-transcoded after the proxy was made).
    """
    path = proxy_path_for(source, crop, scale, proxy_dir)
    if not path.is_file():
        return None
    if source.stat().st_mtime > path.stat().st_mtime:
        return None
    return path
