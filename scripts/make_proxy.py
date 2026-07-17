#!/usr/bin/env python3
"""Generate cropped/scaled proxy media for faster interactive scrubbing in annotate.py.

    uv run python scripts/make_proxy.py --crop 1152 576 460 1171 data/DM26/DJI_20260606051302_0005_D.MP4
"""
import argparse
import pathlib
import subprocess
from fractions import Fraction

from visual_race_timing.logging import get_logger
from visual_race_timing.proxy import canonical_crop, proxy_path_for
from visual_race_timing.video import get_video_metadata

logger = get_logger(__name__)


def _round_even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)


def make_proxy(source: pathlib.Path, crop, scale: float, proxy_dir, crf: int, force: bool) -> pathlib.Path:
    w, h, x, y = canonical_crop(source, crop)
    out_path = proxy_path_for(source, crop, scale, proxy_dir)

    if out_path.is_file() and not force:
        logger.info(f"Proxy already exists, skipping: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(source)
    video_stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    fps_str = video_stream.get("avg_frame_rate") or video_stream["r_frame_rate"]
    timecode = video_stream.get("tags", {}).get("timecode")

    filters = [f"crop={w}:{h}:{x}:{y}"]
    if scale != 1.0:
        filters.append(f"scale={_round_even(w * scale)}:{_round_even(h * scale)}")

    # Short, fixed GOP: VideoLoader.seek_timecode/frame-step keys rely on cheap
    # seeks, which decode forward from the last keyframe — a long GOP would
    # quietly defeat the point of the proxy. sc_threshold=0 stops ffmpeg from
    # inserting extra keyframes at scene cuts, which would otherwise make GOP
    # length (and thus seek cost) unpredictable.
    gop = round(float(Fraction(fps_str)))

    cmd = [
        "ffmpeg", "-y" if force else "-n",
        "-i", str(source),
        "-vf", ",".join(filters),
        "-r", fps_str,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-g", str(gop), "-sc_threshold", "0",
        "-an",
    ]
    if timecode:
        cmd.extend(["-timecode", timecode])
    cmd.append(str(out_path))

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path, nargs="+", help="source video file paths")
    parser.add_argument("--crop", type=int, nargs=4, default=None, metavar=("W", "H", "X", "Y"),
                        help="crop rect in original pixel space: w h x y (default: full frame)")
    parser.add_argument("--scale", type=float, default=1.0, help="scale factor applied after crop")
    parser.add_argument("--proxy-dir", type=pathlib.Path, default=None,
                        help="directory to write proxies to (default: <source_dir>/proxies)")
    parser.add_argument("--crf", type=int, default=18, help="libx264 crf (lower = higher quality)")
    parser.add_argument("--force", action="store_true", help="regenerate even if a matching proxy already exists")
    return parser.parse_args()


def main():
    args = parse_opt()
    for source in args.source:
        out_path = make_proxy(source, args.crop, args.scale, args.proxy_dir, args.crf, args.force)
        logger.info(f"Proxy ready: {out_path}")


if __name__ == "__main__":
    main()
