import pathlib
import subprocess
import json
from typing import Optional, Tuple, List

from joblib import Memory
from timecode import Timecode

# Create joblib memory cache
CACHE_DIR = ".cache"
memory = Memory(str(CACHE_DIR), verbose=0)


@memory.cache
def get_video_metadata(video_path: pathlib.Path) -> dict:
    # Run ffprobe to get the metadata
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error', '-show_entries',
            'stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,pix_fmt,color_space,color_primaries,color_trc,color_range,codec_type:stream_tags=timecode', '-of', 'json', str(video_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Parse the JSON output
    data = json.loads(result.stdout)
    return data


def get_video_height_width(video_path: pathlib.Path) -> Optional[Tuple[int, int]]:
    metadata = get_video_metadata(video_path)
    for stream in metadata['streams']:
        if stream['codec_type'] == 'video':
            return stream['height'], stream['width']
    return None


@memory.cache
def get_timecode(file_path: pathlib.Path) -> Optional[Timecode]:
    metadata = get_video_metadata(file_path)
    timecode = None
    fps = None

    for stream in metadata['streams']:
        if 'timecode' in stream.get('tags', {}):
            timecode = stream['tags']['timecode']

        if stream['codec_type'] == 'video':
            # Extract frame rate (prefer avg_frame_rate, fallback to r_frame_rate)
            new_fps = stream.get('avg_frame_rate', stream.get('r_frame_rate'))
            if new_fps and new_fps != '0/0':
                fps = new_fps

    return Timecode(fps, timecode) if timecode and fps else None
