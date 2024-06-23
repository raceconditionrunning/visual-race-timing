import pathlib
import subprocess
import json
from typing import Optional, Tuple, List

from timecode import Timecode


def get_video_metadata(video_path: pathlib.Path) -> dict:
    # Run ffprobe to get the metadata
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error', '-show_entries',
            'stream=width,height,r_frame_rate,avg_frame_rate,codec_type:stream_tags=timecode', '-of', 'json', str(video_path)
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


def crop_videos(sources: List[pathlib.Path], crop: Tuple[int, int, int, int]) -> List[pathlib.Path]:
    processes = []
    crop_arg = f'{crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}'
    # Colons mess up basic resource parsing in ultralytics, so we'll dashes/underscores
    crop_filename_str = f'{crop[0]}-{crop[1]}_{crop[2]}-{crop[3]}'
    # We can only crop files
    sources = sources if isinstance(sources, list) else [sources]
    sources = [pathlib.Path(source) for source in sources]
    crop_paths = []
    # Start n shells to crop all sources
    for source in sources:
        # Store the cropped file in the same directory as the source
        crop_path = pathlib.Path(source).parent / f'{pathlib.Path(source).stem}_crop_{crop_filename_str}.mp4'
        crop_paths.append(crop_path)
        if crop_path.exists():
            continue
        crop_cmd = f'ffmpeg -i {str(source)} -vf "crop={crop_arg}" -c:a copy {str(crop_path)}'
        proc = subprocess.Popen(crop_cmd, shell=True)
        processes.append(proc)

    for proc in processes:
        proc.wait()
    return crop_paths
