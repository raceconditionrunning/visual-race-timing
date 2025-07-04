#!/usr/bin/env python
import argparse
import pathlib
import subprocess

import tqdm
from timecode import Timecode

from visual_race_timing.drawing import render_timecode_pil
from visual_race_timing.loader import ImageLoader, VideoLoader, FFmpegVideoLoader
from visual_race_timing.logging import get_logger
from visual_race_timing.video import get_video_metadata

logger = get_logger(__name__)


def run(args):
    if len(args.source) == 1 and args.source[0].is_dir():
        loader = ImageLoader(args.source[0])
        is_10bit = False
        pix_fmt = "rgb24"
        vid_metadata = {'color_space': 'bt709'}
        frame_rate = "30"  # Images are assumed to be some slightly variable low fps, which we'll retarget below
    else:
        # First detect if we need 10-bit processing
        vid_metadata = get_video_metadata(args.source[0])['streams'][0]
        is_10bit = vid_metadata.get('pix_fmt', '') == 'yuv420p10le'

        if is_10bit:
            # Use FFmpeg-based loader for 10-bit videos
            logger.info("Using FFmpeg video loader for 10-bit processing")
            loader = FFmpegVideoLoader(args.source)
            is_10bit = True  # Ensure we use 10-bit pipeline
        else:
            # Use standard OpenCV-based loader for 8-bit videos
            logger.info("Using standard video loader for 8-bit processing")
            loader = VideoLoader(args.source)
        frame_rate = str(loader.get_current_time().framerate)

    end_tc = None
    if args.range:
        # It's a list, remove the brackets
        args.range = args.range.strip('[]')
        # Parse the timecode range
        start_tc, end_tc = args.range.split(',')
        start_tc = Timecode(loader.get_current_time().framerate, start_tc)
        end_tc = Timecode(loader.get_current_time().framerate, end_tc)
        loader.seek_timecode(start_tc)

    width, height = loader.get_image_size()

    # Configure FFmpeg based on bit depth
    if is_10bit:
        input_pix_fmt = 'rgb48le'  # 16-bit RGB for 10-bit processing
        output_pix_fmt = 'yuv420p10le'  # 10-bit output
        profile_args = ['-profile:v', 'main10']
        logger.info("Using 10-bit encoding pipeline")
        convert_to_16bit = False  # FFmpegVideoLoader already provides 16-bit
    else:
        input_pix_fmt = 'bgr24'  # 8-bit BGR from OpenCV
        output_pix_fmt = 'yuv420p'  # 8-bit output
        profile_args = []
        convert_to_16bit = False
        logger.info("Using 8-bit encoding pipeline")

    if args.output.is_file():
        logger.error(f"Output file {args.output} already exists. Please specify a new output file.")
        exit(1)

    ffmpeg_cmd = [
                     'ffmpeg',
                     '-f', 'rawvideo',
                     '-vcodec', 'rawvideo',
                     '-s', f'{width}x{height}',  # Frame size
                     '-pix_fmt', input_pix_fmt,
                     '-r', frame_rate,  # Frame rate
                     '-i', '-',  # Read from stdin
                     '-c:v', args.codec,  # Video codec
                 ] + profile_args + [  # Add profile for 10-bit if needed
                     '-preset', args.preset,  # Encoding preset
                     '-crf', str(args.crf),  # Quality setting
                     '-pix_fmt', output_pix_fmt,
        '-tag:v', 'hvc1',  # QuickTime-compatible HEVC tag
        '-color_primaries', vid_metadata.get('color_primaries', 'bt709'),
        # Default to 'bt709' if not specified
        '-colorspace', vid_metadata['color_space'],
        '-color_range', vid_metadata.get('color_range', 'tv'),  # Default to 'tv' if not specified
        '-timecode', str(Timecode(frame_rate, start_seconds=loader.get_current_time().float)),
        '-movflags', '+faststart',
        # Set start timecode

                     str(args.output)
                 ]

    logger.info(f"Starting ffmpeg with command: {' '.join(ffmpeg_cmd)}")

    try:
        # Start ffmpeg process
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        frame_count = 0
        duplicated_frames = 0
        for path, frames, metadata in tqdm.tqdm(loader):
            timecode = metadata[0][0]
            if end_tc and timecode >= end_tc:
                break

            out = render_timecode_pil(metadata[0][0], frames[0])
            # import cv2
            # cv2.imshow('frame', (np.array(out/257).astype(np.uint8)))

            # cv2.waitKey(1)  # Display the frame for a brief moment
            try:
                ffmpeg_process.stdin.write(out.tobytes())
                # Retarget the frame rate to match the desired output frame rate
                current_frame_timecode = Timecode(frame_rate, start_seconds=timecode.float)
                next_frame_timecode = loader.get_current_time()
                while next_frame_timecode and next_frame_timecode.float > (current_frame_timecode + 1).float:
                    # We need to duplicate frames to match the frame rate
                    duplicated_frames += 1
                    ffmpeg_process.stdin.write(out.tobytes())
                    current_frame_timecode += 1

                frame_count += 1


            except BrokenPipeError:
                logger.error("FFmpeg process terminated unexpectedly")
                break

        # Close stdin to signal end of input
        ffmpeg_process.stdin.close()

        # Wait for ffmpeg to finish and get return code
        stderr_output = ffmpeg_process.stderr.read().decode()
        return_code = ffmpeg_process.wait()

        if return_code == 0:
            logger.info(
                f"Successfully encoded {frame_count} frames to {args.output} (with {duplicated_frames} duplicated frames)")
        else:
            logger.error(f"FFmpeg failed with return code {return_code}")
            logger.error(f"FFmpeg stderr: {stderr_output}")

    except Exception as e:
        logger.error(f"Error during video encoding: {e}")
        if 'ffmpeg_process' in locals() and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
        raise


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True,
                        help='Output video file path')
    parser.add_argument('--source', type=pathlib.Path, nargs='+', required=True,
                        help='file paths')
    # Accept a timecode range to optionally trim the video
    parser.add_argument('--range', type=str, default=None,
                        help='Timecode range to trim the video, e.g. "00:00:00:00,00:01:00:00"')
    parser.add_argument('--codec', type=str, default='libx265',
                        help='Video codec (default: libx265)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower',
                                 'veryslow'],
                        help='Encoding preset (default: medium)')
    parser.add_argument('--crf', type=int, default=24,
                        help='Constant Rate Factor for quality (lower = better quality, default: 24)')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
