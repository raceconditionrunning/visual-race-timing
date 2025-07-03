import bisect
import datetime
import math
import pathlib
import os
import subprocess
from typing import List

import cv2
import exif
import numpy as np
from sortedcontainers import SortedDict
from timecode import Timecode
from tqdm import tqdm

from visual_race_timing.video import get_timecode, get_video_height_width, get_video_metadata

from visual_race_timing.logging import get_logger

logger = get_logger(__name__)


class Loader:
    def __init__(self, paths, batch=1, crop=None):
        self.files = paths
        self.bs = batch
        self.source_index = 0
        self.crop = crop

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def get_image_dims(self):
        raise NotImplementedError("get_image_dims must be implemented by subclass")

    def get_image_size(self):
        return tuple(reversed(self.get_image_dims()))

    def get_current_time(self) -> Timecode:
        raise NotImplementedError("get_current_time must be implemented by subclass")

    def get_current_frame(self) -> int:
        raise NotImplementedError("get_current_frame must be implemented by subclass")

    def seek_timecode(self, timecode: Timecode) -> bool:
        raise NotImplementedError("seek_timecode must be implemented by subclass")

    def seek_time(self, time_str: str) -> bool:
        raise NotImplementedError("seek_time must be implemented by subclass")

    def seek_frame(self, target_frame: int) -> bool:
        raise NotImplementedError("seek_frame must be implemented by subclass")

    def seek_timecode_frame(self, target_frame: int) -> bool:
        raise NotImplementedError("seek_timecode_frame must be implemented by subclass")


class ImageLoader(Loader):
    def __init__(self, frame_directory, batch=1, crop=None):
        super().__init__(frame_directory, batch, crop=crop)
        if (frame_directory / "timecode_index").is_dir():
            self.time_to_path = load_timecode_index(frame_directory / "timecode_index")
        else:
            frame_times = get_capture_times_from_exif(frame_directory)
            (frame_directory / "timecode_index").mkdir()
            self.time_to_path = create_timecode_symlinks(frame_times, (frame_directory / "timecode_index"))
            # Get seconds from start of day
            first_frame_date = list(self.time_to_path.keys())[0]
            first_frame_time = first_frame_date - datetime.datetime(first_frame_date.year, first_frame_date.month,
                                                                    first_frame_date.day)
            self.start_timecode = Timecode(1000, start_seconds=first_frame_time.total_seconds())

        first_frame_date = list(self.time_to_path.keys())[0]
        self.files = list(self.time_to_path.values())
        dates = list(self.time_to_path.keys())
        all_as_seconds = [(time - datetime.datetime(first_frame_date.year, first_frame_date.month,
                                                    first_frame_date.day)).total_seconds() for time in dates]
        self._timecodes = [Timecode(1000, start_seconds=seconds) for seconds in all_as_seconds]
        self._source_dims = [cv2.imread(self.files[0]).shape[:2]]

    def __next__(self):
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.source_index >= len(self.files):
                if len(imgs) > 0:
                    return paths, imgs, info
                else:
                    raise StopIteration
            path = self.files[self.source_index]

            timecode = self._timecodes[self.source_index]
            frame_path = path

            img = cv2.imread(str(frame_path))
            if self.crop:
                img = img[self.crop[3]:self.crop[3] + self.crop[1], self.crop[2]:self.crop[2] + self.crop[0]]
            paths.append(frame_path)
            imgs.append(img)
            info.append([timecode, self.source_index, f"image {self.source_index + 1}/{len(self._timecodes)}: "])
            self.source_index += 1
        return paths, imgs, info

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(len(self.time_to_path) / self.bs)  # number of files

    def get_image_dims(self):
        return self._source_dims[0]

    def get_current_time(self) -> Timecode:
        return self._timecodes[self.source_index]

    def get_current_frame(self) -> int:
        return self.source_index

    def seek_timecode_frame(self, target_frame: int):
        target_timecode = Timecode(self._timecodes[0].framerate, frames=target_frame)
        return self.seek_timecode(target_timecode)

    def seek_timecode(self, target_timecode) -> bool:
        i = bisect.bisect_left(self._timecodes, target_timecode)
        if i != len(self._timecodes):
            self.source_index = i
            return True
        else:
            return False

    def seek_time(self, time_str: str) -> bool:
        target_time = Timecode(self._timecodes[0].framerate, time_str)
        return self.seek_timecode(target_time)

    def seek_frame(self, target_frame):
        if not (0 <= target_frame < len(self._timecodes)):
            return False
        self.source_index = target_frame
        return True


class VideoLoader(Loader):
    def __init__(self, paths: List[pathlib.Path], batch=1, vid_stride=1, crop=None):
        self._timecodes = [get_timecode(source) for source in paths]
        # Sort sources by timecode
        paths = [source for _, source in sorted(zip(self._timecodes, paths))]
        self._timecodes = [get_timecode(source) for source in paths]
        super().__init__(paths, batch, crop=crop)
        self._source_dims = [get_video_height_width(source) for source in paths]
        assert all(
            [dim == self._source_dims[0] for dim in self._source_dims]), "All videos must have the same dimensions"
        assert all([timecode.framerate == self._timecodes[0].framerate for timecode in self._timecodes]), \
            "All videos must have the same framerate"
        self.fps = self._timecodes[0].framerate  # Framerate of the first video
        self._frame_lengths = [int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_COUNT)) for source in paths]
        end_timecodes = [timecode + frame_length for timecode, frame_length in
                         zip(self._timecodes, self._frame_lengths)]
        # Calculate gap between videos (if any)
        self.gaps = []
        for i in range(len(end_timecodes) - 1):
            gap = self._timecodes[i + 1].frame_number - end_timecodes[i].frame_number
            self.gaps.append(gap)

        self.num_sources = len(paths)
        self.vid_stride = vid_stride  # video frame-rate stride

        self._new_video(self.files[0])  # new video
        # Index across all sources. The next frame grabbed will have this index.
        self._current_frame = 0
        # Index into the current video source. The next frame grabbed will have this index.
        self._source_frame = 0
        # How many gap frames we have already skipped in the current source
        self.gap_offset = 0
        self.frames_to_stride = self.bs

    def __next__(self):
        """Returns the next batch of images along with their paths and metadata."""
        logger.debug(
            f"User wants next frame. Expected next frame: {self._current_frame} ({self._timecodes[0] + self._current_frame}) "
            f"source_index: {self.source_index},  source_frame: {self._source_frame}, gap_offset: {self.gap_offset}, frames_to_stride: {self.frames_to_stride}")
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.source_index >= self.num_sources:  # end of file list
                if len(imgs) > 0:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.source_index]

            if not self.cap or not self.cap.isOpened():
                self._new_video(path)

            try:
                # source_frame is synced with the capture, so it is the frame that will be retrieved next.
                start_frame = self._timecodes[self.source_index].frames - self._timecodes[
                    0].frames + self._source_frame + self.gap_offset
                # This was updated the last time we retrieved a frame. Our goal is to seek so that we can call retrieve and grab this frame.
                next_frame = self._current_frame
                # Check which source the next frame belongs to. Bisect right to fall to the next source (right) if the frame matches the start timecode
                next_frame_source = bisect.bisect_right(self._timecodes, self._timecodes[0].frames + next_frame) - 1
                if next_frame_source != self.source_index:
                    raise (RuntimeError(
                        f"Reached end of video {path} at frame {self._source_frame + self.gap_offset}."))
                end_of_source_frame = (self._timecodes[next_frame_source] + self._frame_lengths[next_frame_source] -
                                       self._timecodes[0]).frames
                start_of_next_source_frame = self._timecodes[next_frame_source + 1].frames - self._timecodes[
                    0].frames if next_frame_source + 1 < self.num_sources else float('inf')
                # Current frame is the true index which tells us if we are in a gap or not. If we go forward, what's our new offset?
                if end_of_source_frame <= next_frame < start_of_next_source_frame:
                    # For gap frames, we need to manually set the position to the last frame of the current source
                    if self._source_frame != self._frame_lengths[self.source_index]:
                        # A seek operation may have already set up the source target correctly, check and move if not
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_lengths[next_frame_source])
                else:
                    # We need to grab atleast one frame to get the next frame
                    for _ in range(next_frame - start_frame + 1):
                        success = self.cap.grab()
                        if not success:
                            raise RuntimeError(f"Failed to grab frame {next_frame}")

                self._source_frame = int(cv2.VideoCapture.get(self.cap,
                                                              cv2.CAP_PROP_POS_FRAMES))  # NOTE: Doesn't increment past the last frame of the video, so we need to increment it manually

                success, im0 = self.cap.retrieve()
                if success:
                    frame_timecode = Timecode(self.get_fps(), frames=self._timecodes[0].frames) + self._current_frame
                    paths.append(path)
                    # NOTE: Because the cap won't increment past the last frame of the video, the source_frame counter won't increment when retrieving the last frame.
                    # when that happens, this calculated value will be off by one.
                    # alt_frame_timecode = self._timecodes[self.source_index] + self._source_frame - 1 + self.gap_offset
                    if self.crop:
                        # given in ffmpeg format: width:height:x:y
                        im0 = im0[self.crop[3]:self.crop[3] + self.crop[1], self.crop[2]:self.crop[2] + self.crop[0]]
                    imgs.append(im0)
                    frame_length = self._frame_lengths[self.source_index]
                    info.append([frame_timecode, self._current_frame,
                                 f"video {self.source_index + 1}/{self.num_sources} (frame {self._source_frame}/{frame_length}) {path}: "])
                    # Note that source frame is for the _next_ frame
                    self._current_frame = next_frame + self.frames_to_stride
                    if end_of_source_frame <= self._current_frame < start_of_next_source_frame:
                        self.gap_offset = start_of_next_source_frame - self._current_frame
                else:
                    raise RuntimeError(f"Failed to retrieve frame from {path} at frame {self._source_frame}.")
            except RuntimeError:
                # Move to the next file if the current video ended or failed to open
                self.source_index += 1
                self.gap_offset = 0
                if self.cap:
                    self.cap.release()
                if self.source_index < self.num_sources:
                    self._new_video(self.files[self.source_index])

        logger.debug(
            f"Completed batch. Expected next frame: {self._current_frame} ({self._timecodes[0] + self._current_frame}) "
            f"source_index: {self.source_index},  source_frame: {self._source_frame}, gap_offset: {self.gap_offset}, frames_to_stride: {self.frames_to_stride}")
        return paths, imgs, info

    def get_image_dims(self):
        return self._source_dims[0]

    def get_fps(self) -> str:
        return self.fps

    def get_current_time(self) -> Timecode:
        """ Returns the timecode that would be returned by the next call to __next__. Includes real frames and gaps."""
        # Note that this requires gaps to have been accounted for in the current frame calculation
        return self._timecodes[0] + self.get_current_frame()

    def get_current_frame(self) -> int:
        """
        Returns the index of the frame that would be returned by the next call to __next__. This counter increments and includes real frames and gaps.
        """
        return self._current_frame

    def seek_time(self, time_str: str) -> bool:
        target_time = Timecode(self._timecodes[0].framerate, time_str)
        return self.seek_timecode(target_time)

    def _new_video(self, path):
        """Creates a new video capture object for the given path."""
        self._source_frame = 0
        self.cap = cv2.VideoCapture(path)
        # Try to preserve native format. Unfortunately, we can't read yuv420p10le directly with OpenCV, so these won't work.
        #self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)  # Don't convert to RGB
        #self.cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_16UC3)  # Request 16-bit format

        self.gap_offset = 0
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")

    def seek_frame(self, target_frame) -> bool:
        """
        Look for the nth frame of all sources combined.
        """
        logger.debug(f"Seeking to frame {target_frame} ({self._timecodes[0] + target_frame}) across all sources.")
        return self.seek_timecode(self._timecodes[0] + target_frame)

    def seek_timecode(self, target_timecode):
        """Seek to a specific timecode across all sources."""
        if target_timecode < self._timecodes[0] or target_timecode > self._timecodes[-1] + self._frame_lengths[-1]:
            return False

        if target_timecode == self._timecodes[0] + self._current_frame:
            # If we are already at the target timecode, no need to seek
            return True
        # Forward seek optimization
        if target_timecode > self._timecodes[0] + self._current_frame:
            frame_diff = (target_timecode - self._timecodes[0]).frames - self._current_frame
            if 2 <= frame_diff < 5:
                while frame_diff > 1:
                    self.__next__()
                    frame_diff -= 1
                return True

        # Find the source containing this timecode
        source_index = bisect.bisect_right(self._timecodes, target_timecode) - 1

        if source_index is None:
            return False

        # Switch to new source if needed
        if source_index != self.source_index:
            self.source_index = source_index
            self._new_video(self.files[self.source_index])

        # Calculate frame position within the source and seek
        source_start_timecode = self._timecodes[self.source_index]
        source_length = self._frame_lengths[self.source_index]
        if target_timecode == source_start_timecode:
            # If the target timecode is exactly the start of the source, we can seek to frame 0
            target_source_frame = 0
            self.gap_offset = 0
        else:
            target_source_frame = (target_timecode - source_start_timecode).frames
            if target_source_frame >= source_length:
                # This must be a gap, so we need to adjust the target frame
                self.gap_offset = target_source_frame - source_length
                # FIXME: The last index should be length - 1, but for some reason the last frame is actually at length instead? Why?
                target_source_frame = source_length  # Last frame of the current source
            else:
                self.gap_offset = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_source_frame)
        self._source_frame = target_source_frame
        # Update current frame position including gaps
        self._current_frame = self._timecodes[self.source_index].frames - self._timecodes[
            0].frames + target_source_frame + self.gap_offset
        # Verify seek was successful
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == target_source_frame

    def seek_timecode_frame(self, target_frame):
        target_timecode = Timecode(self.get_fps(), frames=target_frame)
        self.seek_timecode(target_timecode)

    def __len__(self):
        """Returns the number of batches in the object."""
        total_frames = sum(self._frame_lengths) + sum(self.gaps)
        return math.ceil(total_frames / self.bs)


class FFmpegVideoLoader(Loader):
    """Custom video loader for 10-bit videos using FFmpeg directly with gap handling"""

    def __init__(self, video_paths, batch=1, vid_stride=1, crop=None, **kwargs):
        super().__init__(video_paths, batch, crop=crop, **kwargs)
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]

        # Initialize timecodes and sort videos by timecode (matching original behavior)
        self._timecodes = [get_timecode(source) for source in self.video_paths]
        self.video_paths = [source for _, source in sorted(zip(self._timecodes, self.video_paths))]
        self._timecodes = [get_timecode(source) for source in self.video_paths]

        # Validate all videos have same dimensions and framerate
        self._source_dims = [get_video_height_width(source) for source in self.video_paths]
        assert all(
            [dim == self._source_dims[0] for dim in self._source_dims]), "All videos must have the same dimensions"
        assert all([timecode.framerate == self._timecodes[0].framerate for timecode in self._timecodes]), \
            "All videos must have the same framerate"

        self.fps = self._timecodes[0].framerate
        self.vid_stride = vid_stride

        # Get video info and frame lengths
        self._get_video_info()
        self._frame_lengths = []
        for video_path in self.video_paths:
            self._frame_lengths.append(int(get_video_metadata(video_path)["streams"][0]["nb_frames"]))

        # Calculate gaps between videos (matching original logic)
        end_timecodes = [timecode + frame_length for timecode, frame_length in
                         zip(self._timecodes, self._frame_lengths)]
        self.gaps = []
        for i in range(len(end_timecodes) - 1):
            gap = self._timecodes[i + 1].frame_number - end_timecodes[i].frame_number
            self.gaps.append(gap)

        self.num_sources = len(self.video_paths)

        # State tracking (matching original)
        self.source_index = 0
        self.current_video_idx = 0  # Keep for compatibility
        self._current_frame = 0  # Index across all sources
        self._source_frame = 0  # Index into current video source
        self.gap_offset = 0  # Gap frames skipped in current source
        self.frames_to_stride = batch

        self.ffmpeg_process = None
        self._start_current_video()
        self._last_frame = None  # Cache for gap handling

    def _get_video_info(self):
        """Get video properties using ffprobe"""
        video_path = self.video_paths[0]  # Use first video for properties

        info = get_video_metadata(video_path)["streams"][0]

        self.fps = info["r_frame_rate"]
        self.pix_fmt = info["pix_fmt"]


        # Set bytes per pixel based on format
        if '10' in self.pix_fmt or 'p10' in self.pix_fmt:
            self.output_fmt = 'rgb48le'  # 16-bit RGB
            self.bytes_per_pixel = 6
            self.is_10bit = True
        else:
            self.output_fmt = 'rgb24'  # 8-bit RGB
            self.bytes_per_pixel = 3
            self.is_10bit = False

    def _start_current_video(self, start_frame=0):
        """Start FFmpeg process for current video with optional frame offset"""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process = None

        if self.source_index >= len(self.video_paths):
            return

        video_path = self.video_paths[self.source_index]
        cmd = ['ffmpeg', '-i', str(video_path)]

        if start_frame > 0:
            # Use select filter for frame-accurate positioning
            # This will decode and select frames starting from start_frame
            remaining_frames = self._frame_lengths[self.source_index] - start_frame
            cmd.extend(['-vf', f'select=gte(n\\,{start_frame})'])
            cmd.extend(['-frames:v', str(remaining_frames)])
            cmd.extend(['-vsync', '0'])  # Prevent frame duplication/dropping
        else:
            cmd.extend(['-frames:v', str(self._frame_lengths[self.source_index])])

        cmd.extend(['-f', 'rawvideo', '-pix_fmt', self.output_fmt, '-'])

        self.ffmpeg_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Starting FFmpeg process for {video_path} with command: {' '.join(cmd)}")

        self._source_frame = start_frame

    def get_current_time(self):
        """Return current timecode (compatible with existing interface)"""
        return self._timecodes[0] + self._current_frame

    def get_current_frame(self):
        """Returns the index of the frame that would be returned by the next call to __next__"""
        return self._current_frame

    def get_image_dims(self):
        """Alias for compatibility"""
        return self._source_dims[0]

    def get_fps(self):
        """Return framerate"""
        return self.fps

    def seek_timecode(self, target_timecode):
        """Optimized seeking using FFmpeg's built-in capabilities"""
        if target_timecode < self._timecodes[0] or target_timecode > self._timecodes[-1] + self._frame_lengths[-1]:
            return False

        # Find target source
        source_index = bisect.bisect_right(self._timecodes, target_timecode) - 1
        if source_index < 0:
            return False

        # Calculate frame position within source
        source_start_timecode = self._timecodes[source_index]
        target_source_frame = target_timecode.frames - source_start_timecode.frames

        # Switch source if needed and seek directly to target frame
        if source_index != self.source_index:
            self.source_index = source_index

        self._start_current_video(target_source_frame)

        # Update positions
        self._current_frame = target_timecode.frames - self._timecodes[0].frames
        self._source_frame = target_source_frame

        return True

    def seek_frame(self, target_frame):
        """Seek to nth frame across all sources"""
        return self.seek_timecode(self._timecodes[0] + target_frame)

    def seek_time(self, time_str):
        """Seek to time string"""
        target_time = Timecode(self._timecodes[0].framerate, time_str)
        return self.seek_timecode(target_time)

    def _read_frame(self):
        """Read a single frame from FFmpeg"""
        if not self.ffmpeg_process:
            return False, None

        frame_size = self._source_dims[0][0] * self._source_dims[0][1] * self.bytes_per_pixel
        frame_data = self.ffmpeg_process.stdout.read(frame_size)

        if len(frame_data) != frame_size:
            return False, None

        if self.bytes_per_pixel == 6:  # 16-bit
            dtype = np.uint16
        else:  # 8-bit
            dtype = np.uint8

        frame_np = np.frombuffer(frame_data, dtype=dtype)
        frame_np = frame_np.reshape((*self._source_dims[0], 3))

        # Create cv::Mat from numpy array
        frame = cv2.Mat(frame_np)

        # Convert RGB to BGR
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Apply crop if specified
        if self.crop:
            # Crop format: width:height:x:y
            frame = frame[self.crop[3]:self.crop[3] + self.crop[1],
                    self.crop[2]:self.crop[2] + self.crop[0]]

        self._source_frame += 1
        self._last_frame = frame  # Cache the frame for gap handling
        return True, frame

    def __next__(self):
        """Returns the next batch of images along with their paths and metadata (matching original interface)"""
        logger.debug(
            f"User wants next frame. Expected next frame: {self._current_frame} ({self._timecodes[0] + self._current_frame}) "
            f"source_index: {self.source_index}, source_frame: {self._source_frame}, gap_offset: {self.gap_offset}")

        paths, imgs, info = [], [], []

        while len(imgs) < self.frames_to_stride:
            if self.source_index >= self.num_sources:
                if len(imgs) > 0:
                    return paths, imgs, info
                else:
                    raise StopIteration

            # Check if we're in a gap
            next_frame = self._current_frame
            next_frame_source = bisect.bisect_right(self._timecodes, self._timecodes[0].frames + next_frame) - 1

            if next_frame_source != self.source_index:
                # Move to next source
                self.source_index += 1
                self.gap_offset = 0
                if self.source_index < self.num_sources:
                    self._start_current_video()
                continue

            # Subtle indexing; frames are 1 indexed
            if self.source_index == self.num_sources - 1 and next_frame >= self._timecodes[next_frame_source].frames + \
                    self._frame_lengths[next_frame_source] - 1:
                # Reached the end of the last source, no more frames to read
                raise StopIteration
            else:
                # Calculate frame boundaries
                end_of_source_frame = (self._timecodes[next_frame_source] +
                                       self._frame_lengths[next_frame_source] -
                                       self._timecodes[0]).frames
                start_of_next_source_frame = (self._timecodes[next_frame_source + 1].frames - self._timecodes[0].frames if
                                              next_frame_source + 1 < self.num_sources else float('inf'))

                # Handle gap frames
                if end_of_source_frame <= next_frame < start_of_next_source_frame:
                    # This is a gap frame - use last frame of current source
                    if self._source_frame < self._frame_lengths[self.source_index]:
                        # Need to read remaining frames to get to the end
                        while self._source_frame <= self._frame_lengths[self.source_index]:
                            success, frame = self._read_frame()
                            if not success:
                                break
                        # Use the last successfully read frame
                    else:
                        # Already at end, use cached last frame
                        if self._last_frame is not None:
                            success, frame = True, self._last_frame
                        else:
                            # No cached frame available (shouldn't happen in normal usage)
                            success, frame = False, None
                else:
                    # Normal frame read
                    success, frame = self._read_frame()

            if success and frame is not None:
                frame_timecode = Timecode(self.get_fps(), frames=self._timecodes[0].frames) + self._current_frame
                paths.append(str(self.video_paths[self.source_index]))
                imgs.append(frame)

                frame_length = self._frame_lengths[self.source_index]
                info.append([frame_timecode, self._current_frame,
                             f"video {self.source_index + 1}/{self.num_sources} (frame {self._source_frame}/{frame_length}) {self.video_paths[self.source_index]}: "])

                # Update frame counters
                self._current_frame = next_frame + self.frames_to_stride

                # Update gap offset if entering a gap
                if end_of_source_frame <= self._current_frame < start_of_next_source_frame:
                    self.gap_offset = start_of_next_source_frame - self._current_frame
            else:
                # Move to next source on failure
                self.source_index += 1
                self.gap_offset = 0
                if self.source_index < self.num_sources:
                    self._start_current_video()

        logger.debug(
            f"Completed batch. Expected next frame: {self._current_frame} ({self._timecodes[0] + self._current_frame}) "
            f"source_index: {self.source_index}, source_frame: {self._source_frame}, gap_offset: {self.gap_offset}")

        return paths, imgs, info

    def __iter__(self):
        """Iterator interface"""
        return self

    def __len__(self):
        """Returns the number of batches in the object"""
        total_frames = sum(self._frame_lengths) + sum(self.gaps)
        return math.ceil(total_frames / self.frames_to_stride)

    def __del__(self):
        """Cleanup FFmpeg process"""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()

def get_capture_times_from_exif(jpg_directory: pathlib.Path) -> SortedDict:
    frame_times = SortedDict()
    # Iterate through the .jpg files in the directory
    for image_path in tqdm(jpg_directory.glob("*.jpg")):
        image_path = image_path.resolve()
        try:
            # Open the image
            with open(image_path, 'rb') as image_file:
                image = exif.Image(image_file)

            capture_time = datetime.datetime.strptime(image["datetime_original"] + ":" + image["subsec_time_original"],
                                                      "%Y:%m:%d %H:%M:%S:%f")
            frame_times[image_path] = capture_time
        except (OSError, KeyError, AttributeError) as e:
            # Handle any exceptions while processing the image
            print(f"Error processing image: {image_path}")
            print(e)
    return frame_times


def load_timecode_index(timecode_index: pathlib.Path):
    files = os.listdir(timecode_index)
    time_frame_paths = SortedDict()
    for file in files:
        # Remove extension
        date_part = ".".join(file.split(".")[:-1])
        time = datetime.datetime.strptime(date_part, "%Y-%m-%d_%H:%M:%S.%f")
        time_frame_paths[time] = (timecode_index / file).resolve()
    return time_frame_paths


def create_timecode_symlinks(frames_by_time: SortedDict, out_dir: pathlib.Path):
    # Make the capture time the filename
    time_frame_paths = SortedDict()
    for i, (frame_path, capture_time) in tqdm(enumerate(frames_by_time.items())):
        link_path = out_dir / capture_time.strftime('%Y-%m-%d_%H:%M:%S.%f')
        os.system(f"ln -s {frame_path} {link_path}")
        time_frame_paths[capture_time] = link_path
    return time_frame_paths
