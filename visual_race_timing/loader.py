import bisect
import datetime
import math
import pathlib
import os
from typing import List

import cv2
import exif
from sortedcontainers import SortedDict
from timecode import Timecode
from tqdm import tqdm

from visual_race_timing.video import get_timecode, get_video_height_width


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
        self._frame_lengths = [int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_COUNT)) for source in paths]
        end_timecodes = [timecode + frame_length for timecode, frame_length in
                         zip(self._timecodes, self._frame_lengths)]
        for i in range(1, len(end_timecodes)):
            assert end_timecodes[i - 1] < self._timecodes[i], "Videos must not have overlapping timecode"

        self.num_sources = len(paths)
        self.vid_stride = vid_stride  # video frame-rate stride

        self._new_video(self.files[0])  # new video
        self._current_frame = 0
        self._source_frame = 0

    def __next__(self):
        """Returns the next batch of images along with their paths and metadata."""
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

            success = True
            for _ in range(self.vid_stride - 1):  # FIXME: Messes with seeks
                success = self.cap.grab()
                if not success:
                    break  # end of video or failure

            if success:
                frame_length = self._frame_lengths[self.source_index]
                self._source_frame = int(cv2.VideoCapture.get(self.cap, cv2.CAP_PROP_POS_FRAMES))
                success, im0 = self.cap.retrieve()
                if success:
                    frame_timecode = self._timecodes[self.source_index] + self._source_frame
                    paths.append(path)
                    if self.crop:
                        # given in ffmpeg format: width:height:x:y
                        im0 = im0[self.crop[3]:self.crop[3] + self.crop[1], self.crop[2]:self.crop[2] + self.crop[0]]
                    imgs.append(im0)
                    info.append([frame_timecode,
                                 f"video {self.source_index + 1}/{self.num_sources} (frame {self._source_frame}/{frame_length}) {path}: "])
                    if self._source_frame == frame_length:  # end of video
                        self.source_index += 1
                        self.cap.release()
            else:
                # Move to the next file if the current video ended or failed to open
                self.source_index += 1
                if self.cap:
                    self.cap.release()
                if self.source_index < self.num_sources:
                    self._new_video(self.files[self.source_index])
                    self._current_frame = sum(self._frame_lengths[:self.source_index])

        return paths, imgs, info

    def get_image_dims(self):
        return self._source_dims[0]

    def get_current_time(self) -> Timecode:
        return self._timecodes[self.source_index] + self._source_frame

    def get_current_frame(self) -> int:
        return self._current_frame

    def seek_time(self, time_str: str) -> bool:
        target_time = Timecode(self._timecodes[0].framerate, time_str)
        return self.seek_timecode(target_time)

    def _new_video(self, path):
        """Creates a new video capture object for the given path."""
        self._source_frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")

    def seek_frame(self, target_frame) -> bool:
        """
        Look for the nth frame of all sources combined.
        """
        start_source_index = self.source_index
        self.source_index = 0
        # Skip sources if we ask for a frame index way out ahead
        while target_frame > self._frame_lengths[self.source_index]:
            target_frame -= self._frame_lengths[self.source_index]
            self.source_index += 1
        if self.source_index != start_source_index:
            self._new_video(self.files[self.source_index])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == target_frame

    def seek_timecode(self, target_timecode):
        if target_timecode < self._timecodes[0] or target_timecode > self._timecodes[-1]:
            return False
        start_source_index = self.source_index
        self.source_index = 0
        while target_timecode > (self._timecodes[self.source_index] + self._frame_lengths[self.source_index]):
            self.source_index += 1
        if self.source_index != start_source_index:
            self._new_video(self.files[self.source_index])
        target_frame = (target_timecode - self._timecodes[self.source_index]).frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == target_frame

    def seek_timecode_frame(self, target_frame):
        target_timecode = Timecode(self._timecodes[0].framerate, frames=target_frame)
        self.seek_timecode(target_timecode)

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(sum(self._frame_lengths) / self.bs)  # number of files


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
