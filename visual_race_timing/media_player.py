import logging
import threading
from queue import Queue

import cv2
from timecode import Timecode
from collections import deque
from visual_race_timing.drawing import render_timecode
from visual_race_timing.loader import ImageLoader, VideoLoader


class MediaPlayer:
    def __init__(self, paused=False):
        self.click_delegate = lambda frame, frame_number, mouse_pt, flags: None
        self.annotation_updated = lambda annotation_id, annotation, frame_number: None
        self.pre_display = lambda frame, frame_number: frame
        self.key_delegate = lambda frame, frame_number, key: None

        self._last_frame_img = None
        # No matter what, we need to read the first frame
        self.needs_advance = 1
        if paused:
            self.paused = True
        else:
            self.paused = False

        self.delay = 30  # Initial delay in ms for playback speed
        self.annotation_mode = False
        self.start_point = None
        self.end_point = None
        self.window_name = 'Edit Race Annotations'

    def get_last_timecode(self) -> Timecode:
        """ Return the timecode of the last frame displayed. Returns None if no frame has been displayed. """
        raise NotImplementedError("get_current_time must be implemented by subclass")

    def seek_timecode(self, timecode: Timecode) -> bool:
        raise NotImplementedError("seek_timecode must be implemented by subclass")

    def seek_timecode_frame(self, frame_number: int) -> bool:
        """ Seek to a specific frame number. Returns True if successful, False otherwise. """
        raise NotImplementedError("seek_timecode_frame must be implemented by subclass")

    def seek_time(self, time_str: str) -> bool:
        raise NotImplementedError("seek_time must be implemented by subclass")

    def play(self):
        raise NotImplementedError("play must be implemented by subclass")

    def render(self):
        frame = self._last_frame_img.copy()
        current_timecode = self.get_last_timecode()

        self.draw_active_annotation(frame)
        render_timecode(current_timecode, frame)
        cv2.imshow(self.window_name, self.pre_display(frame, current_timecode.frames))

    def mouse_callback(self, event, x, y, flags, param):
        if not self.paused:
            return
        if event == cv2.EVENT_LBUTTONUP:
            result = self.click_delegate(self._last_frame_img, self.get_last_timecode().frames, (x, y), flags)
            self.render()
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self.start_point = (x, y)
            dotted = self._last_frame_img.copy()
            cv2.circle(dotted, self.start_point, 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, dotted)
        elif event == cv2.EVENT_RBUTTONUP:
            self.end_point = (x, y)
            # ignore small boxes
            if abs(self.start_point[0] - self.end_point[0]) < 10 or abs(self.start_point[1] - self.end_point[1]) < 10:
                self.start_point = None
                self.end_point = None
                return
            # Normalize the points: top-left, bottom-right
            start_point = (min(self.start_point[0], self.end_point[0]), min(self.start_point[1], self.end_point[1]))
            end_point = (max(self.start_point[0], self.end_point[0]), max(self.start_point[1], self.end_point[1]))
            self.start_point = start_point
            self.end_point = end_point
            # Draw the box so we can confirm
            frame = self._last_frame_img.copy()
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
            render_timecode(self.get_last_timecode(), frame)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

            try:
                success = self.annotation_updated(None, [self.start_point, self.end_point], self.get_last_timecode())
                if not success:
                    print("Error updating annotation")
            except Exception as e:
                print(f"Error updating annotation: {e}")

            self.start_point = None
            self.end_point = None
            self.render()

    def draw_active_annotation(self, frame):
        if self.start_point is not None and self.end_point is not None:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.putText(frame, "", (self.start_point[0], self.start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def handle_key_input(self):
        key = cv2.waitKey(self.delay) & 0xFF

        if key == ord('q'):  # Quit
            return -1
        elif key == ord(' '):  # Pause/play toggle
            self.paused = not self.paused
            if not self.paused:
                self.needs_advance = 1
        elif key == ord(')'):  # Next frame
            self.needs_advance = 1
        elif key == ord('('):  # Previous frame
            self.needs_advance = -1
        elif key == ord('*'):
            pass
        elif key == ord('/'):
            pass
        elif key == ord('s'):  # Seek to time
            timecode = input("Enter time to seek to (HH:MM:SS): ")
            try:
                self.seek_time(timecode)
                self.render()
            except Exception as e:
                print("Invalid time format. Please enter time in HH:MM:SS format.")
                return
        elif key == ord('+'):  # Increase playback speed
            self.delay = max(1, self.delay - 5)
        elif key == ord('-'):  # Decrease playback speed
            self.delay += 5
        else:
            self.key_delegate(self._last_frame_img, self.get_last_timecode().frames, key)


class VideoPlayer(MediaPlayer):
    def __init__(self, sources, paused=False):
        super().__init__(paused)
        self.loader = VideoLoader(sources)
        self._last_timecode = None

    def seek_time(self, time_str: str) -> bool:
        # Check if includes frame number (HH:MM:SS:FF), if not add :00
        if len(time_str.split(':')) == 3 and ";" not in time_str:
            time_str += ':00'
        return self.loader.seek_time(time_str)

    def seek_timecode(self, timecode: Timecode) -> bool:
        return self.loader.seek_timecode(timecode)

    def seek_timecode_frame(self, frame_number: int) -> bool:
        return self.seek_timecode(Timecode(self.loader.get_fps(), frames=frame_number))

    def seek_frame(self, frame_number: int) -> bool:
        return self.loader.seek_frame(frame_number)

    def get_last_timecode(self) -> Timecode:
        return self._last_timecode

    def _advance_frame(self):
        path, frame, metadata = self.loader.__next__()
        self._last_timecode = metadata[0][0]
        self._last_frame_img = frame[0]
        return frame[0]

    def play(self):
        # NORMAL disables right click context menu
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            if self.needs_advance:
                if self.needs_advance < 0:
                    success = self.seek_timecode(self.get_last_timecode() - 1)
                    if not success:
                        logging.info("Failed to seek backward.")
                        break

                frame = self._advance_frame()
                if frame is None:
                    break

                self.render()
                if self.paused:
                    self.needs_advance = 0
            result = self.handle_key_input()
            if result == -1:
                break

        cv2.destroyAllWindows()


class FrameBuffer:
    def __init__(self, max_size=120):  # ~4 seconds at 30fps
        self.buffer = deque(maxlen=max_size)
        self.frame_numbers = deque(maxlen=max_size)
        self.max_size = max_size

    def add_frame(self, frame_num, frame_data, metadata):
        """Add a frame to the buffer with YUV420 compression"""
        # Convert BGR to YUV420 for 50% memory reduction
        yuv_frame = cv2.cvtColor(frame_data[0], cv2.COLOR_BGR2YUV_I420)
        self.frame_numbers.append(frame_num)
        self.buffer.append((yuv_frame, metadata, frame_data[0].shape))

    def get_frame(self, target_frame_num):
        """Get frame from buffer if available, returns None if not found"""
        try:
            idx = list(self.frame_numbers).index(target_frame_num)
            yuv_frame, metadata, original_shape = self.buffer[idx]
            # Convert YUV420 back to BGR
            bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            # Ensure correct shape (cvtColor might return flattened array)
            bgr_frame = bgr_frame.reshape(original_shape)
            return [bgr_frame], metadata
        except ValueError:
            return None

    def has_frame(self, frame_num):
        """Check if frame is in buffer"""
        return frame_num in self.frame_numbers

    def get_oldest_frame_num(self):
        """Get the oldest frame number in buffer"""
        return self.frame_numbers[0] if self.frame_numbers else None

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.frame_numbers.clear()


class BufferedVideoPlayer(VideoPlayer):
    def __init__(self, sources, paused=False, buffer_seconds=4):
        super().__init__(sources, paused)

        # Calculate buffer size based on video fps
        fps = float(self.loader.get_fps()) if hasattr(self.loader, 'get_fps') else 30
        buffer_size = int(fps * buffer_seconds)

        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self._last_sequential_frame = None
        # The next timecode that should be returned by the next call to __next__. This class
        # manipulates this
        self.__buffered_next_frame = None

    def seek_timecode(self, timecode: Timecode) -> bool:
        if self.get_last_timecode() == timecode:
            return True

        target_frame = timecode.frames - self.loader._timecodes[0].frames

        if self.frame_buffer.has_frame(target_frame):
            self.__buffered_next_frame = target_frame
            return True

        self.__buffered_next_frame = None
        return self.loader.seek_timecode(timecode)

    def seek_frame(self, frame_number: int) -> bool:
        # Check buffer first for backward seeks
        current_frame = self.loader.get_current_frame()

        if self.frame_buffer.has_frame(frame_number):
            return True

        return self.loader.seek_frame(frame_number)

    def _advance_frame(self):
        try:
            if self.__buffered_next_frame is not None:
                # We've had seeking operations. Let's first see if we can serve them from the buffer.
                target_frame = self.__buffered_next_frame
                if self.frame_buffer.has_frame(target_frame):
                    frame_data = self.frame_buffer.get_frame(target_frame)
                    self._last_timecode = frame_data[1][0][0]
                    assert frame_data[1][0][1] == target_frame
                    self._last_frame_img = frame_data[0][0]
                    self.__buffered_next_frame = self.__buffered_next_frame + 1
                    return frame_data[0][0]
                else:
                    # If not found in buffer, have to actually seek
                    self.loader.seek_frame(self.__buffered_next_frame)
                    self.__buffered_next_frame = None
            elif self.frame_buffer.has_frame(self.loader.get_current_frame()):
                # TRICK: We don't have any buffered seeks, but we're advancing into a frame that is already in the buffer.
                # We'll treat this as though we've seeked to this point and allow the buffer path above to handle it.
                self.__buffered_next_frame = self.loader.get_current_frame()
                return self._advance_frame()
            path, frame, metadata = self.loader.__next__()
            self._last_timecode = metadata[0][0]
            self._last_frame_img = frame[0]

            # Add to buffer
            self.frame_buffer.add_frame(metadata[0][1], frame, metadata)

            return frame[0]
        except StopIteration:
            # End of video
            return None


class DisplayWindow(threading.Thread):
    def __init__(self, window_name):
        threading.Thread.__init__(self)
        self.img_queue = Queue()
        self.window_name = window_name
        self.stop_event = threading.Event()

    def clear(self):
        with self.img_queue.mutex:
            self.img_queue.queue.clear()

    def run(self):
        cv2.namedWindow(self.window_name)
        while not self.stop_event.is_set():
            if not self.img_queue.empty():
                img = self.img_queue.get()
                cv2.imshow(self.window_name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        cv2.destroyAllWindows()


class PhotoPlayer(MediaPlayer):
    def __init__(self, frame_directory, paused=False):
        self.loader = ImageLoader(frame_directory)
        super().__init__(paused)
        self.index = 0
        self.current_frame_timecode = None

    def seek_time(self, time_str: str) -> bool:
        # Check if includes frame number (HH:MM:SS:FF), if not add :00
        if len(time_str.split(':')) == 3 and ";" not in time_str:
            time_str += ':00'
        return self.seek_timecode(Timecode(1000, start_timecode=time_str))

    def seek_timecode(self, timecode: Timecode) -> bool:
        return self.loader.seek_timecode(timecode)

    def get_last_timecode(self):
        return self.current_frame_timecode

    def _advance_frame(self):
        path, frame, metadata = self.loader.__next__()
        self._last_frame_img = frame[0]
        self._last_frame_timecode = metadata[0][0]
        return frame[0]

    def play(self):
        # NORMAL disables right click context menu
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            if self.needs_advance:
                if self.needs_advance < 0:
                    self.index -= 2

                frame = self._advance_frame()
                if frame is None:
                    break

                self.render()
                if self.paused:
                    self.needs_advance = 0
            result = self.handle_key_input()
            if result == -1:
                break

        cv2.destroyAllWindows()
