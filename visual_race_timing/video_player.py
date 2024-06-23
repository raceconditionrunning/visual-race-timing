import threading
from queue import Queue

import cv2
from timecode import Timecode

from visual_race_timing.drawing import render_timecode
from visual_race_timing.video import get_timecode


class VideoPlayer:
    def __init__(self, sources, paused=False):
        self.cap = cv2.VideoCapture(sources[0])
        self._start_timecodes = [get_timecode(source) for source in sources]

        self.current_frame_img = None
        self.delay = 30  # Initial delay in ms for playback speed
        self._sources = sources
        self._idx = 0
        self.annotation_mode = False
        self.start_point = None
        self.end_point = None

        self.click_delegate = lambda frame, frame_number, mouse_pt, flags: None
        self.annotation_updated = lambda annotation_id, annotation, frame_number: None
        self.pre_display = lambda frame, frame_number: frame
        self.key_delegate = lambda frame, frame_number, key: None

        # No matter what, we need to read the first frame
        self.needs_advance = 1
        if paused:
            self.paused = True
        else:
            self.paused = False

        self.window_name = 'Edit Race Annotations'

    @property
    def source(self):
        return self._sources[self._idx]

    def seek_to_time(self, time_str: str) -> bool:
        # Check if includes frame number (HH:MM:SS:FF), if not add :00
        if len(time_str.split(':')) == 3 and ";" not in time_str:
            time_str += ':00'
        return self.seek_to_timecode(Timecode(self._start_timecodes[0].framerate, time_str))

    def seek_to_timecode(self, timecode: Timecode) -> bool:
        # We roll one back because we have to increment to retrieve the frame
        timecode = timecode - 1
        if self.get_current_time() == timecode:
            return True
        if timecode > self.get_current_time():
            # Optimization: if difference is small, advance instead of invoking expensive seek
            frame_diff = (self.get_current_time() - timecode).frames
            if 1 <= frame_diff < 5:
                while frame_diff > 0:
                    self._advance_frame()
                    frame_diff -= 1
                return True
        # scan from latest to earliest timecodes looking for the right source
        check_idx = len(self._start_timecodes) - 1
        while check_idx > 0 and timecode < self._start_timecodes[check_idx]:
            check_idx -= 1
        if check_idx == -1:
            print("Timecode is before the first source")
            return False
        elif check_idx != self._idx:
            self.cap.release()
            self.cap = cv2.VideoCapture(self._sources[check_idx])
        self._idx = check_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, timecode.frames - self._start_timecodes[self._idx].frames)
        return self.get_current_time() == timecode

    def seek_to_frame(self, frame_number: int) -> bool:
        return self.seek_to_timecode(Timecode(self._start_timecodes[0].framerate, frames=frame_number))

    def get_current_time(self) -> Timecode:
        return self._start_timecodes[self._idx] + int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def next_source(self):
        self.cap.release()
        self._idx += 1
        if self._idx >= len(self._sources):
            return False
        self.cap = cv2.VideoCapture(self.source)
        return True

    def draw_active_annotation(self, frame):
        if self.start_point is not None and self.end_point is not None:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.putText(frame, "", (self.start_point[0], self.start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        if not self.paused:
            return
        if event == cv2.EVENT_LBUTTONUP:
            result = self.click_delegate(self.current_frame_img, self.get_current_time().frames, (x, y), flags)
            self.render()
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self.start_point = (x, y)
            dotted = self.current_frame_img.copy()
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
            frame = self.current_frame_img.copy()
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
            render_timecode(self.get_current_time(), frame)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

            try:
                success = self.annotation_updated(None, [self.start_point, self.end_point], self.get_current_time())
                if not success:
                    print("Error updating annotation")
            except Exception as e:
                print(f"Error updating annotation: {e}")

            self.start_point = None
            self.end_point = None
            self.render()

    def _advance_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            next_available = self.next_source()
            if not next_available:
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
        self.current_frame_img = frame
        return frame

    def render(self):
        frame = self.current_frame_img.copy()
        current_timecode = self.get_current_time()

        self.draw_active_annotation(frame)
        render_timecode(current_timecode, frame)
        cv2.imshow(self.window_name, self.pre_display(frame, current_timecode.frames))

    def play(self):
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
        # NORMAL disables right click context menu
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while self.cap.isOpened():
            if self.needs_advance:
                if self.needs_advance < 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)

                frame = self._advance_frame()
                if frame is None:
                    break

                self.render()
                if self.paused:
                    self.needs_advance = 0

            key = cv2.waitKey(self.delay) & 0xFF

            if key == ord('q'):  # Quit
                break
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
                    self.seek_to_time(timecode)
                    self.render()
                except Exception:
                    print("Invalid time format. Please enter time in HH:MM:SS format.")
                    continue
            elif key == ord('+'):  # Increase playback speed
                self.delay = max(1, self.delay - 5)
            elif key == ord('-'):  # Decrease playback speed
                self.delay += 5
            else:
                self.key_delegate(self.current_frame_img, self.get_current_time().frames, key)

        self.cap.release()
        cv2.destroyAllWindows()


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
