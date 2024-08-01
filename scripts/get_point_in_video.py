import pathlib

import cv2
import sys
from visual_race_timing.loader import VideoLoader, ImageLoader

input_path = pathlib.Path(sys.argv[1])
if input_path.is_dir():
    loader = ImageLoader(input_path)
else:
    loader = VideoLoader([input_path])

rect = None
start_x = None
start_y = None


def get_coordinates(event, x, y, flags, param):
    global rect, start_x, start_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})")
        start_x = x
        start_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        rect = (start_x, start_y, x, y)
        print(f"Crop: {rect[2] - rect[0]}:{rect[3] - rect[1]}, {rect[0]}:{rect[1]}")
        # Calculate next power of 2
        width = 2 ** (rect[2] - rect[0]).bit_length()
        height = 2 ** (rect[3] - rect[1]).bit_length()
        print(f"Next power of 2: {width}:{height}")


cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Video', get_coordinates)

for path, frames, meta in loader:
    frame = frames[0]
    if rect is not None:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
