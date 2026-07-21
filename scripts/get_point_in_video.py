#!/usr/bin/env python3
import argparse
import pathlib

import cv2

from visual_race_timing.drawing import render_timecode
from visual_race_timing.loader import VideoLoader, ImageLoader


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=pathlib.Path, nargs='+',
                        help='one or more video filepaths, or a single directory of images')
    parser.add_argument('--seek-frame', type=int,
                        help='seek frame (index from start)')
    parser.add_argument('--seek-timecode-frame', type=int,
                        help='seek frame (timecode frame index from start)')
    parser.add_argument('--seek-time', type=str, default=None,
                        help='seek time to start at, e.g. 05:00:00')
    opt = parser.parse_args()
    assert opt.seek_frame is None or opt.seek_timecode_frame is None, \
        "Cannot set both --seek-frame and --seek-timecode-frame"
    return opt


args = parse_opt()
if len(args.source) == 1 and args.source[0].is_dir():
    loader = ImageLoader(args.source[0])
else:
    if any(s.is_dir() for s in args.source):
        raise SystemExit("Image directories cannot be combined with other sources; pass a single directory.")
    loader = VideoLoader(args.source)

if args.seek_time:
    if len(args.seek_time.split(':')) == 3 and ';' not in args.seek_time:
        args.seek_time += ":00"
    loader.seek_time(args.seek_time)
elif args.seek_frame is not None:
    loader.seek_frame(args.seek_frame)
elif args.seek_timecode_frame is not None:
    loader.seek_timecode_frame(args.seek_timecode_frame)

rect = None
start_x = None
start_y = None


def get_coordinates(event, x, y, flags, param):
    global rect, start_x, start_y
    if param["frame_shape"] is None:
        return
    frame_h, frame_w = param["frame_shape"][:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})  Normalized: [{x / frame_w:.6f}, {y / frame_h:.6f}]")
        start_x = x
        start_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        rect = (start_x, start_y, x, y)
        # Standardize the rectangle coordinates
        rect = (min(rect[0], rect[2]), min(rect[1], rect[3]), max(rect[0], rect[2]), max(rect[1], rect[3]))
        print(f"Crop: {rect[2] - rect[0]}:{rect[3] - rect[1]}, {rect[0]}:{rect[1]}")
        # Calculate next power of 2
        width = 2 ** (rect[2] - rect[0]).bit_length()
        height = 2 ** (rect[3] - rect[1]).bit_length()
        print(f"Next power of 2: {width}:{height}")


cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
callback_param = {"frame_shape": None}
cv2.setMouseCallback('Video', get_coordinates, callback_param)

for path, frames, meta in loader:
    frame = frames[0]
    callback_param["frame_shape"] = frame.shape
    if rect is not None:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    render_timecode(meta[0][0], frame, frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
