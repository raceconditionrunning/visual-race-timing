#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
from pathlib import Path

import torch
import yaml
from types import SimpleNamespace

from timecode import Timecode
from ultralytics.engine.results import Boxes

from visual_race_timing.annotations import save_txt_annotation, load_annotations
from visual_race_timing.drawing import draw_annotation

from visual_race_timing.geometry import side_of_line
from visual_race_timing.video import get_video_height_width, crop_videos
from visual_race_timing.tracker import RaceTracker

from visual_race_timing.video_player import VideoPlayer, DisplayWindow


class RunnerState:
    BEFORE_LINE_LEFT = 0
    BEFORE_LINE_RIGHT = 1
    AFTER_LINE = 2
    NOT_VISIBLE = 3


def update_crossings(result, finish_line_p0, finish_line_p1, crossings):
    # FIXME
    for i, (box, kp) in enumerate(zip(result.boxes, result.keypoints)):
        if box.id is None:
            # Tracker hasn't ID'd this box yet
            continue
        if kp is None:
            # Must not be running a pose model
            continue
        runner_id = int(box.id.cpu().numpy()[0])
        kpts = kp.xy.squeeze().cpu().numpy()[15:]
        if not np.all(kpts):
            continue
        sides = side_of_line(finish_line_p0, finish_line_p1, kpts)
        if sides[0] == 0 and sides[1] == 0:
            state = RunnerState.BEFORE_LINE_LEFT
        elif sides[0] == 1 and sides[1] == 1:
            state = RunnerState.BEFORE_LINE_RIGHT
        else:
            state = RunnerState.AFTER_LINE

        if runner_id not in runners.keys():
            runners[runner_id] = state
            if state == RunnerState.AFTER_LINE:
                crossings[i] = True
        elif runners[runner_id] == RunnerState.NOT_VISIBLE:
            runners[runner_id] = state
            if state == RunnerState.AFTER_LINE:
                crossings[i] = True
        elif runners[runner_id] == RunnerState.BEFORE_LINE_LEFT and (
                state == RunnerState.BEFORE_LINE_RIGHT or state == RunnerState.AFTER_LINE):
            runners[runner_id] = RunnerState.AFTER_LINE
            crossings[i] = True
        elif runners[runner_id] == RunnerState.BEFORE_LINE_RIGHT and (
                state == RunnerState.BEFORE_LINE_LEFT or state == RunnerState.AFTER_LINE):
            runners[runner_id] = RunnerState.AFTER_LINE
            crossings[i] = True


@torch.no_grad()
def run(args):
    display_window = DisplayWindow("Race Timing")
    display_window.start()

    # Load race configuration from yaml
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    tracker_config = args.project / 'tracker_config.yaml'
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(cfg)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']

    tracker = RaceTracker(
        args.reid_model,
        cfg,
        participants={bib.lower(): name for bib, name in race_config['participants'].items()},
        device=args.device
    )
    tracker.display_delegate = lambda img: display_window.img_queue.put(img, block=False)

    def overlay_annotations(frame, frame_num):
        frame_annotations = track_boxes.get(frame_num, None)
        frame_boxes = detection_boxes.get(frame_num, None)

        if frame_boxes is not None:
            boxes = frame_boxes['boxes']
            kpts = frame_boxes['kpts']
            crossings = frame_boxes['crossings']
            frame = draw_annotation(img=frame, boxes=boxes, keypoints=kpts, crossings=crossings, labels=None,
                                    conf=boxes[:, 4],
                                    kpt_radius=2, colors=[(0, 255, 0)] * len(boxes), line_width=1)
        if frame_annotations is not None:
            boxes = frame_annotations['boxes']
            kpts = frame_annotations['kpts']
            crossings = frame_annotations['crossings']
            bibs = [format(box[4].astype(int), '02x') for box in boxes]
            names = [race_config['participants'].get(bib.upper(), None) for bib in bibs]
            names = [name.split(" ")[0] if name else bib for bib, name in zip(bibs, names)]
            labels = [f"{bib}{' ' + name if name else ''}" for bib, name in zip(bibs, names)]
            frame = draw_annotation(img=frame, boxes=boxes, keypoints=kpts, crossings=crossings, labels=labels,
                                    kpt_radius=2, line_width=1)

        return frame

    player = VideoPlayer(args.source, True)
    player.overlay_delegate = overlay_annotations
    if args.seek_frame:
        args.seek_time = str(Timecode(player.get_last_timecode().framerate, frames=args.seek_timecode_frame))
    if args.seek_time:
        player.seek_time(args.seek_time)

    runners = {}
    finish_line_p0 = race_config['finish_line'][0]
    finish_line_p1 = race_config['finish_line'][1]
    # HACK(nickswalker 6-9-24): Patching a crop into the detector will be a mess and likely break in future versions.
    # Instead, we will crop using ffmpeg before running the detector. We'll persist the cropped file in a predictable
    # location, but first invocation with a new crop will be very expensive.
    original_source_dims = [get_video_height_width(source) for source in args.source]
    if args.crop:
        args.imgsz = args.crop[0], args.crop[1]
        # Apply crop to finish line
        finish_line_p0 = (finish_line_p0[0] - args.crop[2], finish_line_p0[1] - args.crop[3])
        finish_line_p1 = (finish_line_p1[0] - args.crop[2], finish_line_p1[1] - args.crop[3])

        args.source = crop_videos(args.source, args.crop)

    detection_boxes = load_annotations(args.project / 'detections')
    (args.project / 'tracks').mkdir(exist_ok=True, parents=True)
    track_boxes = load_annotations(args.project / 'tracks')

    for frame_num, detections in detection_boxes.items():
        player.seek_frame(frame_num)
        frame = player._advance_frame()

        boxes = Boxes(detections["boxes"], frame.shape[:2])
        tracks = tracker.update(boxes, frame, frame_num)

        # The last column of the tracks array is the original index of the detection. Rearrange them to match
        idx = tracks[:, -1].astype(int)
        boxes = Boxes(tracks[idx, :-1], frame.shape[:2])
        crossings = [False] * len(boxes)
        keypoints = None
        track_boxes[frame_num] = {"boxes": boxes.data, "kpts": keypoints, "crossings": crossings}
        save_txt_annotation(boxes, detections["kpts"], detections["crossings"], args.project / "tracks", replace=True)
        display_window.img_queue.put(overlay_annotations(frame, frame_num))


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('project', type=pathlib.Path,
                        help='save results to project/name')
    parser.add_argument('--reid-model', type=Path, default='osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    # We depend on video files with timecode metadata. Hacking required to support other sources.
    parser.add_argument('--source', type=pathlib.Path, nargs='+',
                        help='filepath(s)')
    parser.add_argument('--crop', nargs='+', type=int, default=None,
                        help='inference area w h x y')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='cuda',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--seek-frame', type=int,
                        help='seek frame (index from start) to start tracking')
    parser.add_argument('--seek-timecode-frame', type=int,
                        help='seek frame (timecode frame index from start) to start tracking')
    parser.add_argument('--seek-time', type=str, default=None, help='seek time to start tracking')
    parser.add_argument('--continue-exp', default=False, action='store_true',
                        help='continue tracking from last frame')

    opt = parser.parse_args()
    assert opt.seek_frame is None or opt.seek_timecode_frame is None, "Cannot set both seek_frame and seek_timecode_frame"
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
