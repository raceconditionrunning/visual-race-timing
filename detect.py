#!/usr/bin/env python
import argparse
import pathlib

import cv2
import numpy as np
from pathlib import Path

import torch
import tqdm
import yaml

from ultralytics import YOLO

from visual_race_timing.annotations import SQLiteAnnotationStore, offset_with_crop
from visual_race_timing.drawing import render_timecode, draw_annotation

from visual_race_timing.geometry import line_segment_to_box_distance
from visual_race_timing.loader import ImageLoader, VideoLoader
from visual_race_timing.media_player import DisplayWindow


@torch.no_grad()
def run(args):
    display_window = DisplayWindow("Race Timing")
    display_window.start()
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )
    # Load race configuration from yaml
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    finish_line_p0 = race_config['finish_line'][0]
    finish_line_p1 = race_config['finish_line'][1]

    if args.crop:
        args.imgsz = args.crop[0], args.crop[1]
        # Apply crop to finish line
        finish_line_p0 = (finish_line_p0[0] - args.crop[2], finish_line_p0[1] - args.crop[3])
        finish_line_p1 = (finish_line_p1[0] - args.crop[2], finish_line_p1[1] - args.crop[3])
    line_seg_pts = [finish_line_p0, finish_line_p1]
    if len(args.source) == 1 and args.source[0].is_dir():
        loader = ImageLoader(args.source[0], batch=args.batch, crop=args.crop)
    else:
        loader = VideoLoader(args.source, batch=args.batch, vid_stride=args.vid_stride, crop=args.crop)

    store = SQLiteAnnotationStore(Path(f"{args.project}/{args.name}/annotations.db"))
    if args.continue_exp:
        # Look in the save directory for the last frame
        last_frame = store.get_last_frame(args.yolo_model.stem)
        if last_frame != 0:
            args.seek_timecode_frame = last_frame

    if args.seek_time:
        if len(args.seek_time.split(':')) == 3 and ';' not in args.seek_time:
            args.seek_time += ":00"
        loader.seek_timecode(args.seek_time)
    elif args.seek_frame:
        loader.seek_frame(args.seek_frame)
    elif args.seek_timecode_frame:
        loader.seek_timecode_frame(args.seek_timecode_frame)

    for source, images, metadata in tqdm.tqdm(loader, total=len(loader)):
        results = yolo.predict(source=images,
                               conf=args.conf,
                               iou=args.iou,
                               agnostic_nms=args.agnostic_nms,
                               show=False,
                               stream=False,
                               device=args.device,
                               show_conf=args.show_conf,
                               save_txt=False,
                               show_labels=args.show_labels,
                               save=False,
                               verbose=args.verbose,
                               exist_ok=args.exist_ok or args.continue_exp,
                               project=args.project,
                               name=args.name,
                               classes=[0],  # Only track people
                               imgsz=args.imgsz,
                               vid_stride=args.vid_stride,
                               line_width=args.line_width,
                               batch=args.batch)

        yolo.predictor.custom_args = args
        yolo.clear_callback('on_predict_start')
        yolo.clear_callback('on_predict_postprocess_end')
        for r, m in zip(results, metadata):
            det = r.boxes.cpu().numpy()
            if len(det) == 0:
                continue

            if line_seg_pts is None:
                on_line_mask = np.ones(len(det), dtype=bool)
            else:
                on_line_mask = line_segment_to_box_distance(line_seg_pts[0], line_seg_pts[1], det.xyxy) < 10

            start_timecode = m[0]
            timecode_frame = start_timecode.frames
            boxes, keypoints = r.boxes[on_line_mask], None
            if len(boxes) == 0:
                continue
            crossings = [False] * len(boxes)
            if r.keypoints is not None:
                keypoints = r.keypoints[on_line_mask]

            if args.show:
                img = draw_annotation(boxes.data, keypoints=keypoints.data if keypoints else None,
                                      img=r.orig_img, line_width=args.line_width, conf=boxes.conf)
                render_timecode(start_timecode, img, img)
                cv2.line(img, finish_line_p0, finish_line_p1, (0, 255, 0), 2)
                display_window.img_queue.put(img, block=False)
            if args.crop:
                boxes, keypoints = offset_with_crop(boxes, keypoints, args.crop, loader._source_dims[0])
            store.save_annotation(timecode_frame, boxes, keypoints, crossings, source=args.yolo_model.stem,
                                  replace=True)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=Path, default=pathlib.Path('data/exp'),
                        help='save results to project')
    parser.add_argument('--yolo-model', type=Path, default='yolov8n',
                        help='yolo model path')
    # We depend on video files with timecode metadata. Hacking required to support other sources.
    parser.add_argument('--source', type=pathlib.Path, nargs='+',
                        help='filepath(s)')
    parser.add_argument('--batch', type=int, default=1,
                        help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1920, 1080],
                        help='inference size h w')
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
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')

    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
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
