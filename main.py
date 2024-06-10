#!/usr/bin/env python
import argparse
import pathlib
import subprocess

import cv2
import numpy as np
from pathlib import Path

import torch
import ultralytics.data.loaders
import yaml
from types import SimpleNamespace

from timecode import Timecode
from ultralytics import YOLO

from visual_race_timing.annotations import save_txt_annotation, offset_with_crop
from visual_race_timing.drawing import render_timecode, draw_annotation

from visual_race_timing.geometry import line_segment_intersects_boxes, side_of_line, line_segment_to_box_distance
from visual_race_timing.timecode import get_timecode, get_video_height_width
from visual_race_timing.tracker import RaceTracker
from ultralytics.utils.plotting import Annotator

# Load race configuration from yaml
race_config = 'config.yaml'
with open(race_config, "r") as f:
    race_config = yaml.load(f.read(), Loader=yaml.FullLoader)


class RunnerState:
    BEFORE_LINE_LEFT = 0
    BEFORE_LINE_RIGHT = 1
    AFTER_LINE = 2
    NOT_VISIBLE = 3


def on_predict_start(predictor, args, persist=True):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    tracker_config = 'tracker_config.yaml'
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(cfg)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = RaceTracker(
            args.reid_model,
            cfg,
            participants={bib.lower(): name for bib, name in race_config['participants'].items()},
            device=args.device
        )
        if hasattr(tracker, 'feature_extractor'):
            pass
            # tracker.feature_extractor.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


def on_predict_postprocess_end(predictor: YOLO, line_seg_pts=None, persist: bool = True) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers across video files.
    """
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"

    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            tracker.update(det, im0s[i])
            continue

        if line_seg_pts is None:
            on_line_mask = np.ones(len(det), dtype=bool)
        else:
            # on_line_mask = line_segment_intersects_boxes(line_seg_pts[0], line_seg_pts[1], det.xyxy)
            on_line_mask = line_segment_to_box_distance(line_seg_pts[0], line_seg_pts[1], det.xyxy) < 10
        tracks = tracker.update(det[on_line_mask], im0s[i])
        # Make sure outside only gets on-the-line detections
        predictor.results[i] = predictor.results[i][on_line_mask]
        if len(tracks) == 0:
            continue
        # The last column of the tracks array is the original index of the detection. Rearrange them to match
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = dict()
        update_args["obb" if is_obb else "boxes"] = torch.as_tensor(tracks[:, :-1])
        predictor.results[i].update(**update_args)


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
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    old_method = ultralytics.data.loaders.LoadImagesAndVideos._new_video
    timecodes = [get_timecode(source) for source in args.source]
    frame_lengths = [int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_COUNT)) for source in args.source]
    if args.continue_exp:
        # Look in the save directory for the last frame
        last_frame = 0
        frame_dir = Path(f"{args.project}/{args.name}/labels")
        for frame_file in frame_dir.glob('frame_*.txt'):
            frame_num = int(frame_file.stem.split('_')[-1])
            if frame_num > last_frame:
                last_frame = frame_num
        args.seek_time = str(Timecode(timecodes[0].framerate, frames=last_frame - 30))
    if args.seek_time:
        if len(args.seek_time.split(':')) == 3 and ';' not in args.seek_time:
            args.seek_time += ":00"
        seek_timecode = Timecode(timecodes[0].framerate, args.seek_time)
        # Scan from end of sources list to look for start_timecode before seek_time
        for i, timecode in enumerate(timecodes):
            if timecode < seek_timecode:
                break
        args.source = args.source[i:]
        frame_lengths = frame_lengths[i:]
        timecodes = timecodes[i:]
        args.seek_frame = (seek_timecode - timecodes[0]).frames

    if args.seek_frame:
        # Skip sources if we ask for a frame index way out ahead
        while args.seek_frame > frame_lengths[0]:
            args.seek_frame -= frame_lengths.pop(0)
            timecodes.pop(0)
            args.source.pop(0)

        def monkey_patch_seek_frame_new_video(self, path):
            """Creates a new video capture object for the given path."""
            self.frame = 0
            self.cap = cv2.VideoCapture(path)
            if args.seek_frame:
                success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, args.seek_frame)
                if success:
                    self.frame = args.seek_frame
                    # Only seek once. Subsequent loads will start at frame 0 now
                    # We're abusing the args obj.
                    args.seek_frame = None
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if not self.cap.isOpened():
                raise FileNotFoundError(f"Failed to open video {path}")
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

        ultralytics.data.loaders.LoadImagesAndVideos._new_video = monkey_patch_seek_frame_new_video

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
        processes = []
        crop_arg = f'{args.crop[0]}:{args.crop[1]}:{args.crop[2]}:{args.crop[3]}'
        # Colons mess up basic resource parsing in ultralytics, so we'll dashes/underscores
        crop_filename_str = f'{args.crop[0]}-{args.crop[1]}_{args.crop[2]}-{args.crop[3]}'
        # We can only crop files
        args.source = args.source if isinstance(args.source, list) else [args.source]
        args.source = [Path(source) for source in args.source]
        crop_paths = []
        # Start n shells to crop all sources
        for source in args.source:
            # Store the cropped file in the same directory as the source
            crop_path = Path(source).parent / f'{Path(source).stem}_crop_{crop_filename_str}.mp4'
            crop_paths.append(crop_path)
            if crop_path.exists():
                continue
            crop_cmd = f'ffmpeg -i {str(source)} -vf "crop={crop_arg}" -c:a copy {str(crop_path)}'
            proc = subprocess.Popen(crop_cmd, shell=True)
            processes.append(proc)

        for proc in processes:
            proc.wait()

        args.source = crop_paths

    crop_dims = [get_video_height_width(source) for source in args.source]
    cv2.namedWindow("Race Tracking")
    for source, start_timecode, source_dims in zip(args.source, timecodes, original_source_dims):
        results = yolo.track(
            source=source,
            conf=args.conf,
            iou=args.iou,
            agnostic_nms=args.agnostic_nms,
            show=False,
            stream=True,
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
            line_width=args.line_width
        )
        # store custom args in predictor
        yolo.predictor.custom_args = args
        yolo.clear_callback('on_predict_start')
        yolo.clear_callback('on_predict_postprocess_end')
        yolo.add_callback('on_predict_start', lambda detector: on_predict_start(detector, args, persist=True))
        yolo.add_callback('on_predict_postprocess_end',
                          lambda detector: on_predict_postprocess_end(detector, [finish_line_p0, finish_line_p1],
                                                                      persist=True))
        wants_exit = False
        if args.save:
            yolo.predictor.save_dir.mkdir(parents=True, exist_ok=True)
            print(f'Saving results to {yolo.predictor.save_dir}')
            video_path = yolo.predictor.save_dir / (str(source.stem) + '_output.mp4')
            vid_writer = cv2.VideoWriter(
                filename=str(video_path),
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=30,  # integer required, floats produce error in MP4 codec
                frameSize=args.imgsz,  # (width, height)
            )

        for r in results:

            # Ultralytics 1-indexes frames, but timecode is for frame 0
            timecode = start_timecode + yolo.predictor.dataset.frame - 1
            timecode_frame = start_timecode.frames + yolo.predictor.dataset.frame - 1
            if len(r.boxes) == 0:
                print(f"No people detected at {timecode}")
                continue
            bibs = [format(int(box.id), "02x") for box in r.boxes]
            names = [race_config['participants'].get(bib.upper(), '').split(" ")[0] for bib in bibs]
            labels = [f'{label} {name}' for label, name in zip(bibs, names)]
            img = draw_annotation(r.boxes.data, keypoints=r.keypoints.data if r.keypoints else None, labels=labels,
                                  img=r.orig_img)

            if args.show:
                render_timecode(timecode, img)
                cv2.line(img, finish_line_p0, finish_line_p1, (0, 255, 0), 2)
                cv2.imshow('Race Tracking', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    wants_exit = True
                    break

            visible_ids = [box.id for box in r.boxes]
            crossings = [False] * len(r.boxes)
            if r.keypoints is not None:
                pass
                # update_crossings(r.boxes, finish_line_p0, finish_line_p1, crossings)

            for runner in runners.keys():
                if runner not in visible_ids:
                    runners[runner] = RunnerState.NOT_VISIBLE
            frame_result_path = f'{yolo.predictor.save_dir}/labels/frame_{timecode_frame}.txt'
            boxes, keypoints = r.boxes, r.keypoints
            if args.crop:
                boxes, keypoints = offset_with_crop(r.boxes, r.keypoints, args.crop, source_dims)
            save_txt_annotation(boxes, keypoints, crossings, Path(frame_result_path), replace=True)

            if args.save:
                vid_writer.write(img)

        if args.save:
            vid_writer.release()  # release final video writer
        if wants_exit:
            break

    ultralytics.data.loaders.LoadImagesAndVideos._new_video = old_method


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default='osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    # We depend on video files with timecode metadata. Hacking required to support other sources.
    parser.add_argument('--source', type=pathlib.Path, nargs='+',
                        help='filepath(s)')
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
    parser.add_argument('--project', default=pathlib.Path('/tmp/tracking/runs/track'),
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--seek-frame', type=int, default=0,
                        help='seek frame to start tracking')
    parser.add_argument('--seek-time', type=str, default=None, help='seek time to start tracking')
    parser.add_argument('--continue-exp', default=False, action='store_true',
                        help='continue tracking from last frame')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
