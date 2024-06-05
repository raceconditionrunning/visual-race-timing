import argparse
import pathlib

import cv2
import numpy as np
from pathlib import Path

import torch
import yaml
from types import SimpleNamespace

from ultralytics import YOLO

from visual_race_timing.geometry import line_segment_intersects_boxes, side_of_line
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


def on_predict_start(predictor, args, persist=False):
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
            cfg
        )
        if hasattr(tracker, 'feature_extractor'):
            pass
            #tracker.feature_extractor.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


def on_predict_postprocess_end(predictor: YOLO, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    line_seg_pts = race_config['finish_line']
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue
        expanded_boxes = det.xyxy
        on_line = line_segment_intersects_boxes(line_seg_pts[0], line_seg_pts[1], expanded_boxes)
        tracks = tracker.update(det[on_line], im0s[i])
        # Make sure outside only gets on-the-line detections
        predictor.results[i] = predictor.results[i][on_line]
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = dict()
        update_args["obb" if is_obb else "boxes"] = torch.as_tensor(tracks[:, :-1])
        predictor.results[i].update(**update_args)


def save_txt(boxes, kpts, crossings, txt_file):
    texts = []
    # Detect/segment/pose
    for j, d in enumerate(boxes):
        c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        line = (c, *(d.xywhn.view(-1)))
        if kpts is not None:
            kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
            line += (*kpt.reshape(-1).tolist(),)
        line += (conf,) + (() if id is None else (id,)) + (crossings[j],)
        texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, "a") as f:
            f.writelines(text + "\n" for text in texts)

@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=False,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.clear_callback('on_predict_start')
    yolo.clear_callback('on_predict_postprocess_end')
    yolo.add_callback('on_predict_start', lambda detector: on_predict_start(detector, args, persist=True))
    yolo.add_callback('on_predict_postprocess_end', lambda detector: on_predict_postprocess_end(detector, persist=True))

    # store custom args in predictor
    yolo.predictor.custom_args = args

    runners = {}
    finish_line_p0 = race_config['finish_line'][0]
    finish_line_p1 = race_config['finish_line'][1]

    if args.save:
        yolo.predictor.save_dir.mkdir(parents=True, exist_ok=True)
        video_path = yolo.predictor.save_dir / 'output.mp4'
        print(str(video_path))
        vid_writer = cv2.VideoWriter(
            filename=str(video_path),
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=30,  # integer required, floats produce error in MP4 codec
            frameSize=args.imgsz,  # (width, height)
        )

    try:
        for r in results:
            img = r.plot(line_width=args.line_width, font_size=0.1)

            if args.show:
                cv2.imshow('Race Tracking', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    break

            visible_ids = [box.id for box in r.boxes]
            crossings = [False] * len(r.boxes)
            for i, (box, kp) in enumerate(zip(r.boxes, r.keypoints)):
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
                elif runners[runner_id] == RunnerState.BEFORE_LINE_LEFT and (state == RunnerState.BEFORE_LINE_RIGHT or state == RunnerState.AFTER_LINE):
                    runners[runner_id] = RunnerState.AFTER_LINE
                    crossings[i] = True
                elif runners[runner_id] == RunnerState.BEFORE_LINE_RIGHT and (state == RunnerState.BEFORE_LINE_LEFT or state == RunnerState.AFTER_LINE):
                    runners[runner_id] = RunnerState.AFTER_LINE
                    crossings[i] = True

                if crossings[i]:
                    annotator = Annotator(img)
                    annotator.box_label(box.xyxy.squeeze().cpu().numpy(), color=(255, 255, 255))
                    img = annotator.result()

            for runner in runners.keys():
                if runner not in visible_ids:
                    runners[runner] = RunnerState.NOT_VISIBLE

            frame_result_path = f'{r.save_dir}/labels/frame_{yolo.predictor.dataset.frame}.txt'
            save_txt(r.boxes, r.keypoints, crossings, frame_result_path)

            if True in crossings and args.show:
                cv2.imshow('Race Tracking', img)
                cv2.waitKey(1)

            if args.save:
                vid_writer.write(img)
    except KeyboardInterrupt:
        pass

    if args.save:
        vid_writer.release()  # release final video writer


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default='osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1920, 1080],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
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
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
