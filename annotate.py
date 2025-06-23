#!/usr/bin/env python
import argparse
import pathlib
from collections import defaultdict
from types import SimpleNamespace
from typing import List

import cv2
import joblib
import numpy as np
import ultralytics.utils.ops
import yaml
from timecode import Timecode
from tqdm import tqdm
from ultralytics.engine.results import Boxes
from ultralytics.utils.metrics import bbox_ioa

from visual_race_timing.annotations import SQLiteAnnotationStore
from visual_race_timing.drawing import draw_annotation

from visual_race_timing.geometry import line_segment_to_box_distance
from visual_race_timing.prompts import ask_for_id
from visual_race_timing.media_player import VideoPlayer, PhotoPlayer, BufferedVideoPlayer

from visual_race_timing.logging import get_logger

logger = get_logger(__name__)


def run(args):
    if len(args.source) == 1 and args.source[0].is_dir():
        player = PhotoPlayer(args.source[0], args.paused)
    else:
        player = BufferedVideoPlayer(args.source, args.paused)

    # Load all annotations
    store = SQLiteAnnotationStore(args.project / 'annotations.db')

    # crossing_map = build_crossing_map(annotations)
    tracker_config = args.project / 'tracker_config.yaml'
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']
    cfg.match_thresh = .8

    if (args.project / 'tracker.pkl').is_file():
        tracker = joblib.load(args.project / 'tracker.pkl')
    else:
       logger.warning(f'No tracker found, initializing from scratch.')
    # Load race configuration from yaml
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    if args.seek_frame:
        args.seek_time = str(Timecode(player.get_last_timecode().framerate, frames=args.seek_timecode_frame))
    if args.seek_time:
        player.seek_time(args.seek_time)

    def overlay_annotations(frame, frame_num):
        frame_notes = store.get_notes(frame_num)
        frame_annotation_boxes, frame_annotation_keypoints, frame_annotation_crossings, _ = store.get_frame_annotation(
            frame_num, frame.shape[:2], "human")
        frame_detection_boxes, frame_detection_keypoints, frame_detection_crossings, _ = store.get_frame_annotation(
            frame_num, frame.shape[:2], args.detection_model)

        if frame_detection_boxes.size > 0:
            frame = draw_annotation(img=frame, boxes=frame_detection_boxes, keypoints=frame_detection_keypoints,
                                    crossings=frame_detection_crossings, labels=None,
                                    conf=frame_detection_boxes[:, 4],
                                    kpt_radius=2 * frame.shape[0] // 1080,
                                    colors=[(0, 255, 0)] * len(frame_detection_boxes),
                                    line_width=1 * frame.shape[0] // 1080)
        if frame_annotation_boxes.size > 0:
            ids = frame_annotation_boxes[:, 4].astype(int)
            bibs = [format(runner_id, '02x') for runner_id in ids]
            names = [race_config['participants'].get(runner_id, None) for runner_id in ids]
            names = [name.split(" ")[0] if name else bib for bib, name in zip(bibs, names)]
            labels = [f"{bib}{' ' + name if name else ''}" for bib, name in zip(bibs, names)]
            frame = draw_annotation(img=frame, boxes=frame_annotation_boxes, keypoints=frame_annotation_keypoints,
                                    crossings=frame_annotation_crossings, labels=labels,
                                    kpt_radius=2 * frame.shape[0] // 1080, line_width=1 * frame.shape[0] // 1080)
        if frame_notes is not None:
            for i, (runner_id, note) in enumerate(frame_notes.items()):
                frame = cv2.putText(frame, f"{runner_id}: {note}", (10 + 10 * i, 10 * frame.shape[0] // 1080),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5 * frame.shape[0] // 1080,
                                    (255, 255, 255), 1 * frame.shape[0] // 1080,
                                    cv2.LINE_AA)
        return frame

    def update_tracker(new_box, runner_id: int):
        # Only use crops larger than 24x24
        if new_box[0, 2] - new_box[0, 0] < 24 or new_box[0, 3] - new_box[0, 1] < 24:
            logger.info("Box too small, ignoring.")
            return False
        else:
            tracker.update_participant_features(player._last_frame_img, new_box, runner_id)
        return True

    def calculate_reid_distances(box, exclude: List[int] = []):
        new_box = np.atleast_2d(box)
        candidate_participants, emb_dists = tracker.guess_id(player._last_frame_img, new_box)

        for ex in exclude:
            if ex in candidate_participants:
                emb_dists[candidate_participants.index(ex)] = np.inf
        emb_dist_ranking = np.argsort(emb_dists[:, 0])
        return list(emb_dists[emb_dist_ranking, 0]), list(np.array(candidate_participants)[emb_dist_ranking])

    def query_for_reid(emb_dists, candidate_participants):
        bibs = [format(part_id, '02x').lower() for part_id in candidate_participants]
        config_bibs = [format(runner_id, '02x').lower() for runner_id in race_config["participants"].keys()]
        bibs.extend([bib for bib in config_bibs if bib not in bibs])
        names = [race_config["participants"].get(int(bib, 16), '') for bib in bibs]
        emb_dists.extend([1 for _ in range(len(names) - len(emb_dists))])
        player.render()
        return ask_for_id([(bib, (name, f"{dist:.2f}")) for bib, name, dist in zip(bibs, names, emb_dists)],
                                       show_default=True, allow_other=True)

    def annotation_updated(annotation_id: str, new_annotation: np.ndarray, timecode, crossing=False, force=None):
        new_box = np.atleast_2d(np.array(
            [new_annotation[0][0], new_annotation[0][1], new_annotation[1][0], new_annotation[1][1], -1, 1.0, 0],
            dtype=np.float32))
        if annotation_id is None:
            emb_dists, candidate_participants = calculate_reid_distances(new_box)
            if force and emb_dists[0] < .15:
                # Force reid to the first candidate
                annotation_id = f"{candidate_participants[0]:02x}"
            else:
                annotation_id = query_for_reid(emb_dists, candidate_participants)
            if annotation_id is None:
                return False
            else:
                update_tracker(new_box, int(annotation_id, 16))
        new_box[:, 4] = int(annotation_id, 16)
        store.update_annotation(timecode.frames, Boxes(new_box, player._last_frame_img.shape[:2]), None,
                                [crossing], "human")
        return True

    def key_delegate(frame, frame_num, key, runner_id: str = None):
        if key == ord('\b'):
            # Jump back 10s
            player.seek_timecode_frame(frame_num - 10 * round(float(player.get_last_timecode().framerate)))
        elif key == 201:  # F12, since we can't detect the delete key
            # Jump forward 10s
            player.seek_timecode_frame(frame_num + 10 * round(float(player.get_last_timecode().framerate)))
        elif key == ord('`'):
            # Make a new note
            # Get runner id
            if runner_id is None:
                bib_name_pairs = [(format(runner_id, '02x').lower(), (name,)) for runner_id, name in
                                  race_config["participants"].items()]
                # Prompt the user to select an annotation to edit
                if runner_id is None:
                    runner_id = ask_for_id(bib_name_pairs)
                    if runner_id is None:
                        return None
            note = input("Enter note: ")

            store.update_notes(frame_num, int(runner_id, 16), note)
            return None
        if key == ord('e'):
            commands = {
                'd': 'delete',
                'c': 'crossing',
                'r': 'reassign',
                'q': 'cancel',
            }
            # Prompt the user to select an action
            print(f"Select an action for runner {runner_id}:")
            for key, value in commands.items():
                print(f"{key}: {value}")
            while True:
                actions = input("Action: ").strip()
                if 'q' in actions:
                    return None
                if len(actions) >= 1:
                    break
            for action in actions:
                key_delegate(frame, frame_num, ord(action), runner_id=runner_id)
                return None
            return None
        elif key == ord('d') or key == ord('c') or key == ord('D') or key == ord('r') or key == ord("R"):
            annotation = store.get_frame_annotation(frame_num, frame.shape[:2], source="human")
            boxes = annotation[0]
            ids = boxes[:, 4].astype(int)
            bibs = [format(runner_id, '02x') for runner_id in ids]
            names = [race_config["participants"].get(runner_id, None) for runner_id in ids]
            # Prompt the user to select an annotation to edit
            if runner_id is None:
                runner_id = ask_for_id([(bib.lower(), (name,)) for bib, name in zip(bibs, names)])
                if runner_id is None:
                    return frame
            if key == ord('d') or key == ord('D'):
                store.delete_frame_annotation(frame_num, runner_id)
                if key == ord('D'):
                    nearby = store.get_nearby(frame_num, buffer_s=5, runner_id=int(runner_id, 16))
                    logger.info(
                        f"Deleting {runner_id} {player.get_last_timecode()}, removing {len(nearby)} other annotations.")
                    for nearby_num in nearby:
                        store.delete_frame_annotation(nearby_num, runner_id)
            elif key == ord('c'):
                marked_crossing = store.mark_frame_crossing(frame_num, runner_id)
                # Check to see if we marked a crossing
                if marked_crossing:
                    nearby_crossings = store.get_nearby(frame_num, buffer_s=5, runner_id=int(runner_id, 16),
                                                  crossing=True)
                    if frame_num in nearby_crossings:
                        nearby_crossings.remove(frame_num)
                    logger.info(
                        f"Marked {runner_id} {player.get_last_timecode()} crossing, removing {len(nearby_crossings)} other crossings.")
                    for nearby_num in nearby_crossings:
                        store.mark_frame_crossing(nearby_num, runner_id,
                                                                      crossing=False)
                else:
                    logger.info(
                        f"Unmarked {runner_id} {player.get_last_timecode()} ({player.get_last_timecode().frames}) as crossing.")
            elif key == ord('r') or key == ord("R"):
                # Can be reassigned to anything, but null out current ID under the assumption we want a different result
                # FIXME: Occasional crasher, probably when reassigning with a single box in the frame
                emb_dists, candidate_participants = calculate_reid_distances(
                    boxes[np.where(ids == int(runner_id, 16))[0]], exclude=[int(runner_id, 16)])
                new_annotation_id = query_for_reid(emb_dists, candidate_participants)
                if new_annotation_id is None:
                    return None
                store.reassign_frame_annotation(frame_num, runner_id, new_annotation_id)

                if key == ord("R"):
                    # Area affect
                    nearby_with_id = store.get_nearby(frame_num, buffer_s=5, runner_id=int(runner_id, 16))
                    for nearby_num in nearby_with_id:
                        store.reassign_frame_annotation(nearby_num, runner_id,
                                                                            new_annotation_id)
            player.render()
            return None
        elif key == ord('[') or key == ord(']') or key == ord('{') or key == ord('}'):
            crossings_only = True if key == ord('{') or key == ord('}') else None
            if key == ord('[') or key == ord('{'):
                next_frame = store.scan_to_annotation(frame_num, previous=True, crossing=crossings_only, source="human")
            else:
                next_frame = store.scan_to_annotation(frame_num, previous=False, crossing=crossings_only,
                                                      source="human")

            if next_frame:
                logger.info(
                    f"Seeking to {next_frame} ({Timecode(player.get_last_timecode().framerate, frames=next_frame)})")
                success = player.seek_timecode_frame(next_frame)
                if not success:
                    logger.error(f"Failed to seek to frame {next_frame}.")
                    return None
                player._advance_frame()
                player.render()
                return None
            else:
                logger.info("No further annotations.")
                return None
        elif key == ord('9') or key == ord('0'):
            # Seek to line detection
            line_seg_pts = [race_config['finish_line'][0], race_config['finish_line'][1]]
            previous = key == ord('9')
            next_frame = store.scan_to_annotation(frame_num, previous=previous, source=args.detection_model,
                                                  custom_check=lambda x: any(
                line_segment_to_box_distance(line_seg_pts[0], line_seg_pts[1],
                                             ultralytics.utils.ops.xywhn2xyxy(
                                                 np.array([x["x_center"], x["y_center"], x["width"], x["height"]]),
                                                 *frame.shape[1::-1])) < 10))

            if next_frame:
                next_timecode = Timecode(player.get_last_timecode().framerate,
                                         frames=next_frame)
                logger.info(
                    f"Found a runner on the line at {next_frame} ({next_timecode}), seeking to it.")
                player.seek_timecode(next_timecode)
                player._advance_frame()
                player.render()
                return None
            logger.info("No further detections on the line.")
            return None
        elif key == ord('(') or key == ord(')'):
            # Track forward/backward
            line_seg_pts = [race_config['finish_line'][0], race_config['finish_line'][1]]
            # tracker.reset()
            start_frame = player.get_last_timecode().frames
            i = 0
            while True:
                detected_boxes, _, _, _ = store.get_frame_annotation(start_frame + i, source=args.detection_model)
                annotation = store.get_frame_annotation(start_frame + i,
                                             {"boxes": np.zeros((0, 7)), "kpts": None, "crossings": []})
                annotated_boxes = annotation["boxes"]
                annotated_crossings = annotation["crossings"]
                if i > 0:
                    # Ignore high IDs; we assume these are changeable in subsequent frames
                    low_id_mask = annotated_boxes[:, 4] <= 0xFF
                    if np.sum(~low_id_mask) > 0:
                        logger.info(f"Ommiting annotation for {hex(int(annotated_boxes[~low_id_mask][0, 4]))}")
                    annotated_boxes = annotated_boxes[low_id_mask]
                    annotated_crossings = np.array(annotated_crossings)[low_id_mask].tolist()
                if len(annotated_boxes) > 0:
                    # Calculate how much each detection overlaps any existing annotated box
                    ioa = bbox_ioa(detected_boxes[:, :4], annotated_boxes[:, :4])
                    # More than 30% overlap, and we throw out this detection as we assume the annotation has it covered
                    detected_boxes = detected_boxes[np.max(ioa, axis=1) < .3, :]

                null_id_detections = np.empty((len(detected_boxes), 7))
                null_id_detections[:, :4] = detected_boxes[:, :4]
                null_id_detections[:, -3] = -1
                null_id_detections[:, -2] = detected_boxes[:, -2]
                null_id_detections[:, -1] = detected_boxes[:, -1]
                combined_crossings = np.concatenate(
                    [annotated_crossings, np.full((len(null_id_detections)), False)])
                annotated_mask = np.full_like(combined_crossings, False, dtype=bool)
                annotated_mask[:len(annotation["crossings"])] = True

                combined = np.vstack([annotated_boxes, null_id_detections])
                # Check if any of the detected boxes are near the line
                on_line_mask = line_segment_to_box_distance(line_seg_pts[0], line_seg_pts[1], combined[:, :4]) < 10
                if any(on_line_mask):
                    as_boxes = Boxes(combined[on_line_mask], frame.shape[:2])
                    out = tracker.update(as_boxes, frame)

                    new_boxes = np.vstack([out[:, :7], combined[annotated_mask & ~on_line_mask, :7]])
                    new_crossings = np.concatenate(
                        (combined_crossings[on_line_mask], combined_crossings[annotated_mask & ~on_line_mask]))
                    boxes, kpts, crossings, sources = store.get_frame_annotation(start_frame + i, source="human")
                    boxes_to_save = Boxes(boxes, player._last_frame_img.shape)
                    store.save_annotation(start_frame + i, boxes_to_save, kpts, crossings, sources)
                    player.render()
                    if i != 0:
                        player.seek_time(str(Timecode(player.get_last_timecode().framerate,
                                                      frames=start_frame + i)))
                        player._advance_frame()
                        player.render()
                        # Check whether the tracked+on-the-line boxes have recent crosses
                        on_line_ids = out[:, 4].astype(int)
                        should_break = False
                        for on_line_id in on_line_ids:
                            crossing_frame_num = store.scan_to_annotation(start_frame + i, previous=True,
                                                                    runner_id=on_line_id,
                                                                    crossing=True, max_scan=300)
                            if crossing_frame_num is None:
                                logger.info(f"No crossing for {hex(on_line_id)}, stopping scan.")
                                should_break = True
                                break
                        if should_break:
                            break
                            return None
                else:
                    out_timecode = Timecode(player.get_last_timecode().framerate, frames=start_frame + i)
                    logger.info(f"Skipping {out_timecode} {out_timecode.frames}")
                    tracker.update(Boxes(np.zeros((0, 7)), frame.shape[:2]), frame)
                i += 1 if key == ord(')') else -1
        return None

    def click_delegate(frame, frame_num, click_pt, flags):
        # First, check if user clicked on any existing boxes
        frame_annotations = store.get_frame_annotation(frame_num, source="human")
        boxes = frame_annotations[0]
        boxes[:, :4] = ultralytics.utils.ops.xywhn2xyxy(boxes[:, :4], *frame.shape[1::-1])
        # Check if the click point is inside any of the boxes
        inside = [box[0] < click_pt[0] < box[2] and box[1] < click_pt[1] < box[3] for box in boxes]
        if len(inside) == 0 or not any(inside):
            pass
        elif np.sum(inside) > 1:
            print("Multiple boxes found, please click inside only one box.")
            return
        else:
            # By default, treat click on the box as start of an edit prompt on this box

            # If shift click, assume this is a crossing reassignment (very common)
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                key_delegate(frame, frame_num, ord('c'),
                             runner_id=format(boxes[np.where(inside)[0][0], 4].astype(int), '02x'))
                key_delegate(frame, frame_num, ord('R'),
                             runner_id=format(boxes[np.where(inside)[0][0], 4].astype(int), '02x'))
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                key_delegate(frame, frame_num, ord('c'),
                             runner_id=format(boxes[np.where(inside)[0][0], 4].astype(int), '02x'))
            else:
                key_delegate(frame, frame_num, ord('e'),
                             runner_id=format(boxes[np.where(inside)[0][0], 4].astype(int), '02x'))
            return None

        detections = store.get_frame_annotation(frame_num, source=args.detection_model)
        detected_boxes = detections[0]
        detected_boxes[:, :4] = ultralytics.utils.ops.xywhn2xyxy(detected_boxes[:, :4], *frame.shape[1::-1])

        player.render()
        # Check if the click point is inside any of the boxes
        inside = [box[0] < click_pt[0] < box[2] and box[1] < click_pt[1] < box[3] for box in detected_boxes]
        if len(inside) == 0 or not any(inside):
            return
        if np.sum(inside) > 1:
            print("Multiple boxes found, please click inside only one box.")
            return
        clicked_box = detected_boxes[np.where(inside)[0][0]]
        # Shift click makes the new box a crossing too
        player.annotation_updated(None, [clicked_box[0:2], clicked_box[2:4]], player.get_last_timecode(),
                                  crossing=((flags & cv2.EVENT_FLAG_SHIFTKEY) > 0),
                                  force=((flags & cv2.EVENT_FLAG_CTRLKEY) > 0))
        return

    player.click_delegate = click_delegate
    player.annotation_updated = annotation_updated
    player.pre_display = overlay_annotations
    player.key_delegate = key_delegate
    player.play()
    cv2.waitKey(0)
    logger.info("Saving tracker state...")
    joblib.dump(tracker, args.project / 'tracker.pkl')


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('project', type=pathlib.Path, default='data/exp')
    parser.add_argument('--source', type=pathlib.Path, nargs='+', required=True,
                        help='file paths')
    parser.add_argument('--seek-frame', type=int,
                        help='seek frame (timecode index from start) to start tracking')
    parser.add_argument('--seek-time', type=str, default=None, help='seek time to start tracking')
    parser.add_argument('--paused', action='store_true', help='start paused')
    parser.add_argument('--reid-model', type=pathlib.Path, default='osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--device', default='cuda',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--detection-model', type=str, default='detection',
                        help='Only display detections from this source')

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
