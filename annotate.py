#!/usr/bin/env python
import argparse
import pathlib
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import yaml
from timecode import Timecode
from ultralytics.engine.results import Boxes
from ultralytics.utils.metrics import bbox_ioa

from visual_race_timing.annotations import save_txt_annotation, \
    scan_to_annotation, get_nearby, mark_frame_crossing, reassign_frame_annotation, delete_frame_annotation, \
    remove_duplicates, build_crossing_map, update_annotation, load_annotations, load_notes, save_notes
from visual_race_timing.drawing import draw_annotation

from visual_race_timing.geometry import line_segment_to_box_distance
from visual_race_timing.prompts import ask_for_id
from visual_race_timing.tracker import get_crops, PartiallySupervisedTracker
from visual_race_timing.media_player import VideoPlayer, PhotoPlayer


def run(args):
    # Load all annotations
    annotations = load_annotations(args.project / 'annotations')
    candidate_boxes = load_annotations(args.project / 'detections')
    notes = load_notes(args.project / 'notes.tsv')

    # crossing_map = build_crossing_map(annotations)
    tracker_config = args.project / 'tracker_config.yaml'
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(cfg)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']
    cfg.match_thresh = .8
    tracker = PartiallySupervisedTracker(args.reid_model, cfg, device="cuda")

    # Load race configuration from yaml
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(args.source) == 1 and args.source[0].is_dir():
        player = PhotoPlayer(args.source[0], args.paused)
    else:
        player = VideoPlayer(args.source, args.paused)
    if args.seek_frame:
        args.seek_time = str(Timecode(player.get_current_time().framerate, frames=args.seek_timecode_frame))
    if args.seek_time:
        player.seek_to_time(args.seek_time)

    def overlay_annotations(frame, frame_num):
        frame_annotations = annotations.get(frame_num, None)
        frame_boxes = candidate_boxes.get(frame_num, None)
        frame_notes = notes.get(frame_num, None)

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
            ids = boxes[:, 4].astype(int)
            bibs = [format(runner_id, '02x') for runner_id in ids]
            names = [race_config['participants'].get(runner_id, None) for runner_id in ids]
            names = [name.split(" ")[0] if name else bib for bib, name in zip(bibs, names)]
            labels = [f"{bib}{' ' + name if name else ''}" for bib, name in zip(bibs, names)]
            frame = draw_annotation(img=frame, boxes=boxes, keypoints=kpts, crossings=crossings, labels=labels,
                                    kpt_radius=2, line_width=1)
        if frame_notes is not None:
            for i, (runner_id, note) in enumerate(frame_notes.items()):
                frame = cv2.putText(frame, f"{runner_id}: {note}", (10 + 10 * i, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1,
                                    cv2.LINE_AA)
        return frame

    def query_for_reid(box, timecode, exclude: List[int] = []):
        new_box = np.atleast_2d(box)
        candidate_participants, emb_dists = tracker.guess_id(player.current_frame_img, new_box)

        for ex in exclude:
            if ex in candidate_participants:
                emb_dists[candidate_participants.index(ex)] = np.inf
        emb_dist_ranking = np.argsort(emb_dists[:, 0])
        dists = list(emb_dists[emb_dist_ranking, 0])
        bibs = [format(candidate_participants[rank_i], '02x').lower() for rank_i in emb_dist_ranking]
        config_bibs = [format(runner_id, '02x').lower() for runner_id in race_config["participants"].keys()]
        bibs.extend([bib for bib in config_bibs if bib not in bibs])
        names = [race_config["participants"].get(int(bib, 16), '') for bib in bibs]
        dists.extend([1 for _ in range(len(names) - len(dists))])
        player.render()
        annotation_id_str = ask_for_id([(bib, (name, f"{dist:.2f}")) for bib, name, dist in zip(bibs, names, dists)],
                                       show_default=True, allow_other=True)
        if annotation_id_str is not None:
            # Only use crops larger than 24x24
            if new_box[0, 2] - new_box[0, 0] < 24 or new_box[0, 3] - new_box[0, 1] < 24:
                print("Box too small, ignoring.")
            else:
                tracker.update_participant_features(player.current_frame_img, new_box, int(annotation_id_str, 16))
            return annotation_id_str

    def annotation_updated(annotation_id: str, new_annotation: np.ndarray, timecode, crossing=False):
        new_box = np.atleast_2d(np.array(
            [new_annotation[0][0], new_annotation[0][1], new_annotation[1][0], new_annotation[1][1], -1, 1.0, 0],
            dtype=np.float32))
        if annotation_id is None:
            annotation_id = query_for_reid(new_box, timecode)
            if annotation_id is None:
                return False
        new_box[:, 4] = int(annotation_id, 16)
        update_annotation(annotations, timecode.frames, Boxes(new_box, player.current_frame_img.shape[:2]), None,
                          [crossing])
        updated_annotation = annotations[timecode.frames]
        boxes_to_save = updated_annotation['boxes']
        boxes_to_save = Boxes(boxes_to_save, player.current_frame_img.shape)
        save_txt_annotation(boxes_to_save, updated_annotation['kpts'], updated_annotation['crossings'],
                            args.project / "annotations" / f'frame_{timecode.frames}.txt', )
        return True

    def key_delegate(frame, frame_num, key, runner_id: str = None):
        if key == ord('`'):
            # Make a new note
            # Get runner id
            if runner_id is None:
                bib_name_pairs = [(format(runner_id, '02x').lower(), (name,)) for runner_id, name in
                                  race_config["participants"].items()]
                # Prompt the user to select an annotation to edit
                if runner_id is None:
                    runner_id = ask_for_id(bib_name_pairs)
                    if runner_id is None:
                        return
            note = input("Enter note: ")

            notes[frame_num][runner_id] = note
            save_notes(notes, args.project / 'notes.tsv')
            return
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
                    return
                if len(actions) >= 1:
                    break
            for action in actions:
                key_delegate(frame, frame_num, ord(action), runner_id=runner_id)
        elif key == ord('d') or key == ord('c') or key == ord('D') or key == ord('r') or key == ord("R"):
            modified = []
            annotation = annotations.get(frame_num, None)
            boxes = annotations[frame_num]['boxes']
            ids = boxes[:, 4].astype(int)
            bibs = [format(runner_id, '02x') for runner_id in boxes]
            names = [race_config["participants"].get(runner_id, None) for runner_id in ids]
            # Prompt the user to select an annotation to edit
            if runner_id is None:
                runner_id = ask_for_id([(bib.lower(), (name,)) for bib, name in zip(bibs, names)])
                if runner_id is None:
                    return frame
            if key == ord('d') or key == ord('D'):
                annotations[frame_num] = delete_frame_annotation(annotation, runner_id)
                modified.append(frame_num)
                if key == ord('D'):
                    nearby = get_nearby(annotations, frame_num, buffer_s=5, runner_id=int(runner_id, 16))
                    print(
                        f"Deleting {runner_id} {player.get_current_time()}, removing {len(nearby)} other annotations.")
                    for nearby_num in nearby:
                        annotations[nearby_num] = delete_frame_annotation(annotations[nearby_num], runner_id)
                        modified.append(nearby_num)
            elif key == ord('c'):
                annotations[frame_num] = mark_frame_crossing(annotation, runner_id)
                modified.append(frame_num)
                # Check to see if we marked a crossing
                if annotations[frame_num]['crossings'][
                    np.where(annotations[frame_num]['boxes'][:, 4] == int(runner_id, 16))[0][0]]:

                    nearby_crossings = get_nearby(annotations, frame_num, buffer_s=5, runner_id=int(runner_id, 16),
                                                  crossing=True)
                    if frame_num in nearby_crossings:
                        nearby_crossings.remove(frame_num)
                    print(
                        f"Marked {runner_id} {player.get_current_time()} crossing, removing {len(nearby_crossings)} other crossings.")
                    for nearby_num in nearby_crossings:
                        annotations[nearby_num] = mark_frame_crossing(annotations[nearby_num], runner_id,
                                                                      crossing=False)
                        modified.append(nearby_num)
                else:
                    print(
                        f"Unmarked {runner_id} {player.get_current_time()} ({player.get_current_time().frames}) as crossing.")
            elif key == ord('r') or key == ord("R"):
                # Can be reassigned to anything
                new_annotation_id = query_for_reid(boxes[np.where(annotation['boxes'][:, 4] == int(runner_id, 16))[0]],
                                                   player.get_current_time(), exclude=[int(runner_id, 16)])
                if new_annotation_id is None:
                    return
                annotations[frame_num] = reassign_frame_annotation(annotation, runner_id, new_annotation_id)
                modified.append(frame_num)

                if key == ord("R"):
                    nearby_with_id = get_nearby(annotations, frame_num, buffer_s=5, runner_id=int(runner_id, 16))
                    for nearby_num in nearby_with_id:
                        annotations[nearby_num] = reassign_frame_annotation(annotations[nearby_num], runner_id,
                                                                            new_annotation_id)
                        modified.append(nearby_num)

            for frame_num in modified:
                # for frame_num in annotations.keys():
                boxes = annotations[frame_num]['boxes']
                kpts = annotations[frame_num]['kpts']
                crossings = annotations[frame_num]['crossings']
                boxes_to_save = Boxes(boxes, player.current_frame_img.shape)
                save_txt_annotation(boxes_to_save, kpts, crossings,
                                    args.project / "annotations" / f'frame_{frame_num}.txt',
                                    replace=True)
            player.render()
        elif key == ord('[') or key == ord(']') or key == ord('{') or key == ord('}'):
            crossings_only = True if key == ord('{') or key == ord('}') else None
            if key == ord('[') or key == ord('{'):
                next_frame = scan_to_annotation(annotations, frame_num, previous=True, crossing=crossings_only)
            else:
                next_frame = scan_to_annotation(annotations, frame_num, previous=False, crossing=crossings_only)

            if next_frame:
                print(f"Seeking to {player.get_current_time()} ({player.get_current_time().frames})")
                player.seek_to_time(str(Timecode(player.get_current_time().framerate, frames=next_frame)))
                player._advance_frame()
                player.render()
            else:
                print("No further annotations.")
        elif key == ord('9') or key == ord('0'):
            # Seek to line detection
            line_seg_pts = [race_config['finish_line'][0], race_config['finish_line'][1]]
            previous = key == ord('9')
            next_frame = scan_to_annotation(candidate_boxes, frame_num, previous=previous, custom_check=lambda x: any(
                line_segment_to_box_distance(line_seg_pts[0], line_seg_pts[1],
                                             x["boxes"][:, :4]) < 10))

            if next_frame:
                player.seek_to_time(str(Timecode(player.get_current_time().framerate,
                                                 frames=next_frame)))
                player._advance_frame()
                player.render()
                print(f"Found a runner on the line at {player.get_current_time()} ({player.get_current_time().frames})")
                return
            print("No further detections on the line.")
        elif key == ord('(') or key == ord(')'):
            # Track forward/backward
            line_seg_pts = [race_config['finish_line'][0], race_config['finish_line'][1]]
            # tracker.reset()
            start_frame = player.get_current_time().frames
            i = 0
            while True:
                detections = candidate_boxes.get(start_frame + i,
                                                 {'boxes': np.zeros((0, 7)), 'kpts': None,
                                                  'crossings': []})
                detected_boxes = detections['boxes']

                annotation = annotations.get(start_frame + i,
                                             {"boxes": np.zeros((0, 7)), "kpts": None, "crossings": []})
                annotated_boxes = annotation["boxes"]
                annotated_crossings = annotation["crossings"]
                if i > 0:
                    # Ignore high IDs; we assume these are changeable in subsequent frames
                    low_id_mask = annotated_boxes[:, 4] <= 0xFF
                    if np.sum(~low_id_mask) > 0:
                        print(f"Ommiting annotation for {hex(int(annotated_boxes[~low_id_mask][0, 4]))}")
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

                    annotation["boxes"] = np.vstack([out[:, :7], combined[annotated_mask & ~on_line_mask, :7]])
                    annotation["crossings"] = np.concatenate(
                        (combined_crossings[on_line_mask], combined_crossings[annotated_mask & ~on_line_mask]))
                    annotations[start_frame + i] = annotation
                    boxes = annotations[start_frame + i]['boxes']
                    kpts = annotations[start_frame + i]['kpts']
                    crossings = annotations[start_frame + i]['crossings']
                    boxes_to_save = Boxes(boxes, player.current_frame_img.shape)
                    save_txt_annotation(boxes_to_save, kpts, crossings,
                                        args.project / "annotations" / f'frame_{start_frame + i}.txt',
                                        replace=True)
                    player.render()
                    if i != 0:
                        player.seek_to_time(str(Timecode(player.get_current_time().framerate,
                                                         frames=start_frame + i)))
                        player._advance_frame()
                        player.render()
                        # Check whether the tracked+on-the-line boxes have recent crosses
                        on_line_ids = out[:, 4].astype(int)
                        should_break = False
                        for on_line_id in on_line_ids:
                            crossing_frame_num = scan_to_annotation(annotations, start_frame + i, previous=True,
                                                                    runner_id=on_line_id,
                                                                    crossing=True, max_scan=300)
                            if crossing_frame_num is None:
                                print(f"No crossing for {hex(on_line_id)}, stopping scan.")
                                should_break = True
                                break
                        if should_break:
                            break
                else:
                    out_timecode = Timecode(player.get_current_time().framerate, frames=start_frame + i)
                    print(f"Skipping {out_timecode} {out_timecode.frames}")
                    tracker.update(Boxes(np.zeros((0, 7)), frame.shape[:2]), frame)
                i += 1 if key == ord(')') else -1

    def click_delegate(frame, frame_num, click_pt, flags):
        # First, check if user clicked on any existing boxes
        if frame_num in annotations:
            frame_annotations = annotations[frame_num]
            boxes = frame_annotations['boxes']
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

        detections = candidate_boxes.get(player.get_current_time().frames, {'boxes': np.zeros((0, 7)), 'kpts': None,
                                                                            'crossings': []})
        detected_boxes = detections['boxes']

        if len(detected_boxes) != 0:
            candidate_boxes[frame_num] = {'boxes': detected_boxes, 'kpts': None, 'crossings': [0] * len(detected_boxes)}
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
        player.annotation_updated(None, [clicked_box[0:2], clicked_box[2:4]], player.get_current_time(),
                                  crossing=((flags & cv2.EVENT_FLAG_SHIFTKEY) > 0))
        return

    player.click_delegate = click_delegate
    player.annotation_updated = annotation_updated
    player.pre_display = overlay_annotations
    player.key_delegate = key_delegate
    player.play()
    cv2.waitKey(0)


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

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
