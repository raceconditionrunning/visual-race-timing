import csv
import pathlib
from collections import defaultdict
from typing import Dict
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import ultralytics.utils
from sortedcontainers import SortedDict
from tqdm import tqdm
from ultralytics.engine.results import Boxes, Keypoints


def save_txt_annotation(boxes: Boxes, kpts: Keypoints, crossings, txt_file: pathlib.Path, replace=False):
    texts = []
    # Detect/segment/pose
    for j, d in enumerate(boxes):
        c, conf, track_id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        line = (c, *(d.xywhn.flatten()), conf)
        line += ((-1,) if track_id is None else (track_id,)) + (int(crossings[j]),)
        if kpts is not None:
            kpt = np.concatenate((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
            line += (*kpt.reshape(-1).tolist(),)
        texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        mode = "w" if replace else "a"
        with open(txt_file, mode) as f:
            f.writelines(text + "\n" for text in texts)
    elif replace and Path(txt_file).exists():
        # Delete empty file
        Path(txt_file).unlink()


def load_txt_annotation(txt_file: pathlib.Path, img_size=(1920, 1080)):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    boxes = []
    kpts = []
    crossings = []
    for line in lines:
        line = line.split()
        obj_class = int(line[0])
        if len(line) >= 8:
            xywhn = np.array([float(x) for x in line[1:5]])
            conf = float(line[5])
            track_id = int(line[6])
            crossings.append(bool(line[7] != "0"))
            """if line[7] != "1" and line[7] != "0":
                print(f"Invalid line: {line} in {txt_file}")"""
            xyxy = ultralytics.utils.ops.xywhn2xyxy(xywhn, *img_size)
        if len(line) >= 59:
            kpt = np.array([float(x) for x in line[8:59]]).reshape(17, 3)
            kpt[:, 0] *= img_size[0]
            kpt[:, 1] *= img_size[1]
            # conf = float(line[56])
            kpts.append(kpt)

        if track_id is not None:
            box = np.hstack((xyxy, [track_id], [conf], [obj_class]))
        else:
            box = np.hstack((xyxy, [conf], [obj_class]))
        boxes.append(box)
    return np.array(boxes), np.array(kpts) if kpts else None, np.array(crossings)


def load_annotations(annotations_path: pathlib.Path) -> SortedDict[int, Dict[str, any]]:
    annotations = SortedDict()
    if not annotations_path.exists():
        return annotations
    for txt_file in tqdm(annotations_path.glob("frame_*.txt")):
        # Extract frame number from the txt file name ("frame_NUMBER.txt")
        frame = int(txt_file.stem.split('_')[-1])
        boxes, kpts, crossings = load_txt_annotation(txt_file)
        boxes, kpts, crossings = remove_duplicates(boxes, kpts, crossings)
        annotations[frame] = {'boxes': boxes, 'kpts': kpts, 'crossings': crossings}
    return annotations


def load_notes(notes_path: pathlib.Path) -> Dict[int, Dict[str, str]]:
    notes = defaultdict(dict)
    if not notes_path.exists():
        return notes

    with open(notes_path, "r") as f:
        # Each line is a note. First column is frame num, second is runner ID, second column is note
        lines = csv.reader(f, delimiter='\t')
        for line in lines:
            frame = int(line[0])
            note = line[2]
            runner_id = format(int(line[1]), '02x')
            notes[frame][runner_id] = note
    return notes


def save_notes(notes: Dict[int, Dict[str, str]], notes_path: pathlib.Path):
    with open(notes_path, "w") as f:
        writer = csv.writer(f, delimiter='\t')
        ordered_frame_nums = sorted(list(notes.keys()))
        for frame in ordered_frame_nums:
            frame_notes = notes[frame]
            for runner_id, note in frame_notes.items():
                writer.writerow([frame, int(runner_id, 16), note])


def distance_to_zeros(arr: np.ndarray) -> np.ndarray:
    large_number = arr.shape[1]
    for batch_idx in range(arr.shape[0]):
        x = arr[batch_idx]
        indices = np.arange(x.size)
        zeroes = x == 0
        if not any(zeroes):
            arr[batch_idx] = np.full_like(x, large_number)
            continue
        forward = indices - np.maximum.accumulate(indices * zeroes)  # forward distance
        forward[np.cumsum(zeroes) == 0] = x.size - 1  # handle absence of zero from edge
        forward = forward * (x != 0)  # set zero positions to zero

        zeroes = zeroes[::-1]
        backward = indices - np.maximum.accumulate(indices * zeroes)  # backward distance
        backward[np.cumsum(zeroes) == 0] = x.size - 1  # handle absence of zero from edge
        backward = backward[::-1] * (x != 0)  # set zero positions to zero

        sign_should_flip = forward < backward
        arr[batch_idx] = np.minimum(forward, backward)  # closest distance (minimum)
        arr[batch_idx][sign_should_flip] *= -1
    return arr


def build_crossing_map(annotations: Dict[int, any]) -> np.ndarray:
    # For each index, for each runner, how many frames away and in what direction is the nearest crossing
    # -1 if no crossing
    # 0 if this frame is a crossing
    all_frame_nums = list(annotations.keys())
    all_frame_nums.sort()
    large_number = len(all_frame_nums)
    all_runner_ids = set()
    for frame_num in all_frame_nums:
        all_runner_ids.update(annotations[frame_num]['boxes'][:, 4].astype(int))
    all_runner_ids = sorted(list(all_runner_ids))
    crossing_map = np.full((len(all_runner_ids), len(all_frame_nums)), large_number, dtype=int)
    for i, frame_num in enumerate(all_frame_nums):
        for j, box in enumerate(annotations[frame_num]['boxes']):
            if annotations[frame_num]['crossings'][j]:
                runner_id = int(box[4])
                runner_idx = all_runner_ids.index(runner_id)
                crossing_map[runner_idx, i] = 0

    distance_map = distance_to_zeros(crossing_map)
    valid_idx = np.where(distance_map < large_number)
    pointed_to = np.zeros_like(distance_map)
    pointed_to[valid_idx] = valid_idx[1] + distance_map[valid_idx]
    all_frame_nums = np.array(all_frame_nums)
    distance_map[valid_idx] = all_frame_nums[pointed_to[valid_idx].flatten()] - all_frame_nums[valid_idx[1]]

    return distance_map


def scan_to_annotation(annotations: SortedDict[int, any], from_frame_num: int, runner_id: int | None = None,
                       crossing=None, previous=False, custom_check=None, max_scan=None) -> Optional[int]:
    if previous:
        to_end_index = annotations.bisect_left(from_frame_num)  # May include from_frame_num
        to_start_index = 0
        if max_scan is not None:
            min_frame_num = from_frame_num - max_scan
            to_start_index = max(to_start_index, annotations.bisect_left(min_frame_num))  # Left bisect result is >=0

    else:
        to_start_index = annotations.bisect(from_frame_num)
        to_end_index = len(annotations)
        if max_scan is not None:
            max_frame_num = from_frame_num + max_scan
            to_end_index = min(to_end_index,
                               annotations.bisect_right(max_frame_num))  # Max index might be off end, but that's fine

    for frame_num in annotations.islice(to_start_index, to_end_index, reverse=previous):
        if frame_num == from_frame_num:
            continue
        annotation = annotations[frame_num]
        if len(annotation['boxes']) == 0:
            continue
        custom_check_result = custom_check(annotation) if custom_check is not None else True
        if not custom_check_result:
            continue
        if runner_id is not None and np.where(annotation['boxes'][:, 4] == runner_id)[0].size > 0:
            # Looking for a specific runner
            if crossing is None:
                return frame_num
            elif annotation['crossings'][np.where(annotation['boxes'][:, 4] == runner_id)[0][0]] == crossing:
                return frame_num
        elif runner_id is None and crossing is not None:
            # Annotation with any runner crossing is fine
            if crossing and any(annotation['crossings']):
                return frame_num
            elif not crossing and not any(annotation['crossings']):
                return frame_num
        elif runner_id is None and crossing is None:
            # Any annotation flies!
            return frame_num
    return None


def get_nearby(annotations: Dict[int, any], to_frame_num: int, buffer_s: int = 5, runner_id: int = None,
               crossing=None) -> List[int]:
    # Get the nearest crossing to the current frame
    nearby = []
    start_frame = to_frame_num - buffer_s * 30
    end_frame = to_frame_num + buffer_s * 30
    for frame_num in range(start_frame, end_frame):
        if frame_num not in annotations:
            continue
        annotation = annotations[frame_num]
        # Check if the runner_id is in the annotations
        if runner_id is not None:
            if runner_id not in annotation['boxes'][:, 4]:
                continue
            if crossing is not None:
                if crossing != annotation['crossings'][np.where(annotation['boxes'][:, 4] == runner_id)[0][0]]:
                    continue
        nearby.append(frame_num)

    return nearby


def delete_frame_annotation(annotation, runner_id):
    boxes = annotation['boxes']
    kpts = annotation['kpts']
    crossings = annotation['crossings']
    # Delete the annotation
    annotation_idx = np.where(boxes[:, 4] == int(runner_id, 16))[0][0]
    boxes = np.delete(boxes, annotation_idx, axis=0)
    if kpts:
        kpts = np.delete(kpts, annotation_idx, axis=0)
    crossings = np.delete(crossings, annotation_idx, axis=0)
    return {'boxes': boxes, 'kpts': kpts, 'crossings': crossings}


def mark_frame_crossing(annotation, runner_id: str, crossing=None):
    # Mark this frame as a crossing
    annotation_idx = np.where(annotation["boxes"][:, 4] == int(runner_id, 16))
    if len(annotation_idx) == 0:
        print(f"Runner {runner_id} not found in annotation")
        return annotation
    annotation_idx = annotation_idx[0][0]
    annotation['crossings'][annotation_idx] = not annotation['crossings'][
        annotation_idx] if crossing is None else crossing
    return annotation


def update_annotation(annotations: Dict[int, any], frame_num: int, boxes: Boxes, kpts: Keypoints, crossings):
    if frame_num in annotations:
        # Check if there's already an annotation matching this id
        existing_annotation = annotations[frame_num]
        existing_boxes = existing_annotation['boxes']
        for i, new_box in enumerate(boxes):
            runner_id = int(new_box.data[0, 4])
            if runner_id in existing_boxes[:, 4]:
                # Get the existing index
                annotation_idx = np.where(existing_boxes[:, 4] == runner_id)[0][0]
                existing_boxes[annotation_idx] = new_box.data
            else:
                # Add the new annotation at the end
                existing_annotation["boxes"] = np.vstack((existing_boxes, new_box.data))
                if existing_annotation['kpts'] is not None:
                    existing_annotation['kpts'] = np.vstack((existing_annotation['kpts'], np.zeros((17, 3))))
                existing_annotation['crossings'] = np.concatenate((existing_annotation['crossings'], [True]))
            annotations[frame_num] = existing_annotation
    else:
        # Chop the internal id off the track result
        annotations[frame_num] = {'boxes': boxes.data, 'kpts': None if kpts is None else kpts.data,
                                  'crossings': crossings}
    return annotations[frame_num]


def reassign_frame_annotation(annotation, from_id, to_id):
    annotation_idx = np.where(annotation['boxes'][:, 4] == int(from_id, 16))[0][0]
    annotation['boxes'][annotation_idx, 4] = int(to_id, 16)
    return annotation


def remove_duplicates(boxes: Boxes, kpts: Keypoints, crossings):
    # Remove any boxes that have a duplicated non-zero track_id
    track_ids = boxes[:, 4]
    unique_ids = np.unique(track_ids)
    for track_id in unique_ids:
        if track_id == -1:
            continue
        track_ids = boxes[:, 4]
        track_idx = np.where(track_ids == track_id)[0]
        if len(track_idx) > 1:
            # Remove the box with the lowest confidence
            min_conf_idx = np.argmin(boxes[track_idx, 5])
            remove_idx = np.delete(track_idx, min_conf_idx)
            boxes = np.delete(boxes, remove_idx, axis=0)
            if kpts is not None:
                kpts = np.delete(kpts, remove_idx, axis=0)
            crossings = np.delete(crossings, remove_idx, axis=0)
    return boxes, kpts, crossings


def offset_with_crop(boxes: Boxes, kpts: Keypoints, crop: List[int], uncropped_size: Tuple[int, int]):
    # Offset the boxes and keypoints by the crop
    new_boxes = boxes.cpu().numpy().data
    new_kpts = kpts.cpu().numpy().data if kpts is not None else None
    # Offset the boxes (they're xyxy) by the number of preceding cropped pixels
    # Crop is [w, h, x, y]
    new_boxes[:, [0, 2]] += crop[2]
    new_boxes[:, [1, 3]] += crop[3]
    new_boxes = Boxes(new_boxes, uncropped_size)

    # Offset the keypoints
    if new_kpts is not None:
        new_kpts = new_kpts.reshape((-1, 17, 3))
        for i in range(len(new_kpts)):
            new_kpts[i][:, 0] += crop[2]
            new_kpts[i][:, 1] += crop[3]
        new_kpts = Keypoints(new_kpts, uncropped_size)
    return new_boxes, new_kpts
