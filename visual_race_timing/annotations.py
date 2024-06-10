import pathlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import ultralytics.utils
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
            crossings.append(bool(line[7] == "1"))
            xyxy = ultralytics.utils.ops.xywhn2xyxy(xywhn, *img_size)
        if len(line) >= 59:
            kpt = np.array([float(x) for x in line[8:59]]).reshape(17, 3)
            kpt[:, 0] *= img_size[0]
            kpt[:, 1] *= img_size[1]
            #conf = float(line[56])
            kpts.append(kpt)

        if track_id is not None:
            box = np.hstack((xyxy, [track_id], [conf], [obj_class]))
        else:
            box = np.hstack((xyxy, [conf], [obj_class]))
        boxes.append(box)
    return np.array(boxes), np.array(kpts) if kpts else None, np.array(crossings)


def get_nearest_crossing(annotations: OrderedDict[int, any], to_frame_num: int, runner_id: int = None,
                         buffer_s: int = 5) -> Optional[int]:
    # Get the nearest crossing to the current frame
    nearest_crossing = int(1e12)
    start_frame = to_frame_num - buffer_s * 30
    end_frame = to_frame_num + buffer_s * 30
    for frame_num in range(start_frame, end_frame):
        if frame_num not in annotations:
            continue
        annotation = annotations[frame_num]
        # Check if the runner_id is in the annotations
        if runner_id is not None:
            if runner_id not in annotation['crossings']:
                continue
        # Get the distance to the current frame
        distance = abs(frame_num - to_frame_num)
        if distance < nearest_crossing:
            nearest_crossing = distance

    return nearest_crossing


def offset_with_crop(boxes: Boxes, kpts: Keypoints, crop: List[int], uncropped_size: Tuple[int, int]):
    # Offset the boxes and keypoints by the crop
    new_boxes = boxes.cpu().numpy().data
    new_kpts = kpts.cpu().numpy().data if kpts is not None else None
    # Offset the boxes (they're xyxy) by the number of preceding cropped pixels
    # Crop is [w, h, x, y]
    new_boxes[:, [0,2]] += crop[2]
    new_boxes[:, [1,3]] += crop[3]
    new_boxes = Boxes(new_boxes, uncropped_size)

    # Offset the keypoints
    if new_kpts is not None:
        new_kpts = new_kpts.reshape((-1, 17, 3))
        for i in range(len(new_kpts)):
            new_kpts[i][:, 0] += crop[2]
            new_kpts[i][:, 1] += crop[3]
        new_kpts = Keypoints(new_kpts, uncropped_size)
    return new_boxes, new_kpts
