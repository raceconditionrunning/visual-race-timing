from copy import deepcopy

import cv2
import numpy as np
import torch
from timecode import Timecode
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Boxes, Masks, Probs, Keypoints, OBB
from ultralytics.utils.plotting import Annotator


def draw_annotation(
        boxes=None,
        masks=None,
        keypoints=None,
        obb=None,
        crossings=None,
        conf=None,
        labels=None,
        probs=None,
        colors=None,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
    ):
        boxes = Boxes(boxes, img.shape) if boxes is not None else None  # native size boxes
        masks = Masks(masks, img.shape) if masks is not None else None  # native size or imgsz masks
        probs = Probs(probs) if probs is not None else None
        keypoints = Keypoints(keypoints, img.shape) if keypoints is not None else None
        obb = OBB(obb, img.shape) if obb is not None else None

        is_obb = obb is not None
        pred_boxes, show_boxes = obb if is_obb else boxes, boxes
        pred_masks, show_masks = masks, masks
        pred_probs, show_probs = probs, probs
        annotator = Annotator(
            deepcopy(img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=labels,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                # Flip the reversed index...
                idx = len(pred_boxes) - i - 1
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                is_crossing = crossings[idx] if crossings is not None else None
                name = labels[idx] if labels else ""
                color = colors[idx] if colors else (255, 0, 0)
                if is_crossing:
                    color = (0, 255, 255)
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=color, rotated=is_obb)

        # Plot Pose results
        if keypoints is not None:
            for k in reversed(keypoints.data):
                annotator.kpts(k, img.shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result()


def render_timecode(timecode: Timecode, frame: np.ndarray):
    timecode_str = f'{timecode}'
    text_x = 10
    text_y = frame.shape[0] - 10
    cv2.putText(frame, timecode_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
