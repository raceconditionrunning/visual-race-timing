from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
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
                c, conf, id = int(d.cls), float(d.conf) if conf is not None else None, None if d.id is None else int(
                    d.id.item())
                is_crossing = crossings[idx] if crossings is not None else None
                name = labels[idx] if labels else ""
                color = colors[idx] if colors else (0, 0, 255)
                txt_color = (0, 0, 0)
                if is_crossing:
                    color = (0, 0, 0)
                    txt_color = (255, 255, 255)
                label = (f"{name} {conf:.2f}" if conf else name)
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=color, txt_color=txt_color, rotated=is_obb)

        # Plot Pose results
        if keypoints is not None:
            for k in reversed(keypoints.data):
                annotator.kpts(k, img.shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result()


def get_monospace_font_path():
    """
    Get the path to the best available monospace font with fallback chain:
    Ubuntu Mono -> Menlo (macOS) -> Consolas (Windows) -> system default
    """
    font_candidates = []

    # Ubuntu Mono (preferred)
    font_candidates.extend([
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-Regular.ttf",
        "/usr/share/fonts/TTF/UbuntuMono-Regular.ttf",
        "/System/Library/Fonts/UbuntuMono-Regular.ttf"
    ])

    # macOS Menlo
    font_candidates.extend([
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf"
    ])

    # Windows Consolas
    font_candidates.extend([
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/Consolas.ttf",
        "C:/Windows/Fonts/cour.ttf",  # Courier New
        "C:/Windows/Fonts/courbd.ttf"
    ])

    # Linux alternatives
    font_candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf"
    ])

    # Check each candidate
    for font_path in font_candidates:
        if os.path.isfile(font_path):
            return font_path

    # If no specific font found, return None to use a generic fallback
    return None

def render_timecode(timecode: Timecode, frame: np.ndarray, out_frame: np.ndarray = None) -> np.ndarray:
    timecode_str = f'{timecode}'
    margin = int(frame.shape[0] / 100)
    text_x = margin
    if out_frame is None:
         out_frame = np.array(frame.copy())

    # White color for 8-bit or 16-bit
    if frame.dtype == np.uint16:
        white_color = (65535, 65535, 65535)
        black_color = (0, 0, 0)
    else:
        white_color = (255, 255, 255)
        black_color = (0, 0, 0)

    font_scale = frame.shape[0] / 2000
    thickness = max(1, int(font_scale * 2))
    text_y_cv = frame.shape[0] - margin  # OpenCV uses bottom-left origin

    cv2.putText(out_frame, timecode_str, (text_x, text_y_cv),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                thickness * 4, lineType=cv2.LINE_AA)
    cv2.putText(out_frame, timecode_str, (text_x, text_y_cv),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, white_color[:3],
                thickness, lineType=cv2.LINE_AA)

    # Frame number
    frame_number_str = f'{timecode.frames}'
    frame_number_width = cv2.getTextSize(frame_number_str, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, thickness)[0][0]
    text_x_right = int(frame.shape[1] - frame_number_width - margin)

    cv2.putText(out_frame, frame_number_str, (text_x_right, text_y_cv),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                thickness * 4, lineType=cv2.LINE_AA)
    cv2.putText(out_frame, frame_number_str, (text_x_right, text_y_cv),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, white_color[:3],
                thickness, lineType=cv2.LINE_AA)

    return out_frame


def render_timecode_pil(timecode: Timecode, frame: np.ndarray, out_frame: np.ndarray = None) -> np.ndarray:
    """Render timecode and frame number on a frame using PIL for custom fonts"""
    timecode_str = f'{timecode}'
    font_size = int(frame.shape[0] / 60)  # Font size based on frame height
    margin = int(font_size / 2)

    if out_frame is None:
        out_frame = np.array(frame.copy())

    # Load font
    font_path = get_monospace_font_path()
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # White color for 8-bit or 16-bit
    if frame.dtype == np.uint16:
        white_color = (65535, 65535, 65535)
        black_color = (0, 0, 0)
    else:
        white_color = (255, 255, 255)
        black_color = (0, 0, 0)
    # Draw timecode
    text_x = margin
    text_y = frame.shape[0] - margin  # PIL uses top-left origin
    timecode_overlay = create_text_overlay(timecode_str, font=font, font_size=font_size, text_color=white_color, outline_color=black_color)
    frame_number_str = f'{timecode.frames}'
    frame_number_overlay = create_text_overlay(frame_number_str, font=font, font_size=font_size, text_color=white_color, outline_color=black_color)
    out = composite_16bit_overlay(out_frame, timecode_overlay, text_x, text_y)
    text_x = int(frame.shape[1] - frame_number_overlay.width - margin)
    out = composite_16bit_overlay(out, frame_number_overlay, text_x, text_y)
    out_frame[:] = out
    return out_frame


def create_text_overlay(text: str, font, font_size: int = 24,
                        text_color: tuple = (255, 255, 255, 255),
                        outline_color: tuple = (0, 0, 0, 255)) -> tuple:
    """Create a text overlay with outline effect"""


    # Get text dimensions
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    # Our drop frame timecodes have a descender (;), but the frame number does not. We use a fixed height so they sit on the same baseline.
    text_height = int(font_size * 1.5) #bbox[3] - bbox[1]

    # Add padding for outline
    outline_thickness = max(1, font_size // 20)
    padding = outline_thickness * 2

    # Create RGBA image for text with padding
    text_img = Image.new('RGBA', (text_width + padding * 2, text_height + padding * 2),
                         (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(text_img)

    # Draw outline (black text offset in multiple directions)
    text_x = padding
    text_y = padding

    for dx in range(-outline_thickness, outline_thickness + 1):
        for dy in range(-outline_thickness, outline_thickness + 1):
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), text,
                          font=font, fill=outline_color)

    # Draw main text (white)
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    return text_img


def composite_16bit_overlay(frame_16bit: np.ndarray, overlay_pil: Image.Image, x: int, y: int) -> np.ndarray:
    """Composite PIL overlay onto 16-bit frame using alpha blending"""

    # Convert PIL overlay to numpy array (8-bit RGBA)
    overlay_8bit = np.array(overlay_pil)  # Shape: (H, W, 4) - RGBA

    # Scale overlay to 16-bit
    overlay_16bit = (overlay_8bit.astype(np.uint32) * 257).astype(np.uint16)

    frame_height, frame_width = frame_16bit.shape[:2]
    overlay_height, overlay_width = overlay_16bit.shape[:2]

    # Ensure overlay fits within frame bounds
    x = max(0, min(x, frame_width - overlay_width))
    y = max(0, min(y, frame_height - overlay_height))

    # Calculate actual overlay region (in case of clipping)
    end_x = min(x + overlay_width, frame_width)
    end_y = min(y + overlay_height, frame_height)
    actual_width = end_x - x
    actual_height = end_y - y

    # Extract regions
    frame_region = frame_16bit[y:end_y, x:end_x]  # Shape: (H, W, 3)
    overlay_region = overlay_16bit[:actual_height, :actual_width]  # Shape: (H, W, 4)

    # Split overlay into RGB and alpha
    overlay_rgb = overlay_region[:, :, :3]  # RGB channels
    overlay_alpha = overlay_region[:, :, 3:4]  # Alpha channel, keep as (H, W, 1)

    # Normalize alpha to 0-1 range (16-bit: 0-65535 -> 0-1)
    alpha_norm = overlay_alpha.astype(np.float32) / 65535.0

    # Alpha blending: result = background * (1 - alpha) + foreground * alpha
    frame_region_float = frame_region.astype(np.float32)
    overlay_rgb_float = overlay_rgb.astype(np.float32)

    blended = (frame_region_float * (1 - alpha_norm) +
               overlay_rgb_float * alpha_norm)

    # Convert back to 16-bit
    frame_16bit[y:end_y, x:end_x] = blended.astype(np.uint16)

    return frame_16bit
