#!/usr/bin/env python
import argparse
import pathlib
from collections import defaultdict
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import ultralytics.utils.ops
import yaml
from timecode import Timecode
from ultralytics.engine.results import Boxes
from ultralytics.utils.metrics import bbox_ioa

from visual_race_timing.annotations import SQLiteAnnotationStore
from visual_race_timing.drawing import draw_annotation, draw_scaled_labels
from visual_race_timing.proxy import find_proxy
from visual_race_timing.reid_bank import DEFAULT_REID_WEIGHTS, ReIDBank, available_reid_models, build_extractor
from visual_race_timing.race_config import assign_start_by_runner, build_start_realtime, get_finish_line
from visual_race_timing.participant_console import ParticipantConsole
from visual_race_timing.timing_prior import TimingPrior, fuse_ranking
from visual_race_timing.tracker import RaceTracker

from visual_race_timing.crossing import guess_crossing_frame, find_nearest_crossing, scan_to_near_line_detection
from visual_race_timing.geometry import line_segment_to_box_distance
from visual_race_timing.prompts import ask_for_id
from visual_race_timing.media_player import VideoPlayer, PhotoPlayer, BufferedVideoPlayer

from visual_race_timing.logging import get_logger

logger = get_logger(__name__)


def _shift_to_local(boxes, kpts, offset_x, offset_y):
    """Shift xyxy boxes (cols 0:4) and keypoints (cols 0,1) from full-frame
    coordinates into a frame offset by (offset_x, offset_y)."""
    if offset_x or offset_y:
        if boxes.size > 0:
            boxes[:, [0, 2]] -= offset_x
            boxes[:, [1, 3]] -= offset_y
        if kpts is not None:
            kpts[:, :, 0] -= offset_x
            kpts[:, :, 1] -= offset_y
    return boxes, kpts


def run(args):
    is_photo_source = len(args.source) == 1 and args.source[0].is_dir()

    sources = args.source
    proxy_kwargs = {}
    if args.crop and not is_photo_source:
        proxies = [find_proxy(s, args.crop, 1.0) for s in args.source]
        if all(proxies):
            sources = proxies
            proxy_kwargs = {'original_sources': args.source}
            logger.info(f"Using proxy media for crop {args.crop}.")
        else:
            logger.info("No matching proxy for one or more sources, decoding originals.")

    if is_photo_source:
        player = PhotoPlayer(args.source[0], args.paused, crop=args.crop)
    else:
        player = BufferedVideoPlayer(sources, args.paused, crop=args.crop, **proxy_kwargs)

    # Load all annotations
    store = SQLiteAnnotationStore(args.project / 'annotations.db')

    # crossing_map = build_crossing_map(annotations)
    tracker_config = args.project / 'tracker_config.yaml'
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']
    cfg.match_thresh = .8
    # boxmot's BotSort expects `cmc_method`; the project config predates boxmot and uses `gmc_method`.
    tracker_kwargs = vars(cfg).copy()
    tracker_kwargs['cmc_method'] = tracker_kwargs.pop('gmc_method')

    # ReID feature bank for the manual click->ID path
    reid_extractor = build_extractor(args.reid_model, device=args.device, half=False)
    bank = ReIDBank.load(args.project / 'reid_bank.npz', reid_extractor)

    # Load race configuration from yaml
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Interactive tracker for the "(" / ")" forward/backward tracking block.
    #  shares the ReID backend with `bank` so the
    # tracker and the manual click->ID bank live in one embedding space.
    participants_by_bib = {format(int(rid), '02x').lower(): name
                           for rid, name in race_config['participants'].items()}
    tracker = RaceTracker(reid_model=reid_extractor.model,
                          participants=participants_by_bib,
                          policy="prompt", use_cmc=False,
                          **tracker_kwargs)

    if args.seek_frame:
        args.seek_time = str(Timecode(player.get_last_timecode().framerate, frames=args.seek_timecode_frame))
    if args.seek_time:
        player.seek_time(args.seek_time)

    # Per-frame cache of the ReID top-1 guess drawn on each detection box, so
    # re-renders of the same frame (e.g. after a click) don't re-run the model.
    _det_guess_cache = {"frame": None, "labels": None}

    def invalidate_det_guesses():
        _det_guess_cache["frame"] = None
        _det_guess_cache["labels"] = None

    def get_finish_line_local(frame_num):
        fps = player.get_last_timecode().framerate
        orig_h, orig_w = player.loader.get_image_dims()
        offset_x, offset_y = player.frame_offset()
        p0, p1 = get_finish_line(race_config, Timecode(fps, frames=frame_num),
                                 frame_width=orig_w, frame_height=orig_h)
        return (p0[0] - offset_x, p0[1] - offset_y), (p1[0] - offset_x, p1[1] - offset_y)

    def detection_guess_labels(frame_num, local_boxes):
        """Label each detection box with the ReID bank's top guess
        ('bib name  0.31'), falling back to the detection confidence when the
        bank has no opinion. Detections on/near the finish line get the same
        timing-prior fusion a crossing confirm applies, so the shown id matches
        what a (shift-)ctrl-click would commit. Computed once per frame (cached)
        and only when the model output is shown, i.e. while paused."""
        if local_boxes.size == 0:
            return None
        if _det_guess_cache["frame"] == frame_num:
            return _det_guess_cache["labels"]

        rankings = bank.guess_batch(player._last_frame_img, local_boxes[:, :4])

        # Which detections are on the line -> fuse the timing prior for those
        fps = player.get_last_timecode().framerate
        line = get_finish_line_local(frame_num)
        on_line = line_segment_to_box_distance(line[0], line[1], local_boxes[:, :4]) < 10
        t = Timecode(fps, frames=frame_num).to_realtime(as_float=True)

        labels = []
        for i in range(len(local_boxes)):
            ids, dists = rankings[i]
            if not ids:
                labels.append(f"{local_boxes[i, 5]:.2f}")
                continue
            if on_line[i]:
                ids, dists, _ = fuse_ranking(get_timing_prior(), ids, dists, t)
            gid, gdist = ids[0], dists[0]
            bib = format(gid, '02x')
            name = race_config['participants'].get(gid)
            first = name.split(" ")[0] if name else bib
            labels.append(f"{bib} {first} {gdist:.2f}")
        _det_guess_cache["frame"] = frame_num
        _det_guess_cache["labels"] = labels
        return labels

    def overlay_annotations(frame, frame_num):
        # Denormalize against the loader's original dims, not frame.shape,
        # then shift into frame's own (possibly offset) pixel space.
        original_dims = player.loader.get_image_dims()
        offset_x, offset_y = player.frame_offset()

        line = get_finish_line_local(frame_num)
        p0 = (int(round(line[0][0])), int(round(line[0][1])))
        p1 = (int(round(line[1][0])), int(round(line[1][1])))
        frame = cv2.line(frame, p0, p1, (220, 150, 60), 1, cv2.LINE_AA)

        frame_notes = store.get_notes(frame_num)
        if player.show_boxes:
            frame_annotation_boxes, frame_annotation_keypoints, frame_annotation_crossings, _ = store.get_frame_annotation(
                frame_num, original_dims, "human")
            frame_detection_boxes, frame_detection_keypoints, frame_detection_crossings, _ = store.get_frame_annotation(
                frame_num, original_dims, args.detection_model)
            frame_annotation_boxes, frame_annotation_keypoints = _shift_to_local(
                frame_annotation_boxes, frame_annotation_keypoints, offset_x, offset_y)
            frame_detection_boxes, frame_detection_keypoints = _shift_to_local(
                frame_detection_boxes, frame_detection_keypoints, offset_x, offset_y)

            # Drop only detections whose box *matches* a crossing box
            if frame_detection_boxes.size > 0 and frame_annotation_boxes.size > 0:
                crossing_mask = np.asarray(frame_annotation_crossings, dtype=bool)
                if crossing_mask.any():
                    det = frame_detection_boxes[:, :4]                       # (D, 4)
                    cross = frame_annotation_boxes[crossing_mask, :4]        # (C, 4)
                    corner_diff = np.abs(det[:, None, :] - cross[None, :, :]).max(axis=2)  # (D, C)
                    keep = ~(corner_diff <= 1.0).any(axis=1)
                    frame_detection_boxes = frame_detection_boxes[keep]
                    frame_detection_crossings = np.asarray(frame_detection_crossings)[keep]
                    if frame_detection_keypoints is not None:
                        frame_detection_keypoints = frame_detection_keypoints[keep]

            if frame_detection_boxes.size > 0:
                # While paused, label each detection with the ReID guess so it can be
                # confirmed by (ctrl-)click without reading the keyboard prompt.
                det_labels = detection_guess_labels(frame_num, frame_detection_boxes) if player.paused else None
                # Draw the green boxes/keypoints via the shared annotator, but render
                # the guess labels ourselves (below) so their font scales with the box.
                frame = draw_annotation(img=frame, boxes=frame_detection_boxes, keypoints=frame_detection_keypoints,
                                        crossings=frame_detection_crossings, labels=None,
                                        conf=None if det_labels else frame_detection_boxes[:, 4],
                                        kpt_radius=2 * frame.shape[0] // 1080,
                                        colors=[(0, 255, 0)] * len(frame_detection_boxes),
                                        line_width=1 * frame.shape[0] // 1080)
                if det_labels:
                    bg = [(0, 200, 0)] * len(frame_detection_boxes)
                    fg = [(0, 0, 0)] * len(frame_detection_boxes)
                    frame = draw_scaled_labels(frame, frame_detection_boxes, det_labels, bg, fg)
            if frame_annotation_boxes.size > 0:
                ids = frame_annotation_boxes[:, 4].astype(int)
                bibs = [format(runner_id, '02x') for runner_id in ids]
                names = [race_config['participants'].get(runner_id, None) for runner_id in ids]
                names = [name.split(" ")[0] if name else bib for bib, name in zip(bibs, names)]
                labels = [f"{bib}{' ' + name if name else ''}" for bib, name in zip(bibs, names)]
                # Draw the boxes/keypoints via the shared annotator (labels=None), then
                # render the labels ourselves so they scale with the bbox. Mirror the
                # annotator's colors: crossings get a black box + white text, others
                # the default red box + black text.
                crossing_flags = list(frame_annotation_crossings) if frame_annotation_crossings is not None else []
                crossing_flags += [False] * (len(frame_annotation_boxes) - len(crossing_flags))
                bg = [(0, 0, 0) if c else (0, 0, 255) for c in crossing_flags]
                fg = [(255, 255, 255) if c else (0, 0, 0) for c in crossing_flags]
                frame = draw_annotation(img=frame, boxes=frame_annotation_boxes, keypoints=frame_annotation_keypoints,
                                        crossings=frame_annotation_crossings, labels=None,
                                        kpt_radius=2 * frame.shape[0] // 1080, line_width=1 * frame.shape[0] // 1080)
                frame = draw_scaled_labels(frame, frame_annotation_boxes, labels, bg, fg)
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
            # new_box is in full-frame coords; shift to match _last_frame_img.
            local_box, _ = _shift_to_local(new_box.copy(), None, *player.frame_offset())
            bank.update(player._last_frame_img, local_box, runner_id)
            # Bank changed -> the drawn detection guesses are now stale.
            invalidate_det_guesses()
        return True

    # --- Timing prior -------------------------------------------------------
    # Fuse each runner's lap history ("are they due to cross now?") into the
    # crossing-ID guess. Built lazily from the store's human crossings and
    # invalidated whenever a crossing is added/toggled so it stays current with
    # the session. Only applied to on-the-line crossing guesses (see below).
    _timing = {"prior": None, "crossing_frames": None}

    def invalidate_timing_prior():
        _timing["prior"] = None
        _timing["crossing_frames"] = None
        # On-line detection labels are fused with the prior, so they're stale too.
        invalidate_det_guesses()

    def refresh_marks():
        """Repaint the transport density strip from current crossing annotations."""
        player.transport.set_marks(store.get_crossing_frames(source="human"))
        if player.console.focused_rid is not None:
            # A runner's ticks are showing -- keep them current too.
            frames = get_crossing_frames_by_runner().get(int(player.console.focused_rid), [])
            player.transport.set_focus_marks(frames)

    def get_timing_prior():
        if _timing["prior"] is None:
            fps = player.get_last_timecode().framerate
            ann = store.load_all_annotations(source="human", crossing=True)
            crossings_by_runner = defaultdict(list)
            for frame_num, data in ann.items():
                frame_boxes = data["boxes"]
                if frame_boxes is None or frame_boxes.size == 0:
                    continue
                secs = Timecode(fps, frames=frame_num).to_realtime(as_float=True)
                for rid in frame_boxes[:, 4].astype(int):
                    if int(rid) in race_config["participants"]:
                        crossings_by_runner[int(rid)].append(secs)
            start_map = build_start_realtime(race_config, fps)
            _timing["prior"] = TimingPrior.build(crossings_by_runner, start_map)
        return _timing["prior"]

    def get_crossing_frames_by_runner():
        """Per-runner confirmed-crossing frame numbers (ascending), cached and
        invalidated alongside the timing prior. Drives the participant console's
        per-runner navigation."""
        if _timing["crossing_frames"] is None:
            ann = store.load_all_annotations(source="human", crossing=True)
            by_runner = defaultdict(list)
            for frame_num, data in ann.items():
                frame_boxes = data["boxes"]
                if frame_boxes is None or frame_boxes.size == 0:
                    continue
                for rid in frame_boxes[:, 4].astype(int):
                    if int(rid) in race_config["participants"]:
                        by_runner[int(rid)].append(frame_num)
            for rid in by_runner:
                by_runner[rid].sort()
            _timing["crossing_frames"] = by_runner
        return _timing["crossing_frames"]

    def console_rows(frame_num):
        """One (rid, confirmed_count, state) row per participant, in config order.
        A runner is 'predict' (amber) once the playhead is at/past their last
        confirmed crossing -- i.e. no confirmed crossing lies ahead."""
        by_runner = get_crossing_frames_by_runner()
        rows = []
        for rid in race_config["participants"]:
            frames = by_runner.get(int(rid), [])
            state = 'neutral' if any(f > frame_num for f in frames) else 'predict'
            rows.append((int(rid), len(frames), state))
        return rows

    def console_seek(action):
        """Dispatch a participant-console arrow. Forward flips to the runner's
        next confirmed crossing; past their last one it projects the next
        (unconfirmed) crossing from the timing prior's expected lap. Backward
        flips to the previous confirmed crossing and clamps at the first."""
        _, rid, direction = action
        frame_num = player.get_last_timecode().frames
        fps = float(player.get_last_timecode().framerate)
        frames = get_crossing_frames_by_runner().get(int(rid), [])

        if direction < 0:
            behind = [f for f in frames if f < frame_num]
            if not behind:
                logger.info(f"{format(rid, '02x')}: at first confirmed crossing.")
                return
            target = behind[-1]
        else:
            ahead = [f for f in frames if f > frame_num]
            if ahead:
                target = ahead[0]
            else:
                # Past the last confirmed crossing: project from expected lap.
                if frames:
                    base = frames[-1]
                else:
                    start_tc = assign_start_by_runner(race_config, fps).get(int(rid))
                    if start_tc is None:
                        logger.info(f"{format(rid, '02x')}: no crossings or wave start to project from.")
                        return
                    base = start_tc.frames
                mu = get_timing_prior().expected_lap(int(rid))
                mu_frames = max(1, round(mu * fps))
                k = 1
                target = base + k * mu_frames
                while target <= frame_num and k < 100:
                    k += 1
                    target = base + k * mu_frames
                logger.info(f"{format(rid, '02x')}: past last confirmed crossing; "
                            f"projecting next at frame {target} (~{mu:.0f}s lap).")

        if player.seek_timecode_frame(target):
            player._advance_frame()
            player.render()
        else:
            logger.error(f"Failed to seek to frame {target}.")

    def console_focus(rid):
        """Toggle per-runner focus from a label click: highlights the row and
        swaps the transport density strip for tick marks at just that runner's
        confirmed crossings. Clicking the same runner again clears focus back
        to the aggregate view."""
        rid = int(rid)
        if player.console.focused_rid == rid:
            player.console.focused_rid = None
            player.transport.set_focus_marks(None)
        else:
            player.console.focused_rid = rid
            player.transport.set_focus_marks(get_crossing_frames_by_runner().get(rid, []))
        player.render()

    def console_dispatch(action):
        """Route a participant-console hit to the seek or focus handler."""
        if action[0] == 'runner_seek':
            console_seek(action)
        elif action[0] == 'focus':
            console_focus(action[1])

    def calculate_reid_distances(box, exclude: List[int] = [], timecode=None, crossing=False):
        new_box = np.atleast_2d(box)
        # box is in full-frame coords; shift to match _last_frame_img.
        local_box, _ = _shift_to_local(new_box.copy(), None, *player.frame_offset())
        candidate_participants, emb_dists = bank.guess(player._last_frame_img, local_box)

        # Drop excluded ids, then return (distances, ids) ranked closest-first.
        paired = [(d, i) for d, i in zip(emb_dists, candidate_participants) if i not in exclude]
        paired.sort(key=lambda x: x[0])
        dists = [d for d, _ in paired]
        ids = [i for _, i in paired]
        # Fuse the timing prior for line crossings only (it models "due to cross
        # now", which is meaningless for a box away from the finish line).
        if crossing and timecode is not None and ids:
            t = timecode.to_realtime(as_float=True)
            ids, dists, _ = fuse_ranking(get_timing_prior(), ids, dists, t)
        return dists, ids

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
            emb_dists, candidate_participants = calculate_reid_distances(
                new_box, timecode=timecode, crossing=crossing)
            if force and emb_dists:
                # Confirm the top candidate outright.
                annotation_id = f"{candidate_participants[0]:02x}"
            else:
                annotation_id = query_for_reid(emb_dists, candidate_participants)
            if annotation_id is None:
                return False
            else:
                update_tracker(new_box, int(annotation_id, 16))
        new_box[:, 4] = int(annotation_id, 16)
        # new_box is in full-frame coords; normalize against original dims.
        store.update_annotation(timecode.frames, Boxes(new_box, player.loader.get_image_dims()), None,
                                [crossing], "human")
        if crossing:
            # New crossing labeled -> the timing prior's history is now stale.
            invalidate_timing_prior()
            refresh_marks()
        return True

    def key_delegate(frame, frame_num, key, runner_id: str = None):
        if key == ord(','):
            # Jump back 2.5s
            player.seek_timecode_frame(frame_num - round(2.5 * float(player.get_last_timecode().framerate)))
            # This seek is slow; drop mashed jump keys so they don't queue up.
            player.flush_input()
        elif key == ord('.'):
            # Jump forward 2.5s
            player.seek_timecode_frame(frame_num + round(2.5 * float(player.get_last_timecode().framerate)))
            player.flush_input()
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
            # 'r'/'R' pass these to calculate_reid_distances, which expects full-frame coords.
            annotation = store.get_frame_annotation(frame_num, player.loader.get_image_dims(), source="human")
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
                # Crossing history changed -> refresh the timing prior lazily.
                invalidate_timing_prior()
                refresh_marks()
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
        elif key == ord('[') or key == ord(']'):
            if key == ord('['):
                next_frame = store.scan_to_annotation(frame_num, previous=True, source="human")
            else:
                next_frame = store.scan_to_annotation(frame_num, previous=False, source="human")

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
            # Seek to the nearest detection overlapping the finish line
            fps = player.get_last_timecode().framerate
            dims = player.loader.get_image_dims()
            direction = 1 if key == ord('0') else -1

            next_frame = scan_to_near_line_detection(store, race_config, fps, dims, frame_num,
                                                     args.detection_model, direction=direction)
            if next_frame is not None:
                logger.info(
                    f"Nearest on-line detection at frame {next_frame} "
                    f"({Timecode(player.get_last_timecode().framerate, frames=next_frame)}); seeking.")
                success = player.seek_timecode_frame(next_frame)
                if not success:
                    logger.error(f"Failed to seek to frame {next_frame}.")
                    return None
                player._advance_frame()
                player.render()
                return None
            else:
                logger.info("No further detections near the line.")
                return None
        elif key == ord('{') or key == ord('}'):
            # Smart seek: nearest inferred crossing (any runner) from the playhead.
            fps = player.get_last_timecode().framerate
            dims = player.loader.get_image_dims()
            direction = 1 if key == ord('}') else -1

            scan = find_nearest_crossing(store, race_config, fps, dims, frame_num, args.detection_model,
                                         direction=direction)

            if scan.best is not None:
                ties = [c for c in scan.candidates
                       if c is not scan.best and abs(c.crossing_frame - scan.best.crossing_frame) <= 2]
                tie_note = f", {len(ties)} tied within 2 frames" if ties else ""
                logger.info(
                    f"Nearest crossing at frame {scan.best.crossing_frame} "
                    f"(confidence {scan.best.result.confidence:.2f}, {len(scan.candidates)} candidate(s) found{tie_note}); "
                    f"seeking. Shift-click the runner there to confirm the crossing.")
                success = player.seek_timecode_frame(scan.best.crossing_frame)
                if not success:
                    logger.error(f"Failed to seek to frame {scan.best.crossing_frame}.")
                    return None
                player._advance_frame()
                player.render()
                return None
            elif scan.reason == "no_confirmed_crossing":
                logger.info(
                    f"No confirmed crossing found ({scan.seeds_tried} seed(s) tried); "
                    f"seeking to the nearest on-line detection instead.")
                success = player.seek_timecode_frame(scan.first_seed_frame)
                if not success:
                    logger.error(f"Failed to seek to frame {scan.first_seed_frame}.")
                    return None
                player._advance_frame()
                player.render()
                return None
            else:
                logger.info(f"No further detections near the line (scanned {scan.frames_scanned} frames).")
                return None
        elif key == ord('(') or key == ord(')'):
            # Track forward/backward
            fps = player.get_last_timecode().framerate
            frame_h, frame_w = frame.shape[:2]
            # tracker.reset()
            start_frame = player.get_last_timecode().frames
            i = 0
            while True:
                line_seg_pts = get_finish_line(race_config, Timecode(fps, frames=start_frame + i),
                                               frame_width=frame_w, frame_height=frame_h)
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
                    # boxmot dets are (N,6)=(x1,y1,x2,y2,conf,cls); ids ride a
                    # side channel. combined cols are [x1,y1,x2,y2,id,conf,cls].
                    sel = combined[on_line_mask]
                    dets = sel[:, [0, 1, 2, 3, 5, 6]].astype(np.float32)
                    known_ids = sel[:, 4]
                    res = tracker.update(dets, frame, known_ids=known_ids)
                    # TrackResults (K,8) -> [x1,y1,x2,y2,id,conf,cls]
                    out = np.asarray(res[:, :7])

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
                    tracker.update(np.zeros((0, 6), dtype=np.float32), frame)
                i += 1 if key == ord(')') else -1
        return None

    def guess_crossing_and_seek(frame_num, clicked_box):
        fps = player.get_last_timecode().framerate
        dims = player.loader.get_image_dims()
        result = guess_crossing_frame(
            store, race_config, fps, dims, frame_num, clicked_box[:4], args.detection_model, direction=1,
        )
        if result.frame is None:
            logger.info(f"No confident crossing guess for the clicked runner ({result.reason}).")
            return
        logger.info(f"Guessed crossing at frame {result.frame} (confidence {result.confidence:.2f}); seeking. "
                   f"Shift-click the runner there to confirm the crossing.")
        success = player.seek_timecode_frame(result.frame)
        if not success:
            logger.error(f"Failed to seek to guessed crossing frame {result.frame}.")
            return
        player._advance_frame()
        player.render()

    def click_delegate(frame, frame_num, click_pt, flags):
        # click_pt is in full-frame coords; denormalize boxes to match.
        original_w, original_h = player.loader.get_image_dims()[::-1]
        # First, check if user clicked on any existing boxes
        frame_annotations = store.get_frame_annotation(frame_num, source="human")
        boxes = frame_annotations[0]
        boxes[:, :4] = ultralytics.utils.ops.xywhn2xyxy(boxes[:, :4], original_w, original_h)
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
        detected_boxes[:, :4] = ultralytics.utils.ops.xywhn2xyxy(detected_boxes[:, :4], original_w, original_h)

        player.render()
        # Check if the click point is inside any of the boxes
        inside = [box[0] < click_pt[0] < box[2] and box[1] < click_pt[1] < box[3] for box in detected_boxes]
        if len(inside) == 0 or not any(inside):
            return
        idxs = np.where(inside)[0]
        if len(idxs) > 1:
            # Disambiguate rather than refuse: pick the tightest box,
            # tie-broken by confidence, instead of forcing an impossible precise
            # click.
            sel = detected_boxes[idxs]
            areas = np.rint((sel[:, 2] - sel[:, 0]) * (sel[:, 3] - sel[:, 1]))
            order = sorted(range(len(idxs)), key=lambda k: (areas[k], -sel[k, 5]))
            chosen = idxs[order[0]]
            logger.info(f"{len(idxs)} overlapping detections at click; using tightest/highest-conf box.")
        else:
            chosen = idxs[0]
        clicked_box = detected_boxes[chosen]
        if flags & cv2.EVENT_FLAG_ALTKEY:
            guess_crossing_and_seek(frame_num, clicked_box)
            return
        # Shift-click confirms a crossing: it marks the box a crossing AND accepts
        # the ReID guess drawn on the box (force), so the common crossing-confirm
        # is keyboard-free.
        shift = (flags & cv2.EVENT_FLAG_SHIFTKEY) > 0
        player.annotation_updated(None, [clicked_box[0:2], clicked_box[2:4]], player.get_last_timecode(),
                                  crossing=shift, force=shift)
        return

    player.click_delegate = click_delegate
    player.annotation_updated = annotation_updated
    player.pre_display = overlay_annotations
    player.key_delegate = key_delegate
    player.console = ParticipantConsole(race_config['participants'])
    player.console_rows = console_rows
    player.console_action = console_dispatch
    player.transport.add_seek_steppers([
        {'type': 'stepper', 'label': 'Detection', 'prev': ord('9'), 'next': ord('0')},
        {'type': 'stepper', 'label': 'Annot', 'prev': ord('['), 'next': ord(']')},
        {'type': 'stepper', 'label': 'Smart', 'prev': ord('{'), 'next': ord('}')},
    ])
    player.transport.add_toggle('Boxes', ord('b'), lambda: player.show_boxes)
    player.transport.add_toggle('Console', ord('p'), lambda: player.console.visible)
    # Density strip: click to seek, filled by crossing annotations.
    player.transport.set_timeline(*player.loader.get_frame_range())
    refresh_marks()
    try:
        player.play()
    finally:
        logger.info("Saving reid bank...")
        bank.save(args.project / 'reid_bank.npz')


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('project', type=pathlib.Path, default='data/exp')
    parser.add_argument('--source', type=pathlib.Path, nargs='+', required=True,
                        help='file paths')
    parser.add_argument('--seek-frame', type=int,
                        help='seek frame (timecode index from start) to start tracking')
    parser.add_argument('--seek-time', type=str, default=None, help='seek time to start tracking')
    parser.add_argument('--paused', action='store_true', help='start paused')
    parser.add_argument('--reid-model', type=pathlib.Path, default=None,
                        help='reid model path, or a name boxmot auto-downloads, e.g. one of: '
                             + ', '.join(available_reid_models())
                             + f'. Defaults to the repo weights ({DEFAULT_REID_WEIGHTS.name}).')
    parser.add_argument('--device', default='cuda',
                        help='device to run on, e.g. cuda, 0, 0,1,2,3, cpu, or mps (Apple Silicon)')
    parser.add_argument('--detection-model', type=str, default='detection',
                        help='Only display detections from this source')
    parser.add_argument('--crop', type=int, nargs=4, default=None, help="display area w h x y")

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
