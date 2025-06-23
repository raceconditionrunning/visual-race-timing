#!/usr/bin/env python
import argparse
import json
import pathlib
from collections import defaultdict

import cv2
import numpy as np
import tabulate
import yaml
from timecode import Timecode
from tqdm import tqdm

from visual_race_timing.annotations import SQLiteAnnotationStore
import iteround
import joblib

from visual_race_timing.drawing import draw_annotation
from visual_race_timing.loader import ImageLoader, VideoLoader


def round_floats(o):
    # The maximal error of rounding a float to 3 decimal places is 0.0005
    # For 220 laps, the maximal error is 0.11 seconds, which is acceptable.
    # We'll use a sum-preserving round anyway.
    if isinstance(o, float): return round(o, 3)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        if all(isinstance(i, float) for i in o):
            return iteround.saferound(o, 3)
        return [round_floats(i) for i in o]
    return o


def timecode_serializer(obj):
    if isinstance(obj, Timecode):
        return str(obj)
    return obj


def snake_to_camel(s):
    a = s.split('_')
    a[0] = a[0].lower()
    if len(a) > 1:
        a[1:] = [u.title() for u in a[1:]]
    return ''.join(a)


def camelize(obj):
    if isinstance(obj, dict):
        return {snake_to_camel(k): camelize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [camelize(x) for x in obj]
    return obj


def main(args):
    # Load race configuration from yaml
    if len(args.sources) == 1 and args.sources[0].is_dir():
        loader = ImageLoader(args.sources[0])
    else:
        loader = VideoLoader(args.sources)
    fps = loader.get_current_time().framerate
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    starts = race_config['starts']
    for start_name, details in starts.items():
        starts[start_name]['time'] = Timecode(fps, details['time'])

    store = SQLiteAnnotationStore(args.project / "annotations.db")
    annotations = store.load_all_annotations(loader.get_image_dims(), "human", crossing=True)
    notes = store.load_notes()
    frame_nums = sorted(list(annotations.keys()))
    participant_lap_times = defaultdict(list)
    crops = defaultdict(list)
    for frame_num in frame_nums:
        boxes = annotations[frame_num]['boxes']
        crossings = annotations[frame_num]['crossings']
        for runner_id, is_crossing in zip(boxes[:, 4], crossings):
            runner_id = int(runner_id)
            if is_crossing:
                participant_lap_times[runner_id].append(Timecode(fps, frames=frame_num))
                crops[frame_num].append(
                    (runner_id, (len(participant_lap_times[runner_id])),
                     boxes[boxes[:, 4].astype(int) == runner_id][0]))

    collated_notes = defaultdict(dict)
    for frame_num, notes_by_runner in notes.items():
        for runner_id, note in notes_by_runner.items():
            collated_notes[runner_id][frame_num] = note

    # Insert start time at start of lap time list depending on bib
    for runner_id in participant_lap_times.keys():
        for start_name, details in starts.items():
            bib_range = details['bibs']
            if bib_range[0] <= runner_id < bib_range[1]:
                participant_lap_times[runner_id] = [details['time']] + participant_lap_times[runner_id]
                break

    tracker = None
    if args.update_tracker:
        from visual_race_timing.tracker import PartiallySupervisedTracker
        from types import SimpleNamespace
        with open(args.project / "tracker_config.yaml", "r") as f:
            tracker_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            tracker_config = SimpleNamespace(**tracker_config)  # easier dict access by dot, instead of ['']
        tracker = PartiallySupervisedTracker(args.reid_model, tracker_config, device=args.device)

    if args.sources:
        frames_needing_crop = sorted(list(crops.keys()))
        crop_out_path = args.project / 'crops'
        crop_out_path.mkdir(exist_ok=True)
        for frame_num in tqdm(frames_needing_crop):
            loader.seek_timecode_frame(frame_num)
            path, frame, metadata = loader.__next__()
            frame = frame[0]
            for runner_id, crop_idx, crop in crops[frame_num]:
                crop = [int(c) for c in crop]
                to_save = frame[crop[1]:crop[3], crop[0]:crop[2]]
                if crop[2] - crop[0] < 24 or crop[3] - crop[1] < 24:
                    print(f"Skipping crop {runner_id} {crop_idx} at frame {frame_num} due to small size")
                    continue
                if tracker:
                    tracker.update_participant_features(frame, crop, runner_id)
                cv2.imwrite(crop_out_path / f"{runner_id:02x}_{crop_idx}_{frame_num}.png", to_save)
        if tracker:
            joblib.dump(tracker, args.project / 'tracker.pkl')

    for runner_id, lap_start_times in participant_lap_times.items():
        runner_notes = collated_notes[format(runner_id, '02x')]
        if runner_id not in race_config['participants']:
            continue
        # list has two extra entries: start time, and 1st lap entrance (which is not a full lap)
        print(f"{race_config['participants'][runner_id]} ({runner_id}): {len(lap_start_times) - 2} laps")
        lap_table = []
        lap_time = [lap_start_times[i].to_realtime(as_float=True) - lap_start_times[i - 1].to_realtime(
            as_float=True) if i > 0 else 0 for i in
                    range(len(lap_start_times))]
        # Timecode seconds may be shorter than wallclock seconds for NTSC frame rates. It's critical to use the realtime conversion.
        # Check the values of the below to see the difference
        # nominal_lapsed = lap_start_times[-1].float - lap_start_times[0].float
        # real_lapsed = lap_start_times[-1].to_realtime(as_float=True) - lap_start_times[0].to_realtime(as_float=True)
        lap_time_change = [(lap_time[i] - lap_time[i - 1]) if i > 1 else 0 for i in range(len(lap_time))]
        for i, lap_start_timecode in enumerate(lap_start_times):
            lap_table.append([i - 1 if i > 1 else "", str(lap_start_timecode), lap_start_timecode.frames, lap_time[i],
                              lap_time_change[i]])
        # Which laps do the notes go before?
        note_frames = sorted(list(runner_notes.keys()))
        note_insertion_indices = np.searchsorted(np.array([entry[2] for entry in lap_table]), note_frames)
        # Check laps which take 70-120% longer than the previous lap. Add a warning label, unless there's a note already there.
        # We skip first (start time) and second (course entrance, not a full lap).
        for i in range(3, len(lap_time)):
            ratio = lap_time[i] / lap_time[i - 1]
            if 2.2 >= ratio >= 1.7 and i not in note_insertion_indices:
                runner_notes[lap_start_times[i].frames] = f"Warning: Lap time is {ratio:.2f}x the previous lap"

        note_frames = sorted(list(runner_notes.keys()))
        for j, note_frame in enumerate(note_frames):
            note = runner_notes[note_frame]
            # We make a fresh copy each iteration because we're mutating
            note_idx = np.searchsorted(np.array([entry[2] for entry in lap_table]), note_frame)
            lap_table.insert(note_idx,
                             [f"Note: {j}", str(Timecode(fps, frames=int(note_frame))), int(note_frame), None, None,
                              note])

        print(tabulate.tabulate(lap_table,
                                headers=['Lap', 'Start Time', 'Start Frame', 'Lap Time', 'Lap Time Change', 'Notes'],
                                floatfmt=".2f"))

    # count all laps
    all_laps = 0
    for runner_id, lap_times in participant_lap_times.items():
        all_laps += len(lap_times)
    print(f"Total laps: {all_laps}")

    output = {"config": race_config, "results": []}

    for runner_id, lap_times in participant_lap_times.items():
        if runner_id not in race_config['participants']:
            continue
        if runner_id in race_config['exclude']:
            continue
        lap_diffs = [
            lap_times[i].to_realtime(as_float=True) - lap_times[i - 1].to_realtime(as_float=True) if i > 0 else 0 for i
            in range(1, len(lap_times))]
        output["results"].append({
            'id': runner_id,
            'name': race_config['participants'][runner_id],
            "lap_times": lap_diffs,
        })
    del output["config"]["participants"]
    del output["config"]["exclude"]

    with open(args.project / "results.json", "w") as f:
        json.dump(camelize(round_floats(output)), f, default=timecode_serializer, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=pathlib.Path)
    parser.add_argument('--sources', type=pathlib.Path, nargs='+')
    parser.add_argument('--update-tracker', action='store_true',)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--reid-model", type=pathlib.Path, default=pathlib.Path("reid_model.pt"))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
