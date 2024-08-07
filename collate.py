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

from visual_race_timing.annotations import load_annotations, load_notes
from visual_race_timing.video_player import VideoPlayer
import iteround


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


def timecode_serialiser(obj):
    if isinstance(obj, Timecode):
        return str(obj)
    return obj


def main(args):
    # Load race configuration from yaml
    fps = '30000/1001'
    race_config = args.project / 'config.yaml'
    with open(race_config, "r") as f:
        race_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    starts = race_config['starts']
    for start_name, details in starts.items():
        starts[start_name]['time'] = Timecode(fps, details['time'])

    annotations = load_annotations(args.project / "annotations")
    notes = load_notes(args.project / "notes.tsv")
    frame_nums = sorted(list(annotations.keys()))
    participant_lap_times = defaultdict(list)
    crops = defaultdict(list)
    for frame_num in frame_nums:
        boxes = annotations[frame_num]['boxes']
        crossings = annotations[frame_num]['crossings']
        for id, is_crossing in zip(boxes[:, 4], crossings):
            id = int(id)
            if is_crossing:
                participant_lap_times[id].append(Timecode(fps, frames=frame_num))
                crops[frame_num].append(
                    (id, (len(participant_lap_times[id])), boxes[boxes[:, 4].astype(int) == id][0][:4]))

    collated_notes = defaultdict(dict)
    for frame_num, notes_by_runner in notes.items():
        for runner_id, note in notes_by_runner.items():
            collated_notes[runner_id][frame_num] = note

    # Insert start time at start of lap time list depending on bib
    for id in participant_lap_times.keys():
        for start_name, details in starts.items():
            bib_range = details['bibs']
            if bib_range[0] <= id < bib_range[1]:
                participant_lap_times[id] = [details['time']] + participant_lap_times[id]
                break

    if args.sources:
        player = VideoPlayer(args.sources)
        player.paused = True
        frames_needing_crop = sorted(list(crops.keys()))
        crop_out_path = args.project / 'crops'
        crop_out_path.mkdir(exist_ok=True)
        for frame_num in tqdm(frames_needing_crop):
            player.seek_to_frame(frame_num)
            frame = player._advance_frame()
            for runner_id, crop_idx, crop in crops[frame_num]:
                crop = [int(c) for c in crop]
                to_save = frame[crop[1]:crop[3], crop[0]:crop[2]]
                cv2.imwrite(crop_out_path / f"{runner_id}_{crop_idx}_{frame_num}.png", to_save)

    for id, lap_start_times in participant_lap_times.items():
        runner_notes = collated_notes[format(id, '02x')]
        note_frames = np.array(sorted(list(runner_notes.keys())))
        id_str = format(id, '02x')
        if id_str.upper() not in race_config['participants']:
            continue
        # list has two extra entries: start time, and 1st lap entrance (which is not a full lap)
        print(f"{race_config['participants'][id_str.upper()]} ({id_str}): {len(lap_start_times) - 2} laps")
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
        # Interleave notes with lap times
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
    for id, lap_times in participant_lap_times.items():
        all_laps += len(lap_times)
    print(f"Total laps: {all_laps}")

    output = {"config": race_config, "results": []}

    for id, lap_times in participant_lap_times.items():
        id_str = format(id, '02x')
        if id_str.upper() not in race_config['participants']:
            continue
        lap_diffs = [
            lap_times[i].to_realtime(as_float=True) - lap_times[i - 1].to_realtime(as_float=True) if i > 0 else 0 for i
            in range(1, len(lap_times))]
        output["results"].append({
            'id': id,
            'name': race_config['participants'][format(id, '02x').upper()],
            "lap_times": lap_diffs,
        })
    del output["config"]["participants"]

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

    with open(args.project / "results.json", "w") as f:
        json.dump(camelize(round_floats(output)), f, default=timecode_serialiser, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=pathlib.Path)
    parser.add_argument('--sources', type=pathlib.Path, nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
