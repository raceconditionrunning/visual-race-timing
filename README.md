# Visual Race Timing

Time runners' laps based on a video. Originally built for Race Condition
Running's [Drumheller Marathon 2024](https://raceconditionrunning.com/drumheller-marathon-24) event. Code based
substantially on Ultralytics and yolo_tracking codebases, from which we inherit the AGPL License.

* Annotation GUI for marking runners crossing the line
* Runner detection to reduce the number of frames that need to be examined
* Runner re-identification to help you keep track of who's who
* Lap time collation and results output

## Usage

1. Record a video of the lap line. The video needs to have timecode metadata
2. Run a detector over all frames (`detect.py`)
3. Create an event configuration file (`config.yaml`)
4. Annotate crossings (`annotate.py`)
5. Collate the results (`collate.py`)

Each of these scripts work on a _project_ directory, which directions with configuration, detection, and annotation
files. The project directory is specified as the first argument to each script.

### Recording a video

Aim for an elevated, static perspective of the finish line. 30fps/1080p is good, and higher resolutions may help some of
the models.

The video needs to have start timecode metadata set to a global reference. Frame numbers in this timecode are used to
key detections and annotations.

### Running the detector

Use the `detect.py` script to run the detector over all frames of the video. The script dumps out a file per frame with
YOLO-format detections (class, x, y, w, h, confidence), normalized as a fraction of frame dimensions. Use the largest
version of YOLO that you can run in a reasonable amount of time. The `--conf` flag sets the confidence threshold for
detections and needs to be manually tuned for your video.

* You should specify a crop (`--crop`) region around the finish line, as there's no need to detect runners outside of
  this area.

### Creating an event configuration file

Create a `config.yaml` file in the project directory. This file contains the event configuration, including the
participants, the finish line, and the start times. The `config.yaml` file is a dictionary with the following keys:

| Key          | Description                                                                                                                                                                     |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| participants | Dictionary with bib number key to runner name value                                                                                                                             |
| finish_line  | Tuple of two points for finish line pixels [[x0, y0], [x1, y1]]. Get these points with `scripts/get_point_in_video.py`                                                          |
| starts       | Dict with named starts (e.g. 'marathon', 'half'), where values are dicts with keys `timecode` and `bib_range` (inclusive tuple of minimum and maximum bib with this start time) |

### Annotating crossings

Annotations are human-verified markings of where in a frame a runner was (bounding box) and whether they crossed the
line in the frame. The `edit.py` script is a video player GUI which allows you to create and edit these annotations. The
workflow is to scrub through detections, promoting them to annnotations as needed, and then to mark the annotations as
crossing or not crossing the line. Detections do not have an identifying number, so the tool will request input for the
runner ID. As you mark more annotations, the tool will attempt to re-identify runners based on their visual features.
You'll need to obtain the weights for the re-identification model from
the [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html). We used `osnet_ain_x1_0`
trained on MSMT17 for cross-domain re-identification.

While the GUI window has focus, you can use the following keyboard commands:

| Input                                | Action                                                       |
|--------------------------------------|--------------------------------------------------------------|
| `left click` on detection            | Promote detection to annotation                              |
| `left click` on annotation           | Begin edit (CLI prompt for command)                          |
| `left click` + `shift` on detection  | Promote detection to annotation crossing                     |
| `left click` + `shift` on annotation | Mark annotation as crossing                                  |
| `left click` + `ctrl` on annotation  | Mark annotation as crossing and reassign runner ID (via CLI) |                                                     
| `right click` -> drag -> release     | Create annotation with start and end corners                 |
| `[` or `]`                           | Prev/next annotation                                         |
| `{` or `}`                           | Prev/next crossing annotation                                |
| `9` or `0`                           | Prev/next detection overlapping finish line                  |
| `(` or `)`                           | Prev/next frame                                              |
| `s`                                  | Seek to timecode (via `HH:MM:SS` CLI input)                  |
| \`                                   | Create note annotation (via CLI)                             |

Notes can be arbitrary text and are saved in a separate file. They are useful for marking things that _didn't_ happen,
like a runner not crossing the line. They will be printed out in the collated lap time tables to help you check your
work.

### Collating the results

The `collate.py` script prints lap time tables and a results.json file.

The results object has a `config` key which contains the race configuration, and a `results` key, which contains a list
of dictionaries with the following keys:

* `id`: Runner bib
* `name`: Runner name
* `lap_times`: Floating point number of seconds for each lap (defined as crossing the line to crossing the line)

## Misc scripts

* `scripts/get_point_in_video.py`: Get a pixel coordinate in a video by clicking on it. Useful for making the finish
  line config.

## Future work

* Integrate partial GPS track data
* Use pose estimation to automatically mark crossings
* Fix up tracking (partially integrated in `edit.py`, and `track.py` is a WIP)
* Support folders of JPGs to handle intervalometer/timelapse recordings