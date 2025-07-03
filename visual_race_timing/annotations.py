import itertools
import sqlite3
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from ultralytics.engine.results import Boxes, Keypoints
from ultralytics.utils.ops import xywhn2xyxy


class SQLiteAnnotationStore:
    # Source type constants
    SOURCE_HUMAN = 'human'
    SOURCE_TRACKING = 'tracking'

    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            conn.executescript('''
                               CREATE TABLE IF NOT EXISTS annotations
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   frame_number
                                   INTEGER,
                                   runner_id
                                   INTEGER, -- -1 when no tracking available
                                   class_id
                                   INTEGER,
                                   x_center
                                   REAL,
                                   y_center
                                   REAL,
                                   width
                                   REAL,
                                   height
                                   REAL,
                                   confidence
                                   REAL,
                                   is_crossing
                                   BOOLEAN,
                                   keypoints
                                   TEXT,
                                   source
                                   TEXT,
                                   created_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   modified_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP
                               );

                               -- Create a partial unique index for tracked detections only
                               CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_tracked
                                   ON annotations(frame_number, runner_id)
                                   WHERE runner_id != -1;

                               CREATE TABLE IF NOT EXISTS notes
                               (
                                   frame_number
                                   INTEGER,
                                   runner_id
                                   INTEGER,
                                   note
                                   TEXT,
                                   PRIMARY
                                   KEY
                               (
                                   frame_number,
                                   runner_id
                               )
                                   );

                               -- Indexes for performance
                               CREATE INDEX IF NOT EXISTS idx_frame ON annotations(frame_number);
                               CREATE INDEX IF NOT EXISTS idx_runner ON annotations(runner_id);
                               CREATE INDEX IF NOT EXISTS idx_crossings ON annotations(is_crossing);
                               CREATE INDEX IF NOT EXISTS idx_frame_crossing ON annotations(frame_number, is_crossing);
                               CREATE INDEX IF NOT EXISTS idx_source ON annotations(source);
                               CREATE INDEX IF NOT EXISTS idx_frame_source ON annotations(frame_number, source);
                               CREATE INDEX IF NOT EXISTS idx_notes_frame ON notes(frame_number);
                               CREATE INDEX IF NOT EXISTS idx_notes_runner ON notes(runner_id);

                               -- Trigger to update modified_at
                               CREATE TRIGGER IF NOT EXISTS update_modified_time 
                AFTER
                               UPDATE ON annotations
                               BEGIN
                               UPDATE annotations
                               SET modified_at = CURRENT_TIMESTAMP
                               WHERE frame_number = NEW.frame_number
                                 AND runner_id = NEW.runner_id;
                               END;
                               ''')

    def _parse_runner_id(self, runner_id):
        """Convert runner_id to int, handling both int and hex string formats"""
        if isinstance(runner_id, str):
            # Check if it's a hex string
            try:
                return int(runner_id, 16)
            except ValueError:
                return int(runner_id)
        return int(runner_id)

    def save_annotation(self, frame_number: int, boxes: Boxes, kpts: Keypoints, crossings, source=None, replace=False,
                        ):
        """Save annotations for a single frame"""
        with sqlite3.connect(self.db_path) as conn:
            if replace is True:
                if source is None or type(source) is not str:
                    raise ValueError("Cannot replace annotations without specifying a source")
                elif source == 'all':
                    # Replace all annotations for this frame
                    conn.execute("DELETE FROM annotations WHERE frame_number = ?", (frame_number,))
                else:
                    conn.execute("DELETE FROM annotations WHERE frame_number = ? AND source = ?",
                                 (frame_number, source))

            for j, box in enumerate(boxes.data):
                if len(box) == 7:
                    # Handle YOLO format: [class_id, x1, y1, x2, y2, confidence, runner_id]
                    x1, y1, x2, y2, runner_id, confidence, class_id = box.tolist()
                else:
                    x_center, y_center, width, height, confidence, class_id = box.tolist()
                    runner_id = -1
                x, y, w, h = boxes.xywhn[j].tolist()
                runner_id = int(runner_id)
                is_crossing = bool(crossings[j]) if j < len(crossings) else False

                # Handle keypoints
                keypoints_json = None
                if kpts is not None and j < len(kpts.data):
                    kpt_data = kpts.data[j].reshape(17, 3).tolist()
                    keypoints_json = json.dumps(kpt_data)

                row_source = source
                if type(source) is not str:
                    row_source = source[j]
                conn.execute('''
                    INSERT OR REPLACE INTO annotations 
                    (frame_number, runner_id, class_id, x_center, y_center, width, height, confidence, is_crossing, keypoints, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (frame_number, runner_id, int(class_id), x, y, w, h, confidence, is_crossing, keypoints_json,
                      row_source))

    def get_frame_annotation(self, frame_number: int, img_shape=None, source=None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, list]:
        """Load annotations for a single frame

        Args:
            frame_number: Frame to load
            img_shape: Image dimensions for keypoint scaling
            source: Filter by source ('human', 'detection', etc.) or None for all
        """
        with sqlite3.connect(self.db_path) as conn:
            if source == 'detection':
                cursor = conn.execute('''
                                      SELECT frame_number,
                                             runner_id,
                                             class_id,
                                             x_center,
                                             y_center,
                                             width,
                                             height,
                                             confidence,
                                             is_crossing,
                                             keypoints,
                                             source
                                      FROM annotations
                                      WHERE frame_number = ?
                                        AND source != 'human'
                                      ORDER BY runner_id
                                      ''', (frame_number,))
            elif source is not None:
                cursor = conn.execute('''
                                      SELECT frame_number,
                                             runner_id,
                                             class_id,
                                             x_center,
                                             y_center,
                                             width,
                                             height,
                                             confidence,
                                             is_crossing,
                                             keypoints,
                                             source
                                      FROM annotations
                                      WHERE frame_number = ?
                                        AND source = ?
                                      ORDER BY runner_id
                                      ''', (frame_number, source))
            else:
                # Return all annotations, ordered by source priority (human first)
                cursor = conn.execute('''
                                      SELECT frame_number,
                                             runner_id,
                                             class_id,
                                             x_center,
                                             y_center,
                                             width,
                                             height,
                                             confidence,
                                             is_crossing,
                                             keypoints,
                                             source
                                      FROM annotations
                                      WHERE frame_number = ?
                                      ORDER BY CASE source
                                                   WHEN 'human' THEN 1
                                                   WHEN 'tracking' THEN 2
                                                   WHEN 'detection' THEN 3
                                                   ELSE 4
                                                   END,
                                               runner_id
                                      ''', (frame_number,))

            rows = cursor.fetchall()

        return self._format_rows(rows, img_shape)

    def _format_rows(self, rows, img_shape):
        if not rows:
            return np.empty((0, 7)), None, np.array([]), []

        boxes = []
        kpts = []
        crossings = []
        sources = []

        for row in rows:
            _, runner_id, class_id, x_center, y_center, width, height, confidence, is_crossing, keypoints_json, row_source = row

            box = np.array([x_center, y_center, width, height, runner_id, confidence, class_id])
            boxes.append(box)
            crossings.append(bool(is_crossing))

            # Handle keypoints - convert back to pixels
            if keypoints_json:
                kpt_data = np.array(json.loads(keypoints_json))
                # Convert normalized coordinates back to pixels
                if img_shape:
                    kpt_data[:, 0] *= img_shape[1]
                    kpt_data[:, 1] *= img_shape[0]
                kpts.append(kpt_data)

        boxes_array = np.array(boxes) if boxes else np.empty((0, 7))
        if img_shape:
            # Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)
            boxes_array[:, :4] = xywhn2xyxy(boxes_array[:, :4], *img_shape[::-1])
        kpts_array = np.array(kpts) if kpts else None
        crossings_array = np.array(crossings)

        return boxes_array, kpts_array, crossings_array, sources

    def load_all_annotations(self, img_shape=None, source=None, crossing=None) -> Dict[int, Dict[str, any]]:
        """Load all annotations, equivalent to your load_annotations function"""
        query_filters = []
        params = []
        with sqlite3.connect(self.db_path) as conn:
            if crossing is not None:
                query_filters.append("is_crossing = ?")
                params.append(crossing)
            if source:
                if source == 'detection':
                    query_filters.append("source != 'human'")
                else:
                    query_filters.append("source = ?")
                    params.append(source)
            cursor = conn.execute(f'''SELECT frame_number,
                                            runner_id,
                                             class_id,
                                             x_center,
                                             y_center,
                                             width,
                                             height,
                                             confidence,
                                             is_crossing,
                                             keypoints,
                                          source FROM annotations WHERE {' AND '.join(query_filters)} ORDER BY frame_number''',
                                  params)

        annotations = {}
        for frame_number, group in itertools.groupby(cursor, key=lambda x: x[0]):
            boxes, kpts, crossings, sources = self._format_rows(list(group), img_shape)
            # boxes, kpts, crossings = self.remove_duplicates(boxes, kpts, crossings)
            annotations[frame_number] = {'boxes': boxes, 'kpts': kpts, 'crossings': crossings, 'sources': sources}

        return annotations

    def save_notes(self, notes: Dict[int, Dict[str, str]]):
        """Save notes to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing notes
            conn.execute("DELETE FROM notes")

            for frame_num, frame_notes in notes.items():
                for runner_id, note in frame_notes.items():
                    runner_id_int = self._parse_runner_id(runner_id)
                    conn.execute('''
                                 INSERT INTO notes (frame_number, runner_id, note)
                                 VALUES (?, ?, ?)
                                 ''', (frame_num, runner_id_int, note))

    def get_notes(self, frame_number: int) -> Dict[str, str]:
        """Get notes for a specific frame"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT runner_id, note FROM notes WHERE frame_number = ?", (frame_number,))
            notes = {format(row[0], '02x'): row[1] for row in cursor.fetchall()}
        return notes

    def update_notes(self, frame_number: int, runner_id: str, note: str):
        """Update or insert a note for a specific frame and runner"""
        runner_id_int = self._parse_runner_id(runner_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                         INSERT INTO notes (frame_number, runner_id, note)
                         VALUES (?, ?, ?) ON CONFLICT(frame_number, runner_id) DO
                         UPDATE SET note = ?
                         ''', (frame_number, runner_id_int, note, note))

    def load_notes(self) -> Dict[int, Dict[str, str]]:
        """Load notes from database"""
        notes = defaultdict(dict)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT frame_number, runner_id, note FROM notes")
            for frame_num, runner_id, note in cursor.fetchall():
                # Convert runner_id back to hex string for compatibility
                notes[frame_num][format(runner_id, '02x')] = note
        return dict(notes)

    def scan_to_annotation(self, from_frame_num: int, runner_id: int = None,
                           crossing=None, previous=False, max_scan=None, source=None,
                           custom_check=None) -> Optional[int]:
        """Find next/previous annotation matching criteria"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable dict-like access to columns

            # Build the query based on parameters
            conditions = []
            params = []

            if previous:
                conditions.append("frame_number < ?")
                order = "DESC"
            else:
                conditions.append("frame_number > ?")
                order = "ASC"
            params.append(from_frame_num)

            if max_scan is not None:
                if previous:
                    conditions.append("frame_number >= ?")
                    params.append(from_frame_num - max_scan)
                else:
                    conditions.append("frame_number <= ?")
                    params.append(from_frame_num + max_scan)

            if runner_id is not None:
                conditions.append("runner_id = ?")
                params.append(runner_id)

            if crossing is not None:
                conditions.append("is_crossing = ?")
                params.append(crossing)

            if source is not None:
                if source == 'detection':
                    conditions.append("source != 'human'")
                else:
                    conditions.append("source = ?")
                    params.append(source)

            query = f'''
                SELECT * FROM annotations 
                WHERE {' AND '.join(conditions)}
                ORDER BY frame_number {order}
            '''

            # Add LIMIT only if no custom check (for performance)
            if custom_check is None:
                query += " LIMIT 1"

            cursor = conn.execute(query, params)

            if custom_check is None:
                # Simple case - return first match
                result = cursor.fetchone()
                return result['frame_number'] if result else None
            else:
                # Custom check case - iterate until condition is met
                try:
                    for row in cursor:
                        row_dict = dict(row)
                        if custom_check(row_dict):
                            return row_dict['frame_number']
                    return None
                except Exception as e:
                    # Handle lambda evaluation errors gracefully
                    print(f"Error in custom_check: {e}")
                    return None

    def get_nearby(self, to_frame_num: int, buffer_s: int = 5, runner_id: int = None,
                   crossing=None, source=None) -> List[int]:
        """Get nearby frames within buffer_s seconds (30fps assumed)"""
        start_frame = to_frame_num - buffer_s * 30
        end_frame = to_frame_num + buffer_s * 30

        with sqlite3.connect(self.db_path) as conn:
            conditions = ["frame_number BETWEEN ? AND ?"]
            params = [start_frame, end_frame]

            if runner_id is not None:
                conditions.append("runner_id = ?")
                params.append(runner_id)

            if crossing is not None:
                conditions.append("is_crossing = ?")
                params.append(crossing)

            if source is not None:
                conditions.append("source = ?")
                params.append(source)

            query = f'''
                SELECT DISTINCT frame_number FROM annotations 
                WHERE {' AND '.join(conditions)}
                ORDER BY frame_number
            '''

            cursor = conn.execute(query, params)
            return [row[0] for row in cursor.fetchall()]

    def delete_frame_annotation(self, frame_number: int, runner_id: str):
        """Delete annotation for specific runner in frame"""
        runner_id_int = self._parse_runner_id(runner_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                         DELETE
                         FROM annotations
                         WHERE frame_number = ?
                           AND runner_id = ?
                         ''', (frame_number, runner_id_int))

    def mark_frame_crossing(self, frame_number: int, runner_id: str, crossing=None) -> bool:
        """Mark/unmark frame as crossing for specific runner
        returns the new crossing state
        """
        runner_id_int = self._parse_runner_id(runner_id)
        with sqlite3.connect(self.db_path) as conn:
            if crossing is None:
                cursor = conn.execute('''
                                      UPDATE annotations
                                      SET is_crossing = NOT is_crossing
                                      WHERE frame_number = ?
                                        AND runner_id = ? RETURNING is_crossing
                                      ''', (frame_number, runner_id_int))
            else:
                cursor = conn.execute('''
                                      UPDATE annotations
                                      SET is_crossing = ?
                                      WHERE frame_number = ?
                                        AND runner_id = ? RETURNING is_crossing
                                      ''', (crossing, frame_number, runner_id_int))

            result = cursor.fetchone()
            return result[0] if result else None

    def get_last_frame(self, source: str = None) -> Optional[int]:
        """Get the last frame number with annotations"""
        with sqlite3.connect(self.db_path) as conn:
            if source:
                cursor = conn.execute('''
                                      SELECT MAX(frame_number)
                                      FROM annotations
                                      WHERE source = ?
                                      ''', (source,))
            else:
                cursor = conn.execute('SELECT MAX(frame_number) FROM annotations')
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None

    def update_annotation(self, frame_number: int, boxes: Boxes, kpts: Keypoints, crossings, source='human'):
        """Update annotations for a frame - simplest approach using existing methods"""

        # Extract runner_ids from the new boxes
        new_runner_ids = []
        for box in boxes.data:
            runner_id = int(box[4])
            new_runner_ids.append(runner_id)

        # Delete existing annotations for only these specific runners
        if new_runner_ids:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join('?' * len(new_runner_ids))
                conn.execute(f'''
                    DELETE FROM annotations 
                    WHERE frame_number = ? AND source = ? AND runner_id IN ({placeholders})
                ''', [frame_number, source] + new_runner_ids)

        # Use existing save_annotation method to insert the new data
        self.save_annotation(frame_number, boxes, kpts, crossings, source=source, replace=False)

    def _save_frame_data(self, frame_number: int, boxes: np.ndarray, kpts: np.ndarray, crossings: np.ndarray,
                         source: str, replace=False):
        """Internal method to save frame data in internal format"""
        with sqlite3.connect(self.db_path) as conn:
            if replace:
                conn.execute("DELETE FROM annotations WHERE frame_number = ? AND source = ?",
                             (frame_number, source))

            for i, box in enumerate(boxes):
                x1, y1, x2, y2, runner_id, confidence, class_id = box
                is_crossing = bool(crossings[i]) if i < len(crossings) else False

                keypoints_json = None
                if kpts is not None and i < len(kpts):
                    keypoints_json = json.dumps(kpts[i].tolist())

                conn.execute('''
                    INSERT OR REPLACE INTO annotations 
                    (frame_number, runner_id, class_id, x_center, y_center, width, height, confidence, is_crossing, keypoints, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (frame_number, int(runner_id), int(class_id), x1, y1, x2, y2, confidence, is_crossing,
                      keypoints_json, source))

    def reassign_frame_annotation(self, frame_number: int, from_id: str, to_id: str):
        """Reassign runner ID in a frame"""
        from_id_int = self._parse_runner_id(from_id)
        to_id_int = self._parse_runner_id(to_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                         UPDATE annotations
                         SET runner_id = ?
                         WHERE frame_number = ?
                           AND runner_id = ?
                         ''', (to_id_int, frame_number, from_id_int))

    def build_crossing_map(self) -> np.ndarray:
        """Build crossing distance map - much more efficient with SQL"""
        with sqlite3.connect(self.db_path) as conn:
            # Get all frame numbers and runner IDs
            cursor = conn.execute("SELECT DISTINCT frame_number FROM annotations ORDER BY frame_number")
            all_frame_nums = [row[0] for row in cursor.fetchall()]

            cursor = conn.execute("SELECT DISTINCT runner_id FROM annotations ORDER BY runner_id")
            all_runner_ids = [row[0] for row in cursor.fetchall()]

            if not all_frame_nums or not all_runner_ids:
                return np.array([])

            # Create crossing map
            crossing_map = np.full((len(all_runner_ids), len(all_frame_nums)), len(all_frame_nums), dtype=int)

            # Fill in crossings
            cursor = conn.execute('''
                                  SELECT frame_number, runner_id
                                  FROM annotations
                                  WHERE is_crossing = 1
                                  ''')

            for frame_num, runner_id in cursor.fetchall():
                if frame_num in all_frame_nums and runner_id in all_runner_ids:
                    frame_idx = all_frame_nums.index(frame_num)
                    runner_idx = all_runner_ids.index(runner_id)
                    crossing_map[runner_idx, frame_idx] = 0

            # Apply distance calculation (reusing your existing function)
            distance_map = self._distance_to_zeros(crossing_map)

            # Convert to actual frame differences
            valid_idx = np.where(distance_map < len(all_frame_nums))
            pointed_to = np.zeros_like(distance_map)
            pointed_to[valid_idx] = valid_idx[1] + distance_map[valid_idx]
            all_frame_nums_array = np.array(all_frame_nums)
            distance_map[valid_idx] = all_frame_nums_array[pointed_to[valid_idx].flatten()] - all_frame_nums_array[
                valid_idx[1]]

            return distance_map

    @staticmethod
    def _distance_to_zeros(arr: np.ndarray) -> np.ndarray:
        """Reusing your existing distance calculation logic"""
        large_number = arr.shape[1]
        for batch_idx in range(arr.shape[0]):
            x = arr[batch_idx]
            indices = np.arange(x.size)
            zeroes = x == 0
            if not any(zeroes):
                arr[batch_idx] = np.full_like(x, large_number)
                continue
            forward = indices - np.maximum.accumulate(indices * zeroes)
            forward[np.cumsum(zeroes) == 0] = x.size - 1
            forward = forward * (x != 0)

            zeroes = zeroes[::-1]
            backward = indices - np.maximum.accumulate(indices * zeroes)
            backward[np.cumsum(zeroes) == 0] = x.size - 1
            backward = backward[::-1] * (x != 0)

            sign_should_flip = forward < backward
            arr[batch_idx] = np.minimum(forward, backward)
            arr[batch_idx][sign_should_flip] *= -1
        return arr

    @staticmethod
    def remove_duplicates(boxes: np.ndarray, kpts: np.ndarray, crossings: np.ndarray):
        """Remove duplicate runner IDs, keeping highest confidence"""
        if len(boxes) == 0:
            return boxes, kpts, crossings

        track_ids = boxes[:, 4]
        unique_ids = np.unique(track_ids)

        for track_id in unique_ids:
            if track_id == -1:
                continue
            track_idx = np.where(track_ids == track_id)[0]
            if len(track_idx) > 1:
                # Keep the box with highest confidence (index 5)
                max_conf_idx = np.argmax(boxes[track_idx, 5])
                keep_idx = track_idx[max_conf_idx]
                remove_idx = track_idx[track_idx != keep_idx]

                boxes = np.delete(boxes, remove_idx, axis=0)
                if kpts is not None:
                    kpts = np.delete(kpts, remove_idx, axis=0)
                crossings = np.delete(crossings, remove_idx, axis=0)

                # Update track_ids array for next iteration
                track_ids = boxes[:, 4]

        return boxes, kpts, crossings


def offset_with_crop(boxes: Boxes, kpts: Keypoints, crop: List[int], uncropped_shape: Tuple[int, int]):
    """
    uncropped_shape: (w, h)
    """
    # Offset the boxes and keypoints by the crop
    new_boxes = boxes.cpu().numpy().data
    new_kpts = kpts.cpu().numpy().data if kpts is not None else None
    # Offset the boxes (they're xyxy) by the number of preceding cropped pixels
    # Crop is [w, h, x, y]
    new_boxes[:, [0, 2]] += crop[2]
    new_boxes[:, [1, 3]] += crop[3]
    new_boxes = Boxes(new_boxes, uncropped_shape)

    # Offset the keypoints
    if new_kpts is not None:
        new_kpts = new_kpts.reshape((-1, 17, 3))
        for i in range(len(new_kpts)):
            new_kpts[i][:, 0] += crop[2]
            new_kpts[i][:, 1] += crop[3]
        new_kpts = Keypoints(new_kpts, uncropped_shape)
    return new_boxes, new_kpts
