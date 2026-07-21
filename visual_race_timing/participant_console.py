"""Participant console: a full-width pane docked below the transport bar that
lists every runner with per-runner crossing navigation.

Each row is ``[bib name] [<] n [>]`` -- ``n`` is the runner's confirmed-crossing
count; the arrows flip the playhead between that runner's confirmed crossings.
Once the playhead is at/past a runner's last confirmed crossing, the row turns
amber (``predict``) to signal that ``>`` now seeks an *unconfirmed* estimate.

"""
import cv2
import numpy as np

# Palette (BGR), a touch darker than the transport bar so the pane reads as a
# distinct third strip.
_BG = (30, 30, 34)
_CELL = (60, 60, 66)
_SEP = (78, 78, 86)
_TEXT = (232, 232, 232)
_PREDICT = (60, 180, 240)   # amber: playhead past this runner's last confirmed crossing
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.5

_PAD = 10          # pane inset
_ROW_H = 28        # per-runner row height
_COL_W_MIN = 200   # minimum column width; column count derives from pane width
_BTN_W = 22        # arrow-button width


class ParticipantConsole:
    def __init__(self, participants):
        """participants: dict mapping runner id (int) -> full name (str)."""
        self.participants = dict(participants)
        self.visible = True
        self._buttons = []   # (x0, y0, x1, y1, rid, direction), composed-image coords
        self._above_h = 0    # height of the image the pane was stacked under

    def hit(self, x, y):
        """Return ('runner_seek', rid, direction) for a click on an arrow, else None."""
        for (x0, y0, x1, y1, rid, direction) in self._buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return ('runner_seek', rid, direction)
        return None

    def compose(self, image, rows):
        """Draw the pane below ``image`` and return the stacked result. ``rows``
        is a list of (rid, count, state) with state in {'neutral', 'predict'};
        the caller controls order and contents. Hit rects are recorded in
        composed-image coordinates (offset by ``image``'s height)."""
        w = image.shape[1]
        self._above_h = image.shape[0]
        self._buttons = []
        rows = list(rows)
        cols = max(1, w // _COL_W_MIN)
        n = len(rows)
        n_grid_rows = max(1, (n + cols - 1) // cols)
        pane_h = 2 * _PAD + n_grid_rows * _ROW_H
        pane = np.full((pane_h, w, 3), _BG, dtype=np.uint8)
        col_w = (w - 2 * _PAD) // cols
        for i, row in enumerate(rows):
            c, r = i % cols, i // cols
            self._draw_row(pane, row, _PAD + c * col_w, _PAD + r * _ROW_H, col_w)
        return np.vstack([image, pane])

    # -- drawing helpers -------------------------------------------------
    def _cell(self, pane, x0, y0, x1, y1):
        cv2.rectangle(pane, (x0, y0), (x1, y1), _CELL, -1)
        cv2.rectangle(pane, (x0, y0), (x1, y1), _SEP, 1)

    def _arrow(self, pane, cx, cy, left=True, s=5):
        if left:
            pts = np.array([[cx + s, cy - s], [cx + s, cy + s], [cx - s, cy]], np.int32)
        else:
            pts = np.array([[cx - s, cy - s], [cx - s, cy + s], [cx + s, cy]], np.int32)
        cv2.fillPoly(pane, [pts], _TEXT)

    def _button(self, x0, y0, x1, y1, rid, direction):
        # Hit rects are offset by the height of the image the pane sits under.
        self._buttons.append((x0, y0 + self._above_h, x1, y1 + self._above_h, rid, direction))

    def _truncate(self, text, max_w):
        if cv2.getTextSize(text, _FONT, _SCALE, 1)[0][0] <= max_w:
            return text
        while text and cv2.getTextSize(text + '…', _FONT, _SCALE, 1)[0][0] > max_w:
            text = text[:-1]
        return text + '…' if text else ''

    def _draw_row(self, pane, row, x0, y0, col_w):
        rid, count, state = row
        color = _PREDICT if state == 'predict' else _TEXT
        cy = y0 + _ROW_H // 2
        y_top, y_bot = y0 + 3, y0 + _ROW_H - 3

        # Controls anchored to the cell's right edge: [<] count [>].
        next_x1 = x0 + col_w - 8
        next_x0 = next_x1 - _BTN_W
        count_txt = str(count)
        (ctw, cth), _ = cv2.getTextSize(count_txt, _FONT, _SCALE, 1)
        count_x1 = next_x0 - 8
        count_x0 = count_x1 - ctw
        prev_x1 = count_x0 - 8
        prev_x0 = prev_x1 - _BTN_W

        # Prev / next arrow cells.
        self._cell(pane, prev_x0, y_top, prev_x1, y_bot)
        self._arrow(pane, (prev_x0 + prev_x1) // 2, cy, left=True)
        self._button(prev_x0, y_top, prev_x1, y_bot, rid, -1)
        self._cell(pane, next_x0, y_top, next_x1, y_bot)
        self._arrow(pane, (next_x0 + next_x1) // 2, cy, left=False)
        self._button(next_x0, y_top, next_x1, y_bot, rid, +1)

        # Crossing count, right-aligned before the next-arrow.
        cv2.putText(pane, count_txt, (count_x0, cy + cth // 2), _FONT, _SCALE, color, 1, cv2.LINE_AA)

        # Bib + first name, left-aligned, truncated to the space left of the controls.
        bib = format(rid, '02x')
        name = self.participants.get(rid) or ''
        first = name.split(' ')[0] if name else ''
        label = self._truncate(f"{bib} {first}".strip(), prev_x0 - (x0 + 4) - 8)
        (_, lth), _ = cv2.getTextSize(label, _FONT, _SCALE, 1)
        cv2.putText(pane, label, (x0 + 4, cy + lth // 2), _FONT, _SCALE, color, 1, cv2.LINE_AA)
