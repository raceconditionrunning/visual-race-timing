"""NLE-style transport bar for MediaPlayer.

A self-contained widget: give it a display frame plus the player's (paused,
delay) state and it returns a composed image with a clickable control strip
docked below the frame. It has a one-way dependency on the player -- callers
route ``hit()``'s key back through their own key dispatch, so the bar never
reaches into the player.
"""
import cv2
import numpy as np

# Palette (BGR) and text style.
_BG = (38, 38, 42)
_CELL = (60, 60, 66)
_SEP = (78, 78, 86)
_TEXT = (232, 232, 232)
_DIM = (150, 150, 150)
_HINT = (118, 118, 126)  # dim key-shortcut glyphs, one notch above the cell fill
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.5
_THICK = 1
_HINT_SCALE = 0.32

# Printable label for the shortcut key a control maps to.
_KEY_LABELS = {ord(' '): 'spc'}


def _key_label(key):
    return _KEY_LABELS.get(key, chr(key))


class TransportBar:
    def __init__(self, height=56):
        self.height = height
        self.visible = True
        # Declarative widget list; callers extend it via add_seek_steppers.
        self.groups = self._default_groups()
        self._buttons = []   # (x0, y0, x1, y1, key), composed-image coords
        self._frame_h = None  # frame height of the last compose (bar sits below)

    def _default_groups(self):
        """Frame stepping and playback controls. Annotation-aware seek
        granularities are inserted by the caller via add_seek_steppers()."""
        return [
            {'type': 'stepper', 'id': 'frame', 'label': 'Frame', 'prev': ord('('), 'next': ord(')')},
            {'type': 'sep'},
            {'type': 'jump', 'label': '-10s', 'key': ord(',')},
            {'type': 'play'},
            {'type': 'jump', 'label': '+10s', 'key': ord('.')},
            {'type': 'sep'},
            {'type': 'speed'},
        ]

    def add_seek_steppers(self, steppers):
        """Insert extra seek-granularity steppers just after the Frame stepper."""
        idx = next((i for i, g in enumerate(self.groups) if g.get('id') == 'frame'), 0) + 1
        self.groups[idx:idx] = steppers

    def in_chrome(self, y):
        """True if a window y-coordinate falls on the bar rather than the frame."""
        return self._frame_h is not None and y >= self._frame_h

    def hit(self, x, y):
        """Return the key a click at (x, y) maps to, or None if it missed."""
        for (x0, y0, x1, y1, key) in self._buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return key
        return None

    def compose(self, display_frame, *, paused, delay):
        """Draw the bar below display_frame and return the composed image,
        recording button hit rects in composed-image coordinates."""
        h, w = self.height, display_frame.shape[1]
        self._frame_h = display_frame.shape[0]
        bar = np.full((h, w, 3), _BG, dtype=np.uint8)
        self._buttons = []
        x, cy = 12, h // 2
        for widget in self.groups:
            x = self._draw_widget(bar, widget, x, cy, h, paused, delay) + 14
        return np.vstack([display_frame, bar])

    # -- drawing helpers -------------------------------------------------
    def _cell(self, bar, x0, y0, x1, y1):
        cv2.rectangle(bar, (x0, y0), (x1, y1), _CELL, -1)
        cv2.rectangle(bar, (x0, y0), (x1, y1), _SEP, 1)

    def _button(self, bar, x0, y0, x1, y1, key):
        # Hit rects are offset by the frame height (the bar sits below it).
        self._buttons.append((x0, y0 + self._frame_h, x1, y1 + self._frame_h, key))
        # Tuck a dim shortcut glyph into the cell's bottom-right corner.
        lbl = _key_label(key)
        (tw, _th), _ = cv2.getTextSize(lbl, _FONT, _HINT_SCALE, 1)
        cv2.putText(bar, lbl, (x1 - tw - 3, y1 - 3), _FONT, _HINT_SCALE, _HINT, 1, cv2.LINE_AA)

    def _arrow(self, bar, cx, cy, left=True, s=5):
        if left:
            pts = np.array([[cx + s, cy - s], [cx + s, cy + s], [cx - s, cy]], np.int32)
        else:
            pts = np.array([[cx - s, cy - s], [cx - s, cy + s], [cx + s, cy]], np.int32)
        cv2.fillPoly(bar, [pts], _TEXT)

    def _draw_widget(self, bar, widget, x, cy, h, paused, delay):
        y0, y1 = 10, h - 10
        t = widget['type']
        if t == 'sep':
            cv2.line(bar, (x + 4, y0), (x + 4, y1), _SEP, 1)
            return x + 8
        if t == 'stepper':
            cw = 26
            self._cell(bar, x, y0, x + cw, y1)
            self._arrow(bar, x + cw // 2, cy, left=True)
            self._button(bar, x, y0, x + cw, y1, widget['prev'])
            lx = x + cw + 8
            (tw, th), _ = cv2.getTextSize(widget['label'], _FONT, _SCALE, _THICK)
            cv2.putText(bar, widget['label'], (lx, cy + th // 2), _FONT, _SCALE, _TEXT, _THICK, cv2.LINE_AA)
            rx = lx + tw + 8
            self._cell(bar, rx, y0, rx + cw, y1)
            self._arrow(bar, rx + cw // 2, cy, left=False)
            self._button(bar, rx, y0, rx + cw, y1, widget['next'])
            return rx + cw
        if t == 'play':
            cw = 40
            self._cell(bar, x, y0, x + cw, y1)
            ccx = x + cw // 2
            if paused:  # stopped -> offer play
                pts = np.array([[ccx - 6, cy - 8], [ccx - 6, cy + 8], [ccx + 8, cy]], np.int32)
                cv2.fillPoly(bar, [pts], _TEXT)
            else:  # playing -> offer pause
                cv2.rectangle(bar, (ccx - 7, cy - 8), (ccx - 2, cy + 8), _TEXT, -1)
                cv2.rectangle(bar, (ccx + 2, cy - 8), (ccx + 7, cy + 8), _TEXT, -1)
            self._button(bar, x, y0, x + cw, y1, ord(' '))
            return x + cw
        if t == 'jump':
            (tw, th), _ = cv2.getTextSize(widget['label'], _FONT, _SCALE, _THICK)
            cw = tw + 16
            self._cell(bar, x, y0, x + cw, y1)
            cv2.putText(bar, widget['label'], (x + 8, cy + th // 2), _FONT, _SCALE, _TEXT, _THICK, cv2.LINE_AA)
            self._button(bar, x, y0, x + cw, y1, widget['key'])
            return x + cw
        if t == 'speed':
            cw = 24
            self._cell(bar, x, y0, x + cw, y1)
            cv2.putText(bar, '-', (x + cw // 2 - 4, cy + 6), _FONT, 0.6, _TEXT, 2, cv2.LINE_AA)
            self._button(bar, x, y0, x + cw, y1, ord('-'))
            rx = x + cw + 8
            txt = f"{30.0 / max(1, delay):.1f}x"
            (tw, th), _ = cv2.getTextSize(txt, _FONT, _SCALE, _THICK)
            cv2.putText(bar, txt, (rx, cy + th // 2), _FONT, _SCALE, _DIM, _THICK, cv2.LINE_AA)
            px = rx + tw + 8
            self._cell(bar, px, y0, px + cw, y1)
            cv2.putText(bar, '+', (px + cw // 2 - 6, cy + 6), _FONT, 0.6, _TEXT, 2, cv2.LINE_AA)
            self._button(bar, px, y0, px + cw, y1, ord('+'))
            return px + cw
        return x
