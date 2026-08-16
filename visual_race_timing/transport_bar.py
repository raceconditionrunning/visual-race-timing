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

# Timeline density strip.
_TL_TRACK = (50, 50, 56)     # strip background (BGR)
_TL_FILL = (150, 200, 110)   # crossing density (soft green)
_TL_HEAD = (245, 245, 245)   # playhead
_TL_FOCUS_TICK = (60, 180, 240)  # per-lap tick when a single runner is focused (matches the console's amber)
_TL_PAD = 12                 # horizontal inset of the track within the strip
_TL_BINS = 320               # density resolution, independent of pixel width

# Printable label for the shortcut key a control maps to.
_KEY_LABELS = {ord(' '): 'spc'}


def _key_label(key):
    return _KEY_LABELS.get(key, chr(key))


class TransportBar:
    def __init__(self, height=56, strip_height=26):
        self.height = height
        self.strip_height = strip_height
        self.visible = True
        # Declarative widget list; callers extend it via add_seek_steppers.
        self.groups = self._default_groups()
        # Right-anchored checkbox toggles (e.g. "show bboxes"), added via add_toggle.
        self.right_widgets = []
        self._buttons = []   # (x0, y0, x1, y1, key), composed-image coords
        self._frame_h = None  # frame height of the last compose (bar sits below)
        self._buttons_y_offset = 0  # window-y of the control row's top

        # Timeline density strip (disabled until set_timeline is called).
        self._tl_start = None      # first absolute frame number
        self._tl_end = None        # last absolute frame number
        self._tl_density = None    # np array, len _TL_BINS, normalized 0..1
        self._tl_focus_frames = None  # None -> aggregate density; else per-runner tick marks
        self._tl_rect = None       # (x0, y0, x1, y1) hit rect in window coords

    def _default_groups(self):
        """Frame stepping and playback controls. Annotation-aware seek
        granularities are inserted by the caller via add_seek_steppers()."""
        return [
            {'type': 'stepper', 'id': 'frame', 'label': 'Frame', 'prev': ord('('), 'next': ord(')')},
            {'type': 'sep'},
            {'type': 'jump', 'label': '-2.5s', 'key': ord(',')},
            {'type': 'play'},
            {'type': 'jump', 'label': '+2.5s', 'key': ord('.')},
            {'type': 'sep'},
            {'type': 'speed'},
        ]

    def add_seek_steppers(self, steppers):
        """Insert extra seek-granularity steppers just after the Frame stepper."""
        idx = next((i for i, g in enumerate(self.groups) if g.get('id') == 'frame'), 0) + 1
        self.groups[idx:idx] = steppers

    def add_toggle(self, label, key, checked_fn):
        """Add a checkbox docked to the right edge of the button row."""
        self.right_widgets.append({'label': label, 'key': key, 'checked_fn': checked_fn})

    def set_timeline(self, start_frame, end_frame):
        """Enable the density strip over the [start_frame, end_frame] frame span.
        Resets any density until set_marks is called."""
        self._tl_start = int(start_frame)
        self._tl_end = int(end_frame)
        self._tl_density = None

    def set_marks(self, frames):
        """Bin an iterable of timeline positions (absolute frame numbers) into the
        strip's density profile. The bar is agnostic to what the marks mean --
        callers map their own domain (e.g. crossings) onto frame positions. Cheap
        to call again whenever the marks change; drawing just reads the cache."""
        if self._tl_start is None or self._tl_end <= self._tl_start:
            self._tl_density = None
            return
        frames = np.asarray(list(frames), dtype=float)
        frames = frames[(frames >= self._tl_start) & (frames <= self._tl_end)]
        if frames.size == 0:
            self._tl_density = np.zeros(_TL_BINS)
            return
        hist, _ = np.histogram(frames, bins=_TL_BINS, range=(self._tl_start, self._tl_end))
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        smoothed = np.convolve(hist.astype(float), kernel, mode='same')
        peak = smoothed.max()
        self._tl_density = smoothed / peak if peak > 0 else smoothed

    def set_focus_marks(self, frames):
        """Switch the density strip to tick marks at ``frames`` """
        self._tl_focus_frames = None if frames is None else np.asarray(list(frames), dtype=float)

    def in_chrome(self, y):
        """True if a window y-coordinate falls on the bar rather than the frame."""
        return self._frame_h is not None and y >= self._frame_h

    def hit(self, x, y):
        """Return the key a click at (x, y) maps to, or None if it missed."""
        for (x0, y0, x1, y1, key) in self._buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return key
        return None

    def hit_timeline(self, x, y):
        """Return the absolute frame number a click on the density strip maps to,
        or None if the click missed the strip (or the strip is disabled)."""
        if self._tl_rect is None or self._tl_start is None:
            return None
        x0, y0, x1, y1 = self._tl_rect
        if not (y0 <= y < y1 and x0 <= x <= x1):
            return None
        frac = (x - x0) / max(1, x1 - x0)
        return int(round(self._tl_start + frac * (self._tl_end - self._tl_start)))

    def compose(self, display_frame, *, paused, delay, current_frame=None):
        """Draw the bar below display_frame and return the composed image,
        recording button hit rects in composed-image coordinates. When a timeline
        has been set, a density strip is stacked between the frame and controls."""
        h, w = self.height, display_frame.shape[1]
        self._frame_h = display_frame.shape[0]
        strip_h = self.strip_height if self._tl_start is not None else 0
        self._buttons_y_offset = self._frame_h + strip_h

        bar = np.full((h, w, 3), _BG, dtype=np.uint8)
        self._buttons = []
        x, cy = 12, h // 2
        for widget in self.groups:
            x = self._draw_widget(bar, widget, x, cy, h, paused, delay) + 14

        rx = w - 12
        for widget in reversed(self.right_widgets):
            rx = self._draw_checkbox(bar, widget, rx, cy, h) - 14

        if strip_h:
            strip = self._draw_timeline(w, strip_h, current_frame)
            return np.vstack([display_frame, strip, bar])
        self._tl_rect = None
        return np.vstack([display_frame, bar])

    def _draw_timeline(self, w, strip_h, current_frame):
        """Render the density strip and record its click-to-seek hit rect."""
        strip = np.full((strip_h, w, 3), _BG, dtype=np.uint8)
        x0, x1 = _TL_PAD, w - _TL_PAD
        y0, y1 = 5, strip_h - 5
        track_w, track_h = x1 - x0, y1 - y0
        cv2.rectangle(strip, (x0, y0), (x1, y1), _TL_TRACK, -1)

        if self._tl_focus_frames is not None:
            self._draw_focus_ticks(strip, x0, x1, y0, y1)
        elif self._tl_density is not None and track_w > 1 and track_h > 1:
            cols = np.arange(track_w)
            bins = (cols / track_w * len(self._tl_density)).astype(int)
            bins = np.clip(bins, 0, len(self._tl_density) - 1)
            ys = (y1 - self._tl_density[bins] * (track_h - 1)).astype(np.int32)
            top = np.stack([x0 + cols, ys], axis=1)
            poly = np.vstack([top, [[x1, y1], [x0, y1]]]).astype(np.int32)
            cv2.fillPoly(strip, [poly], _TL_FILL)

        # Hit rect in window coords (the strip sits just below the frame).
        self._tl_rect = (x0, self._frame_h, x1, self._frame_h + strip_h)

        if current_frame is not None and self._tl_end > self._tl_start:
            frac = (current_frame - self._tl_start) / (self._tl_end - self._tl_start)
            hx = x0 + int(round(min(1.0, max(0.0, frac)) * track_w))
            cv2.line(strip, (hx, y0 - 2), (hx, y1 + 1), _TL_HEAD, 1)
            tri = np.array([[hx - 3, 0], [hx + 3, 0], [hx, y0 - 1]], np.int32)
            cv2.fillPoly(strip, [tri], _TL_HEAD)
        return strip

    def _draw_focus_ticks(self, strip, x0, x1, y0, y1):
        """Draw one full-height tick per frame in ``_tl_focus_frames``"""
        if self._tl_start is None or self._tl_end <= self._tl_start:
            return
        track_w = x1 - x0
        frames = self._tl_focus_frames
        frames = frames[(frames >= self._tl_start) & (frames <= self._tl_end)]
        for f in frames:
            frac = (f - self._tl_start) / (self._tl_end - self._tl_start)
            tx = x0 + int(round(frac * track_w))
            cv2.line(strip, (tx, y0), (tx, y1), _TL_FOCUS_TICK, 2)

    # -- drawing helpers -------------------------------------------------
    def _cell(self, bar, x0, y0, x1, y1):
        cv2.rectangle(bar, (x0, y0), (x1, y1), _CELL, -1)
        cv2.rectangle(bar, (x0, y0), (x1, y1), _SEP, 1)

    def _button(self, bar, x0, y0, x1, y1, key):
        # Hit rects are offset by everything above the control row (frame + strip).
        off = self._buttons_y_offset
        self._buttons.append((x0, y0 + off, x1, y1 + off, key))
        # Tuck a dim shortcut glyph into the cell's bottom-right corner.
        lbl = _key_label(key)
        (tw, _th), _ = cv2.getTextSize(lbl, _FONT, _HINT_SCALE, 1)
        cv2.putText(bar, lbl, (x1 - tw - 3, y1 - 3), _FONT, _HINT_SCALE, _HINT, 1, cv2.LINE_AA)

    def _draw_checkbox(self, bar, widget, x1, cy, h):
        """Draw a checkbox anchored so its right edge lands at x1 (box, then
        label, growing leftward). Returns the widget's left edge, so callers
        can chain further right_widgets further left."""
        y0, y1 = 10, h - 10
        box_s = 14
        label = widget['label']
        (tw, th), _ = cv2.getTextSize(label, _FONT, _SCALE, _THICK)
        cw = box_s + 6 + tw + 8
        x0 = x1 - cw
        bx0, by0 = x0 + 4, cy - box_s // 2
        bx1, by1 = bx0 + box_s, by0 + box_s
        cv2.rectangle(bar, (bx0, by0), (bx1, by1), _SEP, 1)
        if widget['checked_fn']():
            cv2.line(bar, (bx0 + 2, cy), (bx0 + 5, by1 - 2), _TEXT, 2)
            cv2.line(bar, (bx0 + 5, by1 - 2), (bx1 - 2, by0 + 2), _TEXT, 2)
        cv2.putText(bar, label, (bx1 + 6, cy + th // 2), _FONT, _SCALE, _TEXT, _THICK, cv2.LINE_AA)
        off = self._buttons_y_offset
        self._buttons.append((x0, y0 + off, x1, y1 + off, widget['key']))
        return x0

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
