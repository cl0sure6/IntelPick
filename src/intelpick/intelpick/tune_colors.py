"""Tune the HSV colour ranges on the live camera image, then paste them into the config.

    ros2 run intelpick tune_colors --camera http://172.20.10.1:4747/video
    ros2 run intelpick tune_colors --camera /dev/video0

Two windows: the camera image marked with what the detector would find, and the mask of the
colour being tuned (white = counted as that colour). The sliders set the selected range.

  1-9          select colour (in config order: 1 red, 2 green, 3 blue, 4 yellow)
  TAB          next range of that colour (red has two: hue wraps around 0/179)
  left click   show the HSV value under the cursor
  right click  set the selected range around the clicked object
  SPACE        freeze / unfreeze the image          P  print the config block
  Q / ESC      quit and print the config block

Starts from the ranges in the installed config. Stop camera_node first if it uses the same
camera: phone apps like DroidCam serve one viewer at a time.
"""

import argparse
import time

import cv2
import numpy as np

from .camera_source import FrameSource
from .detectors.color import DEFAULT_COLORS, ColorDetector
from .params import installed_params

SLIDERS = [('H min', 179), ('S min', 255), ('V min', 255),
           ('H max', 179), ('S max', 255), ('V max', 255)]
DISPLAY_WIDTH = 960


class ColorRanges:
    """The colours being tuned: each a list of [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi] ranges."""

    def __init__(self, colors):
        self.ranges = {name: [list(map(int, flat[i:i + 6])) for i in range(0, len(flat), 6)]
                       for name, flat in colors.items()}
        self.names = list(self.ranges)
        self.color, self.index = self.names[0], 0

    def select(self, name):
        self.color, self.index = name, 0

    def next_range(self):
        self.index = (self.index + 1) % len(self.ranges[self.color])

    @property
    def current(self):
        return list(self.ranges[self.color][self.index])

    def set_current(self, values):
        lo, hi = list(map(int, values[:3])), list(map(int, values[3:]))
        self.ranges[self.color][self.index] = lo + [max(a, b) for a, b in zip(lo, hi)]

    def set_around(self, h, s, v):
        """Range around a sampled pixel. Near the hue wrap (red) a colour with two ranges gets
        both halves set: [0, h+8] and [h-8, 179] modulo 180."""
        lo_hue, hi_hue = h - 8, h + 8
        r = range_around(h, s, v)
        if len(self.ranges[self.color]) == 2 and (lo_hue < 0 or hi_hue > 179):
            self.ranges[self.color] = [[0, r[1], r[2], hi_hue % 180, 255, 255],
                                       [lo_hue % 180, r[1], r[2], 179, 255, 255]]
            self.index = 0
        else:
            self.set_current(r)

    def flat(self):
        return {n: [v for r in ranges for v in r] for n, ranges in self.ranges.items()}


def sample_hsv(bgr, x, y, radius=2):
    """Median HSV of a small patch: steadier than one pixel on a noisy camera image."""
    h, w = bgr.shape[:2]
    patch = bgr[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return tuple(int(v) for v in np.median(hsv, axis=0))


def range_around(h, s, v):
    """A starting range for an object with this HSV: narrow in hue, generous in saturation
    and brightness, which change with shadows and glare far more than hue does."""
    return [max(0, h - 8), max(40, s - 70), max(40, v - 90), min(179, h + 8), 255, 255]


def format_colors(flat, indent=4):
    pad = ' ' * indent
    lines = [f'{pad}colors:'] + [f'{pad}  {name}: [{", ".join(map(str, values))}]'
                                 for name, values in flat.items()]
    return '\n'.join(lines)


def render(frame, ranges, min_area, max_area, note=''):
    """(annotated camera image, mask of the selected colour) at full resolution."""
    detector = ColorDetector(ranges.flat(), min_area, max_area)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = detector.mask(hsv, ranges.color)
    view = frame.copy()
    for b in detector.detect(frame):
        color = (255, 255, 255) if b.label == ranges.color else (160, 160, 160)
        cv2.circle(view, (int(b.u), int(b.v)), 6, color, -1)
        cv2.putText(view, b.label, (int(b.u) + 9, int(b.v) - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    total = len(ranges.ranges[ranges.color])
    header = f'{ranges.color} range {ranges.index + 1}/{total}: {ranges.current}  {note}'
    for img in (view, mask):
        cv2.putText(img, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return view, mask


def load_start():
    """Colour names, ranges and area limits from the installed config, else the defaults."""
    (det,) = installed_params('detector')
    names = det.get('color_names', list(DEFAULT_COLORS))
    configured = det.get('colors', {})
    colors = {n: configured.get(n, DEFAULT_COLORS.get(n)) for n in names}
    missing = [n for n, v in colors.items() if not v]
    if missing:
        raise SystemExit(f'no HSV range for {missing}; add one to the config first')
    return colors, det.get('min_area', 300), det.get('max_area', 40000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--camera', default='/dev/video0', help='device, index or stream URL')
    ap.add_argument('--width', type=int, default=640, help='USB cameras only')
    ap.add_argument('--height', type=int, default=480, help='USB cameras only')
    ap.add_argument('--snapshot', help='render one frame to this image file and exit (no window)')
    args, _ = ap.parse_known_args()

    colors, min_area, max_area = load_start()
    ranges = ColorRanges(colors)
    try:
        cam = FrameSource(args.camera, args.width, args.height)
    except RuntimeError as e:
        raise SystemExit(str(e))

    state = {'note': 'right-click an object to set the range', 'scale': 1.0, 'frame': None}
    image_win, mask_win = 'IntelPick colours', 'IntelPick mask (white = this colour)'

    def set_sliders():
        for (name, _), value in zip(SLIDERS, ranges.current):
            cv2.setTrackbarPos(name, mask_win, value)

    def on_mouse(event, x, y, *_):
        frame = state['frame']
        if frame is None or event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        x, y = int(x / state['scale']), int(y / state['scale'])
        h, s, v = sample_hsv(frame, x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state['note'] = f'HSV at ({x}, {y}) = ({h}, {s}, {v})'
        else:
            ranges.set_around(h, s, v)
            set_sliders()
            state['note'] = f'range set around HSV ({h}, {s}, {v})'

    if not args.snapshot:
        cv2.namedWindow(image_win)
        cv2.namedWindow(mask_win)
        for name, top in SLIDERS:
            cv2.createTrackbar(name, mask_win, 0, top, lambda _: None)
        set_sliders()
        cv2.setMouseCallback(image_win, on_mouse)

    frozen = False
    give_up = time.monotonic() + 15.0  # --snapshot only: no camera, no point waiting forever
    try:
        while True:
            if not frozen:
                new = cam.read()
                if new is not None:
                    state['frame'] = new
            frame = state['frame']
            if frame is None:
                if args.snapshot and time.monotonic() > give_up:
                    raise SystemExit(f'no frame from {cam.name} in 15 s')
                if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                    break
                continue
            if not args.snapshot:
                ranges.set_current([cv2.getTrackbarPos(name, mask_win) for name, _ in SLIDERS])
            note = ('FROZEN  ' if frozen else '') + state['note']
            view, mask = render(frame, ranges, min_area, max_area, note)
            if args.snapshot:
                cv2.imwrite(args.snapshot, np.hstack([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
                print(f'wrote {args.snapshot}')
                break
            state['scale'] = min(1.0, DISPLAY_WIDTH / frame.shape[1])
            size = (int(frame.shape[1] * state['scale']), int(frame.shape[0] * state['scale']))
            cv2.imshow(image_win, cv2.resize(view, size))
            cv2.imshow(mask_win, cv2.resize(mask, size))

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if ord('1') <= key <= ord('9') and key - ord('1') < len(ranges.names):
                ranges.select(ranges.names[key - ord('1')])
                set_sliders()
            elif key == 9:  # TAB
                ranges.next_range()
                set_sliders()
            elif key == ord(' '):
                frozen = not frozen
            elif key == ord('p'):
                print(format_colors(ranges.flat()))
    finally:
        cam.close()
        cv2.destroyAllWindows()
    if not args.snapshot:
        print('\nPaste into src/intelpick/config/intelpick.yaml under detector: ros__parameters:')
        print(format_colors(ranges.flat()))
