"""Capture colour examples from the same camera used for sorting and box them for training.

The HSV detector suggests boxes on each frozen frame. They are only suggestions: every photo is
checked by hand (recolour, delete, add missing boxes) before it is saved as training truth.
"""

import argparse

import cv2

from .camera_source import FrameSource
from .dataset import COLOR_CLASSES, init_dataset, save_sample
from .detectors.color import ColorDetector
from .tune_colors import load_start

BOX_COLORS = [(0, 0, 255), (0, 200, 0), (255, 80, 0), (0, 220, 255)]  # BGR, COLOR_CLASSES order
UNLABELLED = (255, 0, 255)
MIN_DRAG = 5  # px; a shorter drag is a click


def put_text(image, text, org, color=(255, 255, 255)):
    """Text with a dark outline, readable on any table."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_hints(image, *lines):
    """Key hints on the image itself; the window title is often cut off."""
    for i, line in enumerate(lines):
        put_text(image, line, (10, 25 + 22 * i))
    return image


def _inside(box, outer):
    _, x, y, w, h = box
    _, ox, oy, ow, oh = outer
    return ox <= x and oy <= y and x + w <= ox + ow and y + h <= oy + oh and w * h < ow * oh


def suggest(detector, frame):
    """HSV boxes as [class_id, x, y, w, h]; colours the dataset has no class for are dropped."""
    boxes = [[COLOR_CLASSES.index(name), x, y, w, h]
             for name, x, y, w, h in detector.boxes(frame) if name in COLOR_CLASSES]
    # Shiny or see-through objects break into pieces: keep only the outer box of a colour.
    return [b for b in boxes if not any(o[0] == b[0] and _inside(b, o) for o in boxes)]


class Review:
    """Boxes on one frozen frame, suggested then corrected by hand. Class None = no colour yet."""

    def __init__(self, frame, boxes=()):
        self.frame = frame
        self.boxes = [list(b) for b in boxes]
        self.selected = None
        self.note = ''
        self.show_hints = True

    def box_at(self, x, y):
        """Smallest box containing the point, so a box inside another one stays reachable."""
        hits = [i for i, (_, bx, by, w, h) in enumerate(self.boxes)
                if bx <= x < bx + w and by <= y < by + h]
        return min(hits, key=lambda i: self.boxes[i][3] * self.boxes[i][4], default=None)

    def drag(self, x0, y0, x1, y1):
        """A drag adds a box with no colour and selects it; a short one selects the box clicked."""
        height, width = self.frame.shape[:2]
        x0, x1 = sorted(min(max(x, 0), width) for x in (x0, x1))
        y0, y1 = sorted(min(max(y, 0), height) for y in (y0, y1))
        if x1 - x0 < MIN_DRAG or y1 - y0 < MIN_DRAG:
            self.selected = self.box_at(x1, y1)
            return
        self.boxes.append([None, x0, y0, x1 - x0, y1 - y0])
        self.selected = len(self.boxes) - 1

    def label(self, cls):
        if self.selected is None:
            self.note = 'click a box first, then press its colour'
        else:
            self.boxes[self.selected][0] = cls

    def delete(self, index):
        if index is not None:
            del self.boxes[index]
            self.selected = None

    def next(self):
        if self.boxes:
            self.selected = 0 if self.selected is None else (self.selected + 1) % len(self.boxes)

    def annotations(self):
        """(class_id, x, y, w, h) to save, or None while a box still has no colour."""
        if any(cls is None for cls, *_ in self.boxes):
            return None
        return [tuple(b) for b in self.boxes]

    def render(self, dragging=None):
        image = self.frame.copy()
        for i, (cls, x, y, w, h) in enumerate(self.boxes):
            color = UNLABELLED if cls is None else BOX_COLORS[cls]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 4 if i == self.selected else 2)
            put_text(image, '?' if cls is None else COLOR_CLASSES[cls], (x, max(15, y - 5)), color)
        if dragging is not None:
            cv2.rectangle(image, dragging[:2], dragging[2:], (255, 255, 255), 1)
        if self.show_hints:
            draw_hints(image, f'{len(self.boxes)} boxes | ENTER save photo | ESC skip photo',
                       'drag: add box | click box, then 1 red 2 green 3 blue 4 yellow',
                       'D / right-click: delete box | TAB: next box | H: hide help')
        if self.note:
            put_text(image, self.note, (10, image.shape[0] - 12), UNLABELLED)
        return image


def annotate(frame, suggested=()):
    """Check the boxes on a frozen frame. Returns the annotations to save, or None to skip it."""
    review = Review(frame, suggested)
    window = 'Check boxes: ENTER save, ESC skip'
    drag = {}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag['start'] = drag['end'] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag:
            drag['end'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag:
            review.drag(*drag.pop('start'), x, y)
            drag.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            review.delete(review.box_at(x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            dragging = (*drag['start'], *drag['end']) if drag else None
            cv2.imshow(window, review.render(dragging))
            key = cv2.waitKey(30) & 0xFF
            if key == 255:  # no key
                continue
            review.note = ''
            if key == 27:
                return None
            if key in (13, 10):  # Enter; GTK reports 10
                result = review.annotations()
                if result is not None:
                    return result
                review.note = 'give every magenta "?" box a colour (or delete it)'
            elif ord('1') <= key < ord('1') + len(COLOR_CLASSES):
                review.label(key - ord('1'))
            elif key in (ord('d'), 8):  # D or Backspace
                review.delete(review.selected)
            elif key == 9:  # TAB
                review.next()
            elif key == ord('h'):
                review.show_hints = not review.show_hints
    finally:
        cv2.destroyWindow(window)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera', default='/dev/video0')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--out', default='datasets/colors')
    ap.add_argument('--split', choices=['train', 'val'], required=True)
    ap.add_argument('--no-suggest', action='store_true',
                    help='start every photo with no boxes instead of HSV suggestions')
    args = ap.parse_args()
    config = init_dataset(args.out)
    # Same HSV ranges as the detector node and tune_colors: the installed config, else defaults.
    detector = None if args.no_suggest else ColorDetector(*load_start())
    print(f'Dataset: {config}\nSPACE: freeze and check boxes; B: save empty background; Q: quit')
    cam = FrameSource(args.camera, args.width, args.height)
    latest = None
    try:
        while True:
            frame = cam.read()
            if frame is not None:
                latest = frame
            # A displayed old frame must not be saved again as if it were a new capture.
            if latest is not None:
                preview = draw_hints(latest.copy(),
                                     f'{args.split}: SPACE label | B background | Q quit')
                cv2.imshow('IntelPick dataset', preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key not in (ord(' '), ord('b')):
                continue
            fresh = cam.read()
            if fresh is None:
                fresh = frame
            if fresh is None:
                print('No fresh camera frame; wait for the stream to reconnect.')
                continue
            suggested = suggest(detector, fresh) if detector else []
            # B with objects in view would teach the network they are background: show them first.
            if key == ord(' ') or suggested:
                if key == ord('b'):
                    print(f'B: HSV sees {len(suggested)} objects. Delete the false ones; ENTER '
                          'saves what is left (no boxes = background).')
                annotations = annotate(fresh, suggested)
                if annotations is None:
                    print('Photo skipped, nothing saved.')
                    continue
                if key == ord(' ') and not annotations:
                    print('No boxes; sample skipped. B explicitly saves background.')
                    continue
            else:
                annotations = []
            path = save_sample(config.parent, args.split, fresh, annotations)
            print(f'Saved {path.name}: {len(annotations)} objects. Back to the live view '
                  '(click it if keys do nothing); rearrange and press SPACE for the next photo.')
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
