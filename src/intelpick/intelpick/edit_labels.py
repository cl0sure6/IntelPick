"""Review and fix the boxes of photos already in the dataset, without reshooting them.

    ros2 run intelpick edit_labels --data ~/datasets/colors/data.yaml --newest 13
    ros2 run intelpick edit_labels --data ~/datasets/colors/data.yaml --split val

  click        select a box (the smallest one under the cursor)
  1-4          change the selected box to red / green / blue / yellow
  D            delete the selected box
  A            add a box: drag it, ENTER, then 1-4 for its colour
  U            undo all changes to this photo
  R            reject the photo: moves it out of the dataset (into <dataset>_rejected)
  N / -> / SPACE   next photo        P / <-   previous photo
  Q / ESC      quit

Changes are saved when you move to another photo or quit. Before the first change of a session
the split's label folder is copied to <dataset>_labels_backup_<time>, so everything is undoable.
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import yaml

from .dataset import IMAGE_SUFFIXES, check_rows

CLASS_COLORS = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 220, 255)]  # BGR per class id
DISPLAY_WIDTH = 1280


# --- files (no GUI) -------------------------------------------------------------------------

def dataset_root(config):
    with Path(config).expanduser().open() as f:
        data = yaml.safe_load(f)
    return Path(data['path']).expanduser(), list(data['names'])


def list_images(root, split, newest=0):
    images = sorted((p for p in (root / 'images' / split).iterdir()
                     if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.stat().st_mtime)
    return images[-newest:] if newest else images


def label_path(root, split, image):
    return root / 'labels' / split / f'{image.stem}.txt'


def load_boxes(path, width, height, class_count):
    """Pixel boxes [cls, x0, y0, x1, y1] from a YOLO label file."""
    rows = check_rows(path.read_text(), class_count, path) if path.exists() else []
    return [[cls, (cx - w / 2) * width, (cy - h / 2) * height,
             (cx + w / 2) * width, (cy + h / 2) * height] for cls, cx, cy, w, h in rows]


def boxes_to_text(boxes, width, height, class_count):
    lines = []
    for cls, x0, y0, x1, y1 in boxes:
        x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
        y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f'{cls} {(x0 + x1) / 2 / width:.8f} {(y0 + y1) / 2 / height:.8f} '
                     f'{(x1 - x0) / width:.8f} {(y1 - y0) / height:.8f}')
    text = '\n'.join(lines) + ('\n' if lines else '')
    check_rows(text, class_count, 'edited labels')
    return text


def box_at(boxes, x, y):
    """Index of the smallest box containing (x, y), or None: small boxes inside big ones stay
    clickable."""
    hits = [(abs((b[3] - b[1]) * (b[4] - b[2])), i) for i, b in enumerate(boxes)
            if min(b[1], b[3]) <= x <= max(b[1], b[3]) and min(b[2], b[4]) <= y <= max(b[2], b[4])]
    return min(hits)[1] if hits else None


def backup_labels(root, split):
    dest = root.parent / f'{root.name}_labels_backup_{time.strftime("%Y%m%d_%H%M%S")}' / split
    shutil.copytree(root / 'labels' / split, dest)
    return dest


def reject(root, split, image):
    """Move a photo and its label out of the dataset; nothing is deleted."""
    rejected = root.parent / f'{root.name}_rejected'
    for src, kind in ((image, 'images'), (label_path(root, split, image), 'labels')):
        if src.exists():
            dest = rejected / kind / split / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
    return rejected


# --- GUI ------------------------------------------------------------------------------------

class Editor:

    def __init__(self, root, split, names, images):
        self.root, self.split, self.names, self.images = root, split, names, images
        self.i, self.selected, self.note = 0, None, ''
        self.backup = None
        self.changed_photos = self.rejected = 0
        self.window = 'IntelPick labels'
        self.load()

    def load(self):
        self.image = cv2.imread(str(self.images[self.i]))
        self.h, self.w = self.image.shape[:2]
        self.boxes = load_boxes(label_path(self.root, self.split, self.images[self.i]),
                                self.w, self.h, len(self.names))
        self.original = [list(b) for b in self.boxes]
        self.selected = None
        self.scale = min(1.0, DISPLAY_WIDTH / self.w)

    def save(self):
        if self.boxes == self.original:
            return
        if self.backup is None:
            self.backup = backup_labels(self.root, self.split)
            print(f'backup of {self.split} labels: {self.backup}')
        text = boxes_to_text(self.boxes, self.w, self.h, len(self.names))
        label_path(self.root, self.split, self.images[self.i]).write_text(text)
        self.changed_photos += 1
        self.original = [list(b) for b in self.boxes]

    def draw(self):
        view = self.image.copy()
        thick = max(2, int(3 / self.scale))
        for n, (cls, x0, y0, x1, y1) in enumerate(self.boxes):
            p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
            if n == self.selected:
                cv2.rectangle(view, p0, p1, (255, 255, 255), thick * 3)
            cv2.rectangle(view, p0, p1, CLASS_COLORS[cls % 4], thick)
            cv2.putText(view, f'{n + 1} {self.names[cls]}', (p0[0], max(25, p0[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 / self.scale * 0.8, CLASS_COLORS[cls % 4], thick)
        view = cv2.resize(view, (int(self.w * self.scale), int(self.h * self.scale)))
        edited = '  (edited)' if self.boxes != self.original else ''
        lines = [f'[{self.i + 1}/{len(self.images)}] {self.images[self.i].name}  '
                 f'{len(self.boxes)} boxes{edited}  {self.note}',
                 'click select | 1-4 colour | D delete | A add | U undo | R reject photo | '
                 'N next | P prev | Q quit']
        for k, text in enumerate(lines):
            y = 26 + 26 * k
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return view

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected = box_at(self.boxes, x / self.scale, y / self.scale)
            self.note = ''

    def add_box(self):
        view = cv2.resize(self.image, (int(self.w * self.scale), int(self.h * self.scale)))
        x, y, w, h = cv2.selectROI('drag a box, ENTER to accept, ESC to cancel', view,
                                   showCrosshair=True)
        cv2.destroyWindow('drag a box, ENTER to accept, ESC to cancel')
        if w == 0 or h == 0:
            return
        s = self.scale
        self.boxes.append([0, x / s, y / s, (x + w) / s, (y + h) / s])
        self.selected = len(self.boxes) - 1
        self.note = 'new box: press 1-4 for its colour'

    def go(self, step):
        self.save()
        self.i = max(0, min(len(self.images) - 1, self.i + step))
        self.note = ''
        self.load()

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        while True:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKeyEx(50)
            if key == -1:
                continue
            char = key & 0xFF
            if char in (ord('q'), 27):
                self.save()
                break
            if char in (ord('n'), ord(' ')) or key in (65363, 2555904):  # right arrow
                self.go(1)
            elif char == ord('p') or key in (65361, 2424832):  # left arrow
                self.go(-1)
            elif ord('1') <= char < ord('1') + len(self.names) and self.selected is not None:
                self.boxes[self.selected][0] = char - ord('1')
                self.note = ''
            elif char == ord('d') and self.selected is not None:
                del self.boxes[self.selected]
                self.selected = None
            elif char == ord('a'):
                self.add_box()
            elif char == ord('u'):
                self.boxes = [list(b) for b in self.original]
                self.selected, self.note = None, 'changes undone'
            elif char == ord('r'):
                name = self.images[self.i].name
                where = reject(self.root, self.split, self.images[self.i])
                self.rejected += 1
                print(f'rejected {name} -> {where}')
                del self.images[self.i]
                if not self.images:
                    break
                self.i = min(self.i, len(self.images) - 1)
                self.load()
                self.note = f'rejected {name}'
        cv2.destroyAllWindows()
        print(f'{self.changed_photos} photo label files changed, {self.rejected} photos rejected.')
        if self.backup:
            print(f'Backup of the old labels: {self.backup}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data', default='datasets/colors/data.yaml')
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--newest', type=int, default=0, help='only the N most recently added photos')
    args, _ = ap.parse_known_args()
    root, names = dataset_root(args.data)
    images = list_images(root, args.split, args.newest)
    if not images:
        raise SystemExit(f'no photos in {root / "images" / args.split}')
    print(f'{len(images)} photos from {root / "images" / args.split}')
    Editor(root, args.split, names, images).run()
