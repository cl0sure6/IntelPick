"""Create and check the local YOLO colour dataset used by the capture/training tools."""

import hashlib
import math
from pathlib import Path
from uuid import uuid4

import cv2
import yaml


COLOR_CLASSES = ['red', 'green', 'blue', 'yellow']
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def init_dataset(root, names=COLOR_CLASSES):
    root = Path(root).expanduser().resolve()
    names = list(names)
    if not names or len(names) != len(set(names)) or any(not n.strip() for n in names):
        raise ValueError('class names must be nonempty and unique')
    config = root / 'data.yaml'
    data = {'path': str(root), 'train': 'images/train', 'val': 'images/val', 'names': names}
    if config.exists():
        with config.open() as f:
            existing = yaml.safe_load(f)
        if existing != data:
            raise ValueError(f'{config} already exists with different settings; use another directory')
    for split in ('train', 'val'):
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    if not config.exists():
        with config.open('x') as f:
            yaml.safe_dump(data, f, sort_keys=False)
    return config


def check_rows(text, class_count, source):
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError('expected class cx cy width height')
            cls = int(fields[0])
            cx, cy, w, h = map(float, fields[1:])
            if not 0 <= cls < class_count or not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError('invalid class or nonfinite coordinate')
            if not 0 < w <= 1 or not 0 < h <= 1 or not (
                    w / 2 - 1e-6 <= cx <= 1 - w / 2 + 1e-6 and
                    h / 2 - 1e-6 <= cy <= 1 - h / 2 + 1e-6):
                raise ValueError('box must have positive size and fit inside normalized image bounds')
            rows.append((cls, cx, cy, w, h))
        except ValueError as e:
            raise ValueError(f'{source}:{line_number}: {e}') from e
    return rows


def label_text(annotations, width, height, class_count=4):
    """YOLO label file text for (class_id, pixel x, y, width, height) boxes, checked."""
    lines = []
    for cls, x, y, w, h in annotations:
        lines.append(f'{cls} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} '
                     f'{w / width:.8f} {h / height:.8f}')
    text = '\n'.join(lines) + ('\n' if lines else '')
    check_rows(text, class_count, 'captured annotations')
    return text


def pixel_boxes(text, width, height, class_count=4, source='labels'):
    """Inverse of label_text: (class_id, pixel x, y, width, height) for each row."""
    boxes = []
    for cls, cx, cy, w, h in check_rows(text, class_count, source):
        x0, y0 = round((cx - w / 2) * width), round((cy - h / 2) * height)
        x1, y1 = round((cx + w / 2) * width), round((cy + h / 2) * height)
        boxes.append((cls, max(x0, 0), max(y0, 0), min(x1, width) - max(x0, 0),
                      min(y1, height) - max(y0, 0)))
    return boxes


def save_sample(root, split, frame, annotations, class_count=4):
    """annotations: (class_id, pixel x, y, width, height). Empty means reviewed background."""
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    height, width = frame.shape[:2]
    text = label_text(annotations, width, height, class_count)
    root = Path(root).expanduser().resolve()
    stem = uuid4().hex
    image = root / 'images' / split / f'{stem}.jpg'
    label = root / 'labels' / split / f'{stem}.txt'
    if not cv2.imwrite(str(image), frame):
        raise RuntimeError(f'could not write {image}')
    try:
        label.write_text(text)
    except OSError:
        image.unlink()
        raise
    return image


def validate_dataset(config):
    """Check our directory-based detect format before any expensive training or downloads.

    Require explicit empty label files for reviewed background, so forgotten annotations
    cannot silently teach YOLO that coloured objects are background.
    """
    config = Path(config).expanduser().resolve()
    with config.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError('dataset YAML must be a mapping')
    names = data.get('names')
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError('class IDs must be contiguous and start at 0')
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or any(
            not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError('dataset needs unique, nonempty class names')
    root = Path(data.get('path', config.parent)).expanduser()
    if not root.is_absolute():
        # Ultralytics can resolve relative roots against its global datasets_dir setting,
        # rather than this YAML's parent. Require one unambiguous root for check and train.
        raise ValueError('dataset path must be absolute; update path in data.yaml')
    seen = {}
    summary = {}
    for split in ('train', 'val'):
        # Keep the helper's supported layout explicit; the raw yolo CLI supports other formats.
        if data.get(split) != f'images/{split}':
            raise ValueError(f'{split} must be images/{split}; see docs/YOLO.md')
        images = sorted(p for p in (root / 'images' / split).glob('*')
                        if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f'no images in {root / "images" / split}')
        counts = [0] * len(names)
        for image in images:
            decoded = cv2.imread(str(image))
            if decoded is None:
                raise ValueError(f'unreadable image: {image}')
            digest = hashlib.sha256(str(decoded.shape).encode() + decoded.tobytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f'train/val leakage: duplicate image {image}')
            seen[digest] = split
            label = root / 'labels' / split / f'{image.stem}.txt'
            if not label.is_file():
                raise ValueError(f'missing annotation: {label}; use an empty file for background')
            for cls, *_ in check_rows(label.read_text(), len(names), label):
                counts[cls] += 1
        missing = [name for name, count in zip(names, counts) if count == 0]
        if missing:
            raise ValueError(f'{split} has no labelled examples for: {", ".join(missing)}')
        summary[split] = {'images': len(images), 'objects': dict(zip(names, counts))}
    return summary
