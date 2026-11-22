"""edit_labels: everything except the window."""

import cv2
import numpy as np
import pytest

from intelpick.dataset import COLOR_CLASSES, init_dataset, save_sample
from intelpick.edit_labels import (Editor, box_at, boxes_to_text, label_path, list_images,
                                   load_boxes, reject)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'colors'
    init_dataset(root)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    # a big lid boxed only around its inner part, a pen tip, and a real cap
    image = save_sample(root, 'train', frame, [(0, 400, 200, 200, 200), (0, 100, 100, 12, 20),
                                               (2, 800, 500, 60, 60)])
    return root, image


def test_load_and_save_round_trip_is_lossless(dataset):
    root, image = dataset
    path = label_path(root, 'train', image)
    boxes = load_boxes(path, 1280, 720, 4)
    assert [b[0] for b in boxes] == [0, 0, 2]
    assert boxes[0][1:] == pytest.approx([400, 200, 600, 400])
    assert boxes_to_text(boxes, 1280, 720, 4) == path.read_text()


def test_click_picks_the_smallest_box_under_the_cursor():
    boxes = [[0, 0, 0, 500, 500], [1, 100, 100, 150, 150]]
    assert box_at(boxes, 120, 120) == 1  # inside both: the small one
    assert box_at(boxes, 400, 400) == 0
    assert box_at(boxes, 900, 900) is None


def test_boxes_are_clamped_and_degenerate_ones_dropped():
    text = boxes_to_text([[3, -50, -50, 100, 100], [1, 10, 10, 10.5, 40]], 1280, 720, 4)
    assert len(text.splitlines()) == 1 and text.startswith('3 0.0390625')


def test_editor_saves_with_one_backup_and_reject_moves_files(dataset):
    root, image = dataset
    editor = Editor(root, 'train', COLOR_CLASSES, list_images(root, 'train'))
    editor.save()
    assert editor.backup is None  # nothing changed, nothing written

    del editor.boxes[1]  # drop the pen tip
    editor.boxes[0][1:] = [300, 100, 700, 500]  # redraw the lid around all of it
    editor.save()
    assert editor.backup is not None and (editor.backup / f'{image.stem}.txt').exists()
    saved = load_boxes(label_path(root, 'train', image), 1280, 720, 4)
    assert len(saved) == 2 and saved[0][1:] == pytest.approx([300, 100, 700, 500])

    editor.boxes[0][0] = 1
    editor.save()
    assert len(list(root.parent.glob('colors_labels_backup_*'))) == 1  # one backup per session

    where = reject(root, 'train', image)
    assert not image.exists() and not label_path(root, 'train', image).exists()
    assert (where / 'images' / 'train' / image.name).exists()
    assert (where / 'labels' / 'train' / f'{image.stem}.txt').exists()


def test_newest_selects_the_most_recent_photos(tmp_path):
    root = tmp_path / 'colors'
    init_dataset(root)
    frame = np.zeros((10, 10, 3), np.uint8)
    paths = []
    for n in range(3):
        frame[:] = n * 40  # distinct images
        paths.append(save_sample(root, 'train', frame, []))
        cv2.waitKey(1)
    for n, p in enumerate(paths):  # make the order explicit
        import os
        os.utime(p, (1000 + n, 1000 + n))
    assert list_images(root, 'train', newest=2) == paths[1:]
