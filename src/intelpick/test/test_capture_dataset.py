"""capture_dataset suggestion and review logic, without opening a window."""

import os

import cv2
import numpy as np
import pytest

from intelpick.capture_dataset import Review, label_file, reject, saved_photos, suggest
from intelpick.dataset import (COLOR_CLASSES, check_rows, init_dataset, label_text, pixel_boxes,
                               save_sample)
from intelpick.detectors.color import DEFAULT_COLORS, ColorDetector


def _scene():
    img = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(img, (100, 100), (159, 159), (0, 0, 220), -1)  # red, 60x60
    cv2.rectangle(img, (400, 300), (449, 349), (220, 60, 0), -1)  # blue, 50x50
    return img


def test_hsv_suggestions_are_class_ids_with_tight_boxes():
    boxes = sorted(suggest(ColorDetector(DEFAULT_COLORS), _scene()))
    assert [b[0] for b in boxes] == [COLOR_CLASSES.index('red'), COLOR_CLASSES.index('blue')]
    for (_, x, y, w, h), expected in zip(boxes, [(100, 100, 60, 60), (400, 300, 50, 50)]):
        assert np.allclose((x, y, w, h), expected, atol=3)


def test_colours_without_a_dataset_class_are_not_suggested():
    colors = {'blue': DEFAULT_COLORS['blue'], 'purple': [130, 80, 50, 160, 255, 255]}
    img = _scene()
    cv2.rectangle(img, (250, 50), (299, 99), (200, 0, 150), -1)
    assert [b[0] for b in suggest(ColorDetector(colors), img)] == [COLOR_CLASSES.index('blue')]


def test_pieces_inside_a_box_of_the_same_colour_are_dropped():
    img = _scene()
    cv2.rectangle(img, (115, 100), (144, 144), (200, 200, 200), -1)  # glare notch in the red
    cv2.rectangle(img, (122, 108), (137, 123), (0, 0, 220), -1)  # red piece left in the notch
    cv2.rectangle(img, (415, 315), (434, 334), (0, 0, 220), -1)  # red label on the blue: kept
    boxes = suggest(ColorDetector(DEFAULT_COLORS, min_area=50), img)
    assert sorted(b[0] for b in boxes) == [0, 0, 2]
    assert max(b[3] for b in boxes if b[0] == 0) >= 55


def test_accepting_suggestions_saves_valid_labels(tmp_path):
    img = _scene()
    review = Review(img, suggest(ColorDetector(DEFAULT_COLORS), img))
    root = init_dataset(tmp_path).parent
    image = save_sample(root, 'train', img, review.annotations())
    rows = check_rows((root / 'labels' / 'train' / f'{image.stem}.txt').read_text(), 4, 'saved')
    assert len(rows) == 2


def test_drawn_box_blocks_saving_until_it_has_a_colour():
    review = Review(_scene())
    review.drag(300, 200, 250, 150)  # dragged up-left
    assert review.boxes == [[None, 250, 150, 50, 50]] and review.selected == 0
    assert review.annotations() is None
    review.label(COLOR_CLASSES.index('green'))
    assert review.annotations() == [(1, 250, 150, 50, 50)]


def test_drag_past_the_edge_is_clamped_to_the_image():
    review = Review(_scene())
    review.drag(600, 450, 700, 520)
    assert review.boxes == [[None, 600, 450, 40, 30]]


def test_click_selects_smallest_box_and_relabels_or_deletes_it():
    review = Review(_scene(), [[0, 50, 50, 300, 300], [2, 100, 100, 60, 60]])
    review.drag(120, 120, 121, 121)  # a click, not a new box
    assert len(review.boxes) == 2 and review.selected == 1
    review.label(COLOR_CLASSES.index('yellow'))
    assert review.boxes[1][0] == 3
    review.delete(review.selected)
    assert review.boxes == [[0, 50, 50, 300, 300]] and review.selected is None
    review.drag(500, 20, 501, 21)  # click on empty table deselects
    assert review.selected is None


def test_label_without_selection_only_explains():
    review = Review(_scene(), [[0, 100, 100, 60, 60]])
    review.label(2)
    assert review.boxes[0][0] == 0 and review.note


def test_tab_cycles_selection_and_render_keeps_frame_untouched():
    img = _scene()
    review = Review(img, [[0, 100, 100, 60, 60], [2, 400, 300, 50, 50]])
    review.next()
    review.next()
    review.next()
    assert review.selected == 0
    view = review.render((10, 10, 40, 40))
    assert view.shape == img.shape and not np.array_equal(view, img)
    assert np.array_equal(img, _scene())


def test_saved_labels_round_trip_to_the_same_pixel_boxes():
    boxes = [(0, 100, 100, 60, 60), (3, 0, 0, 1280, 720), (2, 1201, 37, 79, 11)]
    assert pixel_boxes(label_text(boxes, 1280, 720), 1280, 720) == boxes


def _saved(tmp_path, count):
    root = init_dataset(tmp_path / 'colors').parent
    images = []
    for i in range(count):
        images.append(save_sample(root, 'train', _scene(), [(i % 4, 10 * i, 10, 50, 50)]))
        os.utime(images[-1], (1000 + i, 1000 + i))  # capture order independent of uuid names
    return root, images


def test_review_lists_photos_in_capture_order_and_limits_to_newest(tmp_path):
    root, images = _saved(tmp_path, 4)
    assert saved_photos(root, 'train') == images
    assert saved_photos(root, 'train', newest=2) == images[2:]
    assert label_file(images[0]) == root / 'labels' / 'train' / f'{images[0].stem}.txt'


def test_rejected_photo_and_labels_move_next_to_the_dataset(tmp_path):
    root, images = _saved(tmp_path, 2)
    label = label_file(images[0]).read_text()
    target = reject(images[0])
    assert target == tmp_path / 'colors_rejected'
    assert (target / 'images' / 'train' / images[0].name).exists()
    assert (target / 'labels' / 'train' / f'{images[0].stem}.txt').read_text() == label
    assert saved_photos(root, 'train') == images[1:]


def test_reject_never_overwrites_an_earlier_rejected_photo(tmp_path):
    root, images = _saved(tmp_path, 1)
    clash = tmp_path / 'colors_rejected' / 'labels' / 'train' / f'{images[0].stem}.txt'
    clash.parent.mkdir(parents=True)
    clash.write_text('keep me\n')
    with pytest.raises(FileExistsError):
        reject(images[0])
    assert images[0].exists() and clash.read_text() == 'keep me\n'
