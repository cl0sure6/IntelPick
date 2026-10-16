"""tune_colors logic, without opening a window."""

import cv2
import numpy as np
import yaml

from intelpick.detectors.color import DEFAULT_COLORS, ColorDetector
from intelpick.tune_colors import ColorRanges, format_colors, range_around, render, sample_hsv


def _scene():
    img = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(img, (100, 100), (160, 160), (0, 0, 220), -1)  # red
    cv2.rectangle(img, (400, 300), (450, 350), (220, 60, 0), -1)  # blue
    return img


def test_ranges_select_cycle_and_flatten_back():
    ranges = ColorRanges(DEFAULT_COLORS)
    assert ranges.color == 'red' and len(ranges.ranges['red']) == 2
    ranges.next_range()
    assert ranges.current == DEFAULT_COLORS['red'][6:]
    ranges.next_range()
    assert ranges.index == 0
    assert ranges.flat() == DEFAULT_COLORS


def test_slider_max_below_min_is_clamped():
    ranges = ColorRanges(DEFAULT_COLORS)
    ranges.select('blue')
    ranges.set_current([100, 120, 50, 90, 255, 255])  # H max dragged below H min
    assert ranges.current == [100, 120, 50, 100, 255, 255]


def test_sample_and_range_around_a_blue_object():
    h, s, v = sample_hsv(_scene(), 425, 325)
    lo_h, _, _, hi_h, _, _ = range_around(h, s, v)
    assert lo_h <= h <= hi_h and hi_h - lo_h == 16


def test_right_click_near_the_red_wrap_sets_both_halves():
    ranges = ColorRanges(DEFAULT_COLORS)
    ranges.set_around(2, 200, 180)  # red sampled just above hue 0
    assert ranges.ranges['red'][0][3] == 10 and ranges.ranges['red'][1][0] == 174
    ranges.select('blue')
    ranges.set_around(110, 200, 180)  # one range, nowhere near the wrap
    assert ranges.current[0] == 102 and ranges.current[3] == 118


def test_printed_block_is_valid_config_yaml():
    block = format_colors(DEFAULT_COLORS)
    assert yaml.safe_load(block)['colors'] == DEFAULT_COLORS
    ColorDetector(yaml.safe_load(block)['colors'])  # and the detector accepts it


def test_render_marks_selected_colour_and_masks_it():
    ranges = ColorRanges(DEFAULT_COLORS)
    view, mask = render(_scene(), ranges, 300, 40000)
    assert mask[130, 130] == 255 and mask[325, 425] == 0  # red selected: only red is white
    assert view.shape == (480, 640, 3)
