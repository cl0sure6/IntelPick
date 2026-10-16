"""YOLO adapter contract tests without downloading weights or importing PyTorch."""

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from intelpick.detectors.color import DEFAULT_COLORS, ColorDetector
from intelpick.detectors.yolo import YoloDetector


def box(xyxy=(20, 20, 80, 80), confidence=0.9, cls=0):
    return SimpleNamespace(xyxy=np.array([xyxy]), conf=np.array([confidence]), cls=np.array([cls]))


@pytest.fixture
def model(monkeypatch, tmp_path):
    result = SimpleNamespace(names={0: 'cube'}, boxes=[box()])
    calls = []

    class FakeYolo:
        task = 'detect'

        def __init__(self, path):
            calls.append(path)

        def predict(self, image, **kwargs):
            calls.append(kwargs)
            return [result]

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=FakeYolo))
    weights = tmp_path / 'best.pt'
    weights.touch()
    return weights, result, calls


def red_image():
    img = np.full((100, 100, 3), 180, np.uint8)
    img[20:80, 20:80] = (0, 0, 220)
    return img


def test_trained_color_labels_go_directly_to_bins(model):
    weights, result, calls = model
    result.names = {0: 'red'}
    detector = YoloDetector(weights, label_mode='class')
    blob, = detector.detect(red_image())
    assert blob.label == 'red'
    assert (blob.u, blob.v, blob.area) == (50, 50, 3600)
    assert blob.confidence == pytest.approx(0.9)
    assert calls[-1] == dict(conf=0.5, device='cpu', imgsz=640, verbose=False)


@pytest.mark.parametrize('mode,label', [('color', 'red'), ('color_class', 'red_cube')])
def test_hsv_modes_have_explicit_sort_labels(model, mode, label):
    weights, _, _ = model
    detector = YoloDetector(weights, color_tagger=ColorDetector(DEFAULT_COLORS), label_mode=mode)
    assert detector.detect(red_image())[0].label == label


def test_unknown_or_ambiguous_color_is_skipped(model):
    weights, _, _ = model
    detector = YoloDetector(weights, color_tagger=ColorDetector(DEFAULT_COLORS))
    assert detector.detect(np.full((100, 100, 3), 180, np.uint8)) == []
    image = red_image()
    image[20:80, 50:80] = (220, 0, 0)
    assert detector.detect(image) == []


def test_boxes_are_clipped_before_cropping_and_mapping(model):
    weights, result, _ = model
    result.boxes = [box((-20.5, -10, 80.5, 80)), box((90, 20, 130, 80))]
    blobs = YoloDetector(weights).detect(red_image())
    assert (blobs[0].u, blobs[0].v) == (40.25, 40)
    assert (blobs[1].u, blobs[1].v) == (95, 50)


def test_bad_small_large_or_low_confidence_boxes_are_skipped(model):
    weights, result, _ = model
    result.boxes = [box((80, 80, 20, 20)), box((0, 0, 2, 2)),
                    box((float('nan'), 0, 80, 80)), box(confidence=float('nan')),
                    box(confidence=0.3), box((110, 110, 150, 150)), box((0, 0, 100, 100))]
    assert YoloDetector(weights, max_area=5000).detect(red_image()) == []
    result.boxes = []
    assert YoloDetector(weights).detect(red_image()) == []


def test_missing_weights_fail_without_download(tmp_path):
    with pytest.raises(ValueError, match='weights not found'):
        YoloDetector(tmp_path / 'missing.pt')


def test_classifier_weights_fail(model, monkeypatch):
    weights, _, _ = model
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(
        YOLO=lambda _: SimpleNamespace(task='classify')))
    with pytest.raises(ValueError, match='task=detect'):
        YoloDetector(weights)


@pytest.mark.parametrize('values', [[180, 0, 0, 190, 255, 255],
                                    [10, 0, 0, 0, 255, 255],
                                    [-1, 0, 0, 10, 255, 255],
                                    [0.5, 0, 0, 10, 255, 255]])
def test_bad_hsv_ranges_are_rejected(values):
    with pytest.raises(ValueError, match='HSV bounds'):
        ColorDetector({'red': values})


def test_red_wraparound_and_empty_crop():
    colors = ColorDetector(DEFAULT_COLORS)
    hsv = np.full((40, 40, 3), (175, 200, 200), np.uint8)
    assert colors.dominant_color(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)) == 'red'
    assert colors.dominant_color(np.empty((0, 0, 3), np.uint8)) is None
