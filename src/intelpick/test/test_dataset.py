from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from intelpick.dataset import check_rows, init_dataset, save_sample, validate_dataset
from intelpick.train_yolo import training_options


def populated_dataset(root):
    config = init_dataset(root)
    for split, brightness in [('train', 120), ('val', 180)]:
        frame = np.full((100, 200, 3), brightness, np.uint8)
        annotations = [(cls, cls * 40, 10, 30, 40) for cls in range(4)]
        save_sample(root, split, frame, annotations)
    return config


def test_capture_writes_standard_yolo_layout_and_normalized_boxes(tmp_path):
    config = populated_dataset(tmp_path)
    summary = validate_dataset(config)
    assert summary['train'] == {'images': 1, 'objects': dict(red=1, green=1, blue=1, yellow=1)}
    label, = (tmp_path / 'labels/train').glob('*.txt')
    first = check_rows(label.read_text(), 4, label)[0]
    assert first == pytest.approx((0, 0.075, 0.3, 0.15, 0.4))
    data = yaml.safe_load(config.read_text())
    assert data['path'] == str(tmp_path.resolve())


def test_restarting_capture_preserves_config_and_rejects_changed_classes(tmp_path):
    config = init_dataset(tmp_path)
    original = config.read_bytes()
    assert init_dataset(tmp_path) == config
    assert config.read_bytes() == original
    with pytest.raises(ValueError, match='different settings'):
        init_dataset(tmp_path, ['red', 'blue'])


def test_missing_annotation_is_not_treated_as_background(tmp_path):
    config = populated_dataset(tmp_path)
    label, = (tmp_path / 'labels/train').glob('*.txt')
    label.unlink()
    with pytest.raises(ValueError, match='missing annotation'):
        validate_dataset(config)


def test_background_is_explicit_and_does_not_replace_positive_examples(tmp_path):
    config = populated_dataset(tmp_path)
    image = save_sample(tmp_path, 'train', np.zeros((100, 200, 3), np.uint8), [])
    assert (tmp_path / 'labels/train' / f'{image.stem}.txt').read_text() == ''
    assert validate_dataset(config)['train']['images'] == 2


def test_training_validation_leakage_is_rejected(tmp_path):
    config = populated_dataset(tmp_path)
    image, = (tmp_path / 'images/train').glob('*.jpg')
    save_sample(tmp_path, 'val', cv2.imread(str(image)), [])
    # Copy exact bytes: JPEG recompression can change pixels on nonuniform scenes.
    # Identify the added background sample, independent of UUID sorting.
    target = next(p for p in (tmp_path / 'images/val').glob('*.jpg')
                  if not (tmp_path / 'labels/val' / f'{p.stem}.txt').read_text())
    target.write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match='leakage'):
        validate_dataset(config)


@pytest.mark.parametrize('row', ['4 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2',
                               '0 0.01 0.5 0.2 0.2', '0 0.5 0.5 0 0.2',
                               '0 0.5 0.5', '0.5 0.5 0.5 0.2 0.2'])
def test_bad_annotations_fail_before_training(row):
    with pytest.raises(ValueError):
        check_rows(row, 4, 'test label')


def test_incomplete_dataset_is_rejected(tmp_path):
    config = init_dataset(tmp_path)
    with pytest.raises(ValueError, match='no images'):
        validate_dataset(config)
    save_sample(tmp_path, 'train', np.zeros((100, 100, 3), np.uint8), [(0, 20, 20, 30, 30)])
    with pytest.raises(ValueError, match='no labelled examples'):
        validate_dataset(config)


def test_relative_dataset_root_cannot_resolve_differently_during_training(tmp_path):
    config = populated_dataset(tmp_path)
    data = yaml.safe_load(config.read_text())
    data['path'] = '.'
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='path must be absolute'):
        validate_dataset(config)


def test_training_preserves_color_semantics(tmp_path):
    args = SimpleNamespace(data=str(tmp_path / 'data.yaml'), epochs=80, imgsz=640, batch=8,
                           device='cpu', project=str(tmp_path), name='train', workers=2)
    options = training_options(args)
    assert options['hsv_h'] == options['hsv_s'] == options['bgr'] == 0.0
    assert options['hsv_v'] > 0
