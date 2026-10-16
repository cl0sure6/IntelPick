from types import SimpleNamespace

import pytest

from intelpick.tracking import StableTargets


def detection(label='red', x=0.25, y=0.0, confidence=0.9):
    return SimpleNamespace(label=label, x=x, y=y, confidence=confidence)


def test_target_requires_consecutive_frames_and_matches_label_and_position():
    tracker = StableTargets(['red', 'blue'], stable_frames=3)
    for now in (0.0, 0.1):
        tracker.update([detection()], now)
        assert tracker.target(now) is None
    tracker.update([detection(x=0.254)], 0.2)
    assert tracker.target(0.2).x == 0.254
    tracker.update([detection(label='blue')], 0.3)
    assert tracker.target(0.3) is None


def test_empty_frame_or_camera_gap_breaks_stability():
    tracker = StableTargets(['red'], stable_frames=2, max_age=0.5)
    tracker.update([detection()], 0.0)
    tracker.update([], 0.1)
    tracker.update([detection()], 0.2)
    assert tracker.target(0.2) is None
    tracker.update([detection()], 1.0)
    assert tracker.target(1.0) is None
    tracker.update([detection()], 1.1)
    assert tracker.target(1.1) is not None
    assert tracker.target(1.7) is None  # camera stopped after a previously stable target


def test_closest_stable_target_wins_and_reset_discards_history():
    tracker = StableTargets(['red'], stable_frames=2)
    for now in (0.0, 0.1):
        tracker.update([detection(x=0.32), detection(x=0.20)], now)
    assert tracker.target(0.1).x == 0.20
    tracker.clear()
    assert tracker.target(0.1) is None


@pytest.mark.parametrize('d', [detection(label='red_cube'), detection(x=float('nan')),
                             detection(y=float('inf')), detection(confidence=float('nan')),
                             detection(confidence=0.1), detection(x=0.15), detection(x=0.41)])
def test_unknown_nonfinite_uncertain_and_unreachable_targets_are_rejected(d):
    assert not StableTargets(['red']).pickable(d)


def test_grasp_offset_cannot_take_the_jaws_outside_reach():
    tracker = StableTargets(['red'], offset=0.02)
    assert not tracker.pickable(detection(x=0.39))
    assert tracker.pickable(detection(x=0.30))


@pytest.mark.parametrize('kwargs', [{'stable_frames': 0}, {'max_age': 0},
                                  {'stable_tol': -0.1}, {'approach': 'unknown'}])
def test_invalid_tracking_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        StableTargets(['red'], **kwargs)
