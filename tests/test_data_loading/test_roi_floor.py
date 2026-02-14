import numpy as np
import pytest

from src.data_loading.roi_floor import RoIFloor, RoIFloorConfig


def test_returns_no_roi_when_no_update():
    config = RoIFloorConfig(x_size=5, y_size=5, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    roi = floor.get_roi()
    assert roi is None


def test_returns_roi_in_the_middle_of_floor():
    config = RoIFloorConfig(x_size=5, y_size=5, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[2, 2]])
    signals = np.array([[180, 220, 200, 180, 200, 200, 200, 210]])
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[0, 4:8, 4:8] = np.array(
        [
            [200, 200, 200, 205],
            [200, 200, 205, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    expected_roi_x = 1
    expected_roi_y = 1

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == expected_roi_x
    assert roi is not None
    assert roi.y == expected_roi_y
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_returns_roi_with_higher_signal_values():
    # Arrange floor
    config = RoIFloorConfig(x_size=5, y_size=5, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[1, 1], [3, 3]])
    signals = np.array(
        [
            [180, 220, 200, 180, 200, 200, 200, 210],
            [180, 220, 200, 180, 200, 200, 220, 210],
        ],
    )
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[0, 4:8, 4:8] = np.array(
        [
            [200, 200, 220, 215],
            [200, 200, 215, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    expected_roi_x = 2
    expected_roi_y = 2

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == expected_roi_x
    assert roi is not None
    assert roi.y == expected_roi_y
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_x_0():
    # Arrange floor
    config = RoIFloorConfig(x_size=5, y_size=6, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[0, 1]])
    signals = np.array([[200, 200, 200, 200, 200, 200, 200, 200]])
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[0, 0:4, 4:8] = np.ones((1, 4, 4)) * 200

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == 0
    assert roi is not None
    assert roi.y == 0
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_y_0():
    # Arrange floor
    config = RoIFloorConfig(x_size=5, y_size=6, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[1, 0]])
    signals = np.array([[200, 200, 200, 200, 200, 200, 200, 200]])
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[0, 4:8, 0:4] = np.ones((1, 4, 4)) * 200

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == 0
    assert roi is not None
    assert roi.y == 0
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_x_max():
    # Arrange floor
    config = RoIFloorConfig(x_size=5, y_size=6, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[4, 1]])
    signals = np.array([[200, 200, 200, 200, 200, 200, 200, 200]])
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[0, 8:12, 4:8] = np.ones((1, 4, 4)) * 200
    expected_roi_x = 2
    expected_roi_y = 0

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == expected_roi_x
    assert roi is not None
    assert roi.y == expected_roi_y
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_y_max():
    # Arrange floor
    config = RoIFloorConfig(x_size=5, y_size=6, history_maxlen=1, roi_size=3)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[1, 5]])
    signals = np.array([[200, 200, 200, 200, 200, 200, 200, 200]])
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[0, 4:8, 8:12] = np.ones((1, 4, 4)) * 200
    expected_roi_x = 0
    expected_roi_y = 3

    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == expected_roi_x
    assert roi is not None
    assert roi.y == expected_roi_y
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_roi_size_5():
    # Arrange floor
    config = RoIFloorConfig(x_size=5, y_size=10, history_maxlen=2, roi_size=5)
    floor = RoIFloor(config)

    assert config.roi_size is not None

    # Arrange region of interest
    positions = np.array([[1, 2], [3, 9]])
    signals = np.array(
        [
            [200, 100, 100, 100, 100, 100, 100, 100],
            [200, 200, 200, 200, 200, 200, 200, 200],
        ],
    )
    expected_roi_history = np.ones((config.history_maxlen, config.roi_size * 4, config.roi_size * 4)) * 127
    expected_roi_history[-1, 12:16, 16:20] = np.ones((1, 4, 4)) * 200
    expected_roi_x = 0
    expected_roi_y = 5
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None
    assert roi.x == expected_roi_x

    assert roi is not None
    assert roi.y == expected_roi_y

    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_roi_too_large_for_floor():
    config = RoIFloorConfig(x_size=2, y_size=2, history_maxlen=1, roi_size=3)
    with pytest.raises(ValueError, match="RoI too large for floor size"):
        RoIFloor(config)
