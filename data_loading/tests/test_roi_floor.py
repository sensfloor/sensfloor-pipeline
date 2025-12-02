import numpy as np
import pytest

from data_loading.roi_floor import RoIFloor


def test_returns_no_roi_when_no_update():
    floor = RoIFloor(x_size=5, y_size=5, history_maxlen=1, roi_size=3)
    roi = floor.get_roi()
    assert roi is None


def test_returns_roi_in_the_middle_of_floor():
    # Arrange floor
    x_size = 5
    y_size = 5
    history_maxlen = 1
    roi_size = 3
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(2, 2)]
    signals = np.array([[180, 120, 100, 180, 100, 100, 100, 200]])
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[0, 4:8, 4:8] = np.array(
        [
            [150, 200, 180, 150],
            [100, 150, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.x == 1
    assert roi is not None and roi.y == 1
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_returns_roi_with_higher_signal_values():
    # Arrange floor
    x_size = 5
    y_size = 5
    history_maxlen = 1
    roi_size = 3
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(1, 1), (3, 3)]
    signals = np.array(
        [
            [180, 120, 100, 180, 100, 100, 100, 200],
            [180, 120, 100, 180, 100, 100, 200, 200],
        ]
    )
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[0, 4:8, 4:8] = np.array(
        [
            [200, 200, 180, 150],
            [200, 200, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.x == 2
    assert roi is not None and roi.y == 2
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_x_0():
    # Arrange floor
    x_size = 5
    y_size = 6
    history_maxlen = 1
    roi_size = 3
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(0, 1)]
    signals = np.array([[100, 100, 100, 100, 100, 100, 100, 100]])
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[0, 0:4, 4:8] = np.ones((1, 4, 4)) * 100

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.x == 0
    assert roi is not None and roi.y == 0
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_y_0():
    # Arrange floor
    x_size = 5
    y_size = 6
    history_maxlen = 1
    roi_size = 3
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(1, 0)]
    signals = np.array([[100, 100, 100, 100, 100, 100, 100, 100]])
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[0, 4:8, 0:4] = np.ones((1, 4, 4)) * 100

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.x == 0
    assert roi is not None and roi.y == 0
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_x_maxlen():
    # Arrange floor
    x_size = 5
    y_size = 6
    history_maxlen = 1
    roi_size = 3
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(4, 1)]
    signals = np.array([[100, 100, 100, 100, 100, 100, 100, 100]])
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[0, 8:12, 4:8] = np.ones((1, 4, 4)) * 100

    # Update floor
    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.x == 2
    assert roi is not None and roi.y == 0
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_adjust_roi_to_fit_y_maxlen():
    # Arrange floor
    x_size = 5
    y_size = 6
    history_maxlen = 1
    roi_size = 3
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(1, 5)]
    signals = np.array([[100, 100, 100, 100, 100, 100, 100, 100]])
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[0, 4:8, 8:12] = np.ones((1, 4, 4)) * 100

    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.x == 0
    assert roi is not None and roi.y == 3
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_roi_size_5():
    # Arrange floor
    x_size = 5
    y_size = 10
    history_maxlen = 2
    roi_size = 5
    floor = RoIFloor(x_size, y_size, history_maxlen, roi_size)

    # Arrange region of interest
    positions = [(1, 2), (3, 9)]
    signals = np.array(
        [
            [200, 100, 100, 100, 100, 100, 100, 100],
            [200, 200, 200, 200, 200, 200, 200, 200],
        ]
    )
    expected_roi_history = np.zeros((history_maxlen, roi_size * 4, roi_size * 4))
    expected_roi_history[-1, 12:16, 16:20] = np.ones((1, 4, 4)) * 200

    floor.update(positions, signals)
    roi = floor.get_roi()

    assert roi is not None and roi.y == 5
    np.testing.assert_array_equal(roi.history, expected_roi_history)


def test_roi_too_large_for_floor():
    x_size = 2
    y_size = 2
    history_maxlen = 1
    roi_size = 3

    with pytest.raises(ValueError):
        RoIFloor(x_size, y_size, history_maxlen, roi_size)
