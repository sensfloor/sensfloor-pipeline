import numpy as np

from data_loading.roi_floor import RoIFloor


def test_returns_no_roi_when_no_update():
    floor = RoIFloor(x=5, y=5, history_maxlen=1, roi_size=3)
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
    pass
