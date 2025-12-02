import numpy as np

from data_loading.floor import Floor, interpolate_signal


def test_has_correct_shape():
    floor = Floor(10, 8, history_maxlen=1)
    assert floor.shape == (10, 8)


def test_update_single_patch():
    floor = Floor(1, 1, history_maxlen=1)
    positions = [(0, 0)]
    signals = np.array([[180, 120, 100, 180, 100, 100, 100, 200]])
    expected_patches = np.array(
        [
            [150, 200, 180, 150],
            [100, 150, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )
    floor.update(positions, signals)
    np.testing.assert_array_equal(floor.patches, expected_patches)


def test_update_multiple_patches():
    x_size = 5
    y_size = 5
    floor = Floor(x_size, y_size, history_maxlen=1)
    positions = [(2, 2), (4, 4)]
    signals = np.array(
        [
            [180, 120, 100, 180, 100, 100, 100, 200],
            [200, 100, 100, 180, 800, 100, 1000, 200],
        ]
    )
    expected_patches = np.zeros((x_size * 4, y_size * 4))
    expected_patches[8:12, 8:12] = np.array(
        [
            [150, 200, 180, 150],
            [100, 150, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )
    expected_patches[16:20, 16:20] = np.array(
        [
            [600, 200, 200, 150],
            [1000, 600, 150, 100],
            [100, 450, 140, 100],
            [450, 800, 180, 140],
        ]
    )
    floor.update(positions, signals)
    np.testing.assert_array_equal(floor.patches, expected_patches)


def test_floor_history():
    x_size = 5
    y_size = 5
    floor = Floor(x_size, y_size, history_maxlen=2)
    # Update 1
    positions1 = [(0, 2)]
    signals1 = np.array([[180, 120, 100, 180, 100, 100, 100, 200]])
    floor.update(positions1, signals1)
    expected_patches1 = np.zeros((x_size * 4, y_size * 4))
    expected_patches1[0:4, 8:12] = np.array(
        [
            [150, 200, 180, 150],
            [100, 150, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )
    np.testing.assert_array_equal(floor.patches, expected_patches1)

    # Update 2
    positions2 = [(0, 2)]
    signals2 = np.array([[200, 100, 100, 180, 800, 100, 1000, 200]])
    expected_patches2 = np.zeros((x_size * 4, y_size * 4))
    expected_patches2[0:4, 8:12] = np.array(
        [
            [600, 200, 200, 150],
            [1000, 600, 150, 100],
            [100, 450, 140, 100],
            [450, 800, 180, 140],
        ]
    )
    floor.update(positions2, signals2)
    np.testing.assert_array_equal(floor.patches, expected_patches2)

    history = floor.history
    expected_history = np.stack([expected_patches1, expected_patches2])
    np.testing.assert_array_equal(history, expected_history)


def test_floor_history_rotates():
    x_size = 5
    y_size = 5
    floor = Floor(x_size, y_size, history_maxlen=1)
    # Update 1
    positions1 = [(0, 2)]
    signals1 = np.array([[180, 120, 100, 180, 100, 100, 100, 200]])
    floor.update(positions1, signals1)
    expected_history1 = np.zeros((1, x_size * 4, y_size * 4))
    expected_history1[0, 0:4, 8:12] = np.array(
        [
            [150, 200, 180, 150],
            [100, 150, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )
    np.testing.assert_array_equal(floor.history, expected_history1)

    # Update 2
    positions2 = [(0, 2)]
    signals2 = np.array([[200, 100, 100, 180, 800, 100, 1000, 200]])
    expected_history2 = np.zeros((1, x_size * 4, y_size * 4))
    expected_history2[0, 0:4, 8:12] = np.array(
        [
            [600, 200, 200, 150],
            [1000, 600, 150, 100],
            [100, 450, 140, 100],
            [450, 800, 180, 140],
        ]
    )
    floor.update(positions2, signals2)
    np.testing.assert_array_equal(floor.history, expected_history2)


def test_history_returns_always_same_length():
    x_size = 20
    y_size = 20
    floor = Floor(x_size, y_size, history_maxlen=10)
    history = floor.history
    expected_history = np.zeros((10, 20 * 4, 20 * 4))
    np.testing.assert_array_equal(history, expected_history)


def test_signal_interpolation():
    signal = np.array([180, 120, 100, 180, 100, 100, 100, 200])
    expected_signal = np.array(
        [
            [150, 200, 180, 150],
            [100, 150, 150, 120],
            [100, 100, 140, 100],
            [100, 100, 180, 140],
        ]
    )
    interpolated_signal = interpolate_signal(signal)
    np.testing.assert_array_equal(expected_signal, interpolated_signal)
