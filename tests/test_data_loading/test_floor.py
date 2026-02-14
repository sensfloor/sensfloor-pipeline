import numpy as np

from src.data_loading.floor import Floor, FloorConfig, interpolate_signal


def test_has_correct_shape():
    config = FloorConfig(x_size=10, y_size=8, history_maxlen=1)
    floor = Floor(config)
    assert floor.shape == (10, 8)


def test_update_single_patch():
    config = FloorConfig(x_size=1, y_size=1, history_maxlen=1)
    floor = Floor(config)

    positions = np.array([[0, 0]])
    signals = np.array([[180, 220, 200, 180, 200, 200, 200, 210]])
    expected_patches = np.array(
        [
            [200, 200, 200, 205],
            [200, 200, 205, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    floor.update(positions, signals)
    np.testing.assert_array_equal(floor.patches, expected_patches)


def test_update_multiple_patches():
    x_size = 5
    y_size = 5
    config = FloorConfig(x_size=x_size, y_size=y_size, history_maxlen=1)
    floor = Floor(config)

    positions = np.array([[2, 2], [4, 4]])
    signals = np.array(
        [
            [180, 220, 200, 180, 200, 200, 200, 210],
            [200, 200, 200, 180, 800, 200, 1000, 200],
        ],
    )
    expected_patches = np.ones((x_size * 4, y_size * 4)) * 127
    expected_patches[8:12, 8:12] = np.array(
        [
            [200, 200, 200, 205],
            [200, 200, 205, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    expected_patches[16:20, 16:20] = np.array(
        [
            [500, 200, 1000, 600],
            [800, 500, 600, 200],
            [180, 190, 200, 200],
            [190, 200, 200, 200],
        ],
    )
    floor.update(positions, signals)
    np.testing.assert_array_equal(floor.patches, expected_patches)


def test_floor_history():
    x_size = 5
    y_size = 5
    config = FloorConfig(x_size=x_size, y_size=y_size, history_maxlen=2)
    floor = Floor(config)

    # Update 1
    positions1 = np.array([[0, 2]])
    signals1 = np.array([[180, 220, 200, 180, 200, 200, 200, 210]])
    floor.update(positions1, signals1)
    expected_patches1 = np.ones((x_size * 4, y_size * 4)) * 127
    expected_patches1[0:4, 8:12] = np.array(
        [
            [200, 200, 200, 205],
            [200, 200, 205, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    np.testing.assert_array_equal(floor.patches, expected_patches1)

    # Update 2
    positions2 = np.array([[0, 2]])
    signals2 = np.array([[200, 200, 200, 180, 800, 200, 1000, 200]])
    expected_patches2 = np.ones((x_size * 4, y_size * 4)) * 127
    expected_patches2[0:4, 8:12] = np.array(
        [
            [500, 200, 1000, 600],
            [800, 500, 600, 200],
            [180, 190, 200, 200],
            [190, 200, 200, 200],
        ],
    )
    floor.update(positions2, signals2)
    np.testing.assert_array_equal(floor.patches, expected_patches2)

    history = floor.history
    expected_history = np.stack([expected_patches1, expected_patches2])
    np.testing.assert_array_equal(history, expected_history)


def test_floor_history_rotates_when_history_is_full():
    x_size = 5
    y_size = 5
    config = FloorConfig(x_size=x_size, y_size=y_size, history_maxlen=1)
    floor = Floor(config)

    # Update 1
    positions1 = np.array([[0, 2]])
    signals1 = np.array([[180, 220, 200, 180, 200, 200, 200, 210]])
    floor.update(positions1, signals1)
    expected_history1 = np.ones((1, x_size * 4, y_size * 4)) * 127
    expected_history1[0, 0:4, 8:12] = np.array(
        [
            [200, 200, 200, 205],
            [200, 200, 205, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    np.testing.assert_array_equal(floor.history, expected_history1)

    # Update 2
    positions2 = np.array([[0, 2]])
    signals2 = np.array([[200, 180, 170, 220, 240, 160, 180, 150]])
    expected_history2 = np.ones((1, x_size * 4, y_size * 4)) * 127
    expected_history2[0, 0:4, 8:12] = np.array(
        [
            [200, 160, 180, 165],
            [240, 200, 165, 150],
            [220, 195, 190, 200],
            [195, 170, 180, 190],
        ],
    )
    floor.update(positions2, signals2)
    np.testing.assert_array_equal(floor.history, expected_history2)


def test_history_returns_always_same_length():
    x_size = 20
    y_size = 20
    config = FloorConfig(x_size=x_size, y_size=y_size, history_maxlen=10)
    floor = Floor(config)

    history = floor.history
    expected_history = np.ones((10, 20 * 4, 20 * 4)) * 127
    np.testing.assert_array_equal(history, expected_history)


def test_signal_interpolation():
    signal = np.array([180, 220, 200, 180, 200, 200, 200, 210])
    expected_signal = np.array(
        [
            [200, 200, 200, 205],
            [200, 200, 205, 210],
            [180, 190, 200, 180],
            [190, 200, 220, 200],
        ],
    )
    interpolated_signal = interpolate_signal(signal)
    np.testing.assert_array_equal(expected_signal, interpolated_signal)


def test_signal_clipping():
    config = FloorConfig(x_size=1, y_size=1, history_maxlen=1)
    floor = Floor(config)

    positions = np.array([[0, 0]])
    signals = np.array([[137, 100, 200, 180, 147, 120, 167, 0]])
    expected_patches = np.array(
        [
            [137, 127, 167, 147],
            [147, 137, 147, 127],
            [180, 190, 127, 127],
            [190, 200, 127, 127],
        ],
    )
    floor.update(positions, signals)
    np.testing.assert_array_equal(floor.patches, expected_patches)


def test_remove_noise():
    config = FloorConfig(x_size=2, y_size=1, history_maxlen=1, remove_noise=True)
    floor = Floor(config)

    positions_first_patch = np.array([[0, 0]])
    positions_both_patches = np.array([[0, 0], [1, 0]])
    signals_first_patch = np.array([[137, 100, 200, 180, 147, 120, 167, 0]])
    signals_both_patches = np.array([[137, 100, 200, 180, 147, 120, 167, 0], [137, 100, 200, 180, 147, 120, 167, 0]])
    expected_patches = np.array(
        [
            [137, 127, 167, 147],
            [147, 137, 147, 127],
            [180, 190, 127, 127],
            [190, 200, 127, 127],
            # second patch
            [127, 127, 127, 127],
            [127, 127, 127, 127],
            [127, 127, 127, 127],
            [127, 127, 127, 127],
        ],
    )
    # Update both patches
    floor.update(positions_both_patches, signals_both_patches)

    # Update only first patch
    floor.update(positions_first_patch, signals_first_patch)
    floor.update(positions_first_patch, signals_first_patch)
    floor.update(positions_first_patch, signals_first_patch)
    floor.update(positions_first_patch, signals_first_patch)
    floor.update(positions_first_patch, signals_first_patch)
    floor.update(positions_first_patch, signals_first_patch)

    # Expect resetted patch
    np.testing.assert_array_equal(floor.patches, expected_patches)
