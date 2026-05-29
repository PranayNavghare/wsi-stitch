import numpy as np

from stitch_engine.patch_folder import _finalize_uint8_average


def test_finalize_uint8_average_rounds_overlap() -> None:
    values = np.array([[[10, 20, 30], [11, 21, 31]]], dtype=np.uint16)
    weights = np.array([[[2], [0]]], dtype=np.uint8)

    averaged = _finalize_uint8_average(values, weights)

    assert np.array_equal(averaged, np.array([[[5, 10, 15], [0, 0, 0]]], dtype=np.uint8))
