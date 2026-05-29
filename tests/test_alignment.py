import numpy as np

from stitch_engine.alignment import _search_translation


def test_search_translation_recovers_known_offset() -> None:
    image_a = np.zeros((64, 64), dtype=np.float32)
    image_a[12:48, 20:58] = np.arange(36 * 38, dtype=np.float32).reshape(36, 38)

    dx = 30
    dy = 5
    image_b = np.zeros_like(image_a)
    ax0 = dx
    ay0 = dy
    ax1 = 64
    ay1 = 64
    image_b[0 : ay1 - ay0, 0 : ax1 - ax0] = image_a[ay0:ay1, ax0:ax1]

    result = _search_translation(
        image_a,
        image_b,
        expected_dx=32,
        expected_dy=0,
        jitter=8,
        min_overlap=8,
    )

    assert result.dx == dx
    assert result.dy == dy
    assert result.score > 0.9
