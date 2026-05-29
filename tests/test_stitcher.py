import numpy as np

from stitch_engine import StitchCanvas, stitch_blocks


def test_stitch_blocks_averages_overlap() -> None:
    blocks = np.array(
        [
            np.ones((2, 2, 1), dtype=np.float32),
            np.ones((2, 2, 1), dtype=np.float32) * 3,
        ]
    )

    stitched = stitch_blocks(
        blocks,
        [(0, 0, 2, 2), (1, 0, 3, 2)],
        shape=(2, 3, 1),
    )

    assert np.array_equal(stitched[:, :, 0], np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32))


def test_canvas_clips_blocks_at_edges() -> None:
    canvas = StitchCanvas((2, 2, 1))
    canvas.add(np.ones((3, 3, 1), dtype=np.float32), (-1, -1, 2, 2))

    assert np.array_equal(canvas.finalize(), np.ones((2, 2, 1), dtype=np.float32))


def test_canvas_accepts_weight_map() -> None:
    canvas = StitchCanvas((1, 2, 1))
    canvas.add(np.array([[[2], [4]]], dtype=np.float32), (0, 0, 2, 1), weight=np.array([[1, 2]]))

    assert np.array_equal(canvas.finalize(), np.array([[[2], [4]]], dtype=np.float32))
