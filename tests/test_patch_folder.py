from stitch_engine.patch_folder import _build_chunk_index


def test_build_chunk_index_maps_overlapping_boxes() -> None:
    boxes = [
        (0, 0, 1024, 1024),
        (900, 0, 1924, 1024),
        (0, 900, 1024, 1924),
    ]

    index = _build_chunk_index(boxes, 1024)

    assert index[(0, 0)] == {0, 1, 2}
    assert index[(0, 1)] == {1}
    assert index[(1, 0)] == {2}
