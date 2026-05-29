from stitch_engine.patch_folder import _cell_boxes_from_base_grid


def test_cell_boxes_from_base_grid() -> None:
    boxes = [
        (0, 0, 10, 10),
        (5, 0, 15, 10),
        (0, 5, 10, 15),
        (5, 5, 15, 15),
    ]

    cells = _cell_boxes_from_base_grid(boxes)

    assert cells == [
        (0, 0, 8, 8),
        (8, 0, 15, 8),
        (0, 8, 8, 15),
        (8, 8, 15, 15),
    ]
