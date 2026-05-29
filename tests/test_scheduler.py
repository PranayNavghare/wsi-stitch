from stitch_engine import GridSpec, plan_grid, plan_variable_grid
from stitch_engine.scheduler import variable_starts_from_strides


def test_fixed_grid_plans_partial_edge_tiles() -> None:
    tiles = plan_grid(
        GridSpec(
            image_shape=(10, 8),
            patch_input_shape=(4, 4),
            patch_output_shape=(4, 4),
            stride_shape=(3, 3),
        )
    )

    assert [tile.output_box for tile in tiles] == [
        (0, 0, 4, 4),
        (3, 0, 7, 4),
        (6, 0, 10, 4),
        (0, 3, 4, 7),
        (3, 3, 7, 7),
        (6, 3, 10, 7),
        (0, 4, 4, 8),
        (3, 4, 7, 8),
        (6, 4, 10, 8),
    ]


def test_variable_grid_uses_explicit_starts() -> None:
    tiles = plan_variable_grid(
        image_shape=(20, 20),
        patch_input_shape=(6, 6),
        patch_output_shape=(4, 4),
        output_xs=[0, 3, 9],
        output_ys=[0, 2],
    )

    assert tiles[0].input_box == (-1, -1, 5, 5)
    assert tiles[1].output_box == (3, 0, 7, 4)
    assert tiles[-1].output_box == (9, 2, 13, 6)


def test_variable_starts_from_strides() -> None:
    starts = variable_starts_from_strides(length=20, window=5, strides=[4, 3, 6, 6])

    assert starts == [0, 4, 7, 13, 15]
