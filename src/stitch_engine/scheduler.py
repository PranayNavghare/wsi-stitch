from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


IntPair = tuple[int, int]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Tile:
    """One read window and its corresponding output placement."""

    index: int
    input_box: Box
    output_box: Box

    @property
    def output_width(self) -> int:
        return self.output_box[2] - self.output_box[0]

    @property
    def output_height(self) -> int:
        return self.output_box[3] - self.output_box[1]

    @property
    def input_width(self) -> int:
        return self.input_box[2] - self.input_box[0]

    @property
    def input_height(self) -> int:
        return self.input_box[3] - self.input_box[1]


@dataclass(frozen=True)
class GridSpec:
    """Fixed-stride grid settings in x/y coordinate order."""

    image_shape: IntPair
    patch_input_shape: IntPair
    patch_output_shape: IntPair
    stride_shape: IntPair
    include_partial: bool = True
    keep_input_in_bounds: bool = False
    keep_output_in_bounds: bool = False


def plan_grid(spec: GridSpec) -> list[Tile]:
    """Plan a fixed-stride grid with explicit input and output boxes."""

    xs = _axis_starts(
        length=spec.image_shape[0],
        window=spec.patch_output_shape[0],
        stride=spec.stride_shape[0],
        include_partial=spec.include_partial,
    )
    ys = _axis_starts(
        length=spec.image_shape[1],
        window=spec.patch_output_shape[1],
        stride=spec.stride_shape[1],
        include_partial=spec.include_partial,
    )
    return plan_variable_grid(
        image_shape=spec.image_shape,
        patch_input_shape=spec.patch_input_shape,
        patch_output_shape=spec.patch_output_shape,
        output_xs=xs,
        output_ys=ys,
        keep_input_in_bounds=spec.keep_input_in_bounds,
        keep_output_in_bounds=spec.keep_output_in_bounds,
    )


def plan_variable_grid(
    *,
    image_shape: IntPair,
    patch_input_shape: IntPair,
    patch_output_shape: IntPair,
    output_xs: Sequence[int],
    output_ys: Sequence[int],
    keep_input_in_bounds: bool = False,
    keep_output_in_bounds: bool = False,
) -> list[Tile]:
    """Plan tiles from explicit output starts, enabling variable overlap."""

    _validate_shape("image_shape", image_shape)
    _validate_shape("patch_input_shape", patch_input_shape)
    _validate_shape("patch_output_shape", patch_output_shape)
    if patch_input_shape[0] < patch_output_shape[0] or patch_input_shape[1] < patch_output_shape[1]:
        raise ValueError("patch_input_shape must be greater than or equal to patch_output_shape.")

    x_margin = (patch_input_shape[0] - patch_output_shape[0]) // 2
    y_margin = (patch_input_shape[1] - patch_output_shape[1]) // 2

    tiles: list[Tile] = []
    index = 0
    for y in _unique_sorted_ints(output_ys):
        for x in _unique_sorted_ints(output_xs):
            output_box = (x, y, x + patch_output_shape[0], y + patch_output_shape[1])
            input_box = (
                x - x_margin,
                y - y_margin,
                x - x_margin + patch_input_shape[0],
                y - y_margin + patch_input_shape[1],
            )
            if keep_output_in_bounds and not _within_bounds(output_box, image_shape):
                continue
            if keep_input_in_bounds and not _within_bounds(input_box, image_shape):
                continue
            tiles.append(Tile(index=index, input_box=input_box, output_box=output_box))
            index += 1
    return tiles


def variable_starts_from_strides(
    *,
    length: int,
    window: int,
    strides: Iterable[int],
    include_partial: bool = True,
) -> list[int]:
    """Build one axis of starts from a stride sequence."""

    _validate_positive("length", length)
    _validate_positive("window", window)
    starts = [0]
    current = 0
    for stride in strides:
        _validate_positive("stride", stride)
        current += stride
        if current + window > length:
            if include_partial and starts[-1] != max(length - window, 0):
                starts.append(max(length - window, 0))
            break
        starts.append(current)
    return starts


def _axis_starts(*, length: int, window: int, stride: int, include_partial: bool) -> list[int]:
    _validate_positive("length", length)
    _validate_positive("window", window)
    _validate_positive("stride", stride)
    if length <= window:
        return [0]

    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if include_partial and starts[-1] != last:
        starts.append(last)
    return starts


def _within_bounds(box: Box, image_shape: IntPair) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= image_shape[0] and box[3] <= image_shape[1]


def _unique_sorted_ints(values: Sequence[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _validate_shape(name: str, shape: IntPair) -> None:
    if len(shape) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    _validate_positive(f"{name}[0]", shape[0])
    _validate_positive(f"{name}[1]", shape[1])


def _validate_positive(name: str, value: int) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be positive.")
