from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from stitch_engine.scheduler import Box


class StitchCanvas:
    """Accumulate blocks into a canvas and normalize overlaps."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        *,
        dtype: np.dtype = np.float32,
        work_dir: str | Path | None = None,
    ) -> None:
        if len(shape) != 3:
            raise ValueError("shape must be (height, width, channels).")
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if work_dir is None:
            self.values = np.zeros(shape, dtype=self.dtype)
            self.weights = np.zeros((*shape[:2], 1), dtype=self.dtype)
        else:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            self.values = np.memmap(
                Path(work_dir) / "stitch_values.dat",
                dtype=self.dtype,
                mode="w+",
                shape=shape,
            )
            self.weights = np.memmap(
                Path(work_dir) / "stitch_weights.dat",
                dtype=self.dtype,
                mode="w+",
                shape=(*shape[:2], 1),
            )
            self.values[:] = 0
            self.weights[:] = 0

    def add(self, block: np.ndarray, output_box: Box, *, weight: np.ndarray | float = 1.0) -> None:
        """Add a block at an output box, clipping at canvas boundaries."""

        if block.ndim != 3:
            raise ValueError("block must have shape (height, width, channels).")
        x0, y0, x1, y1 = output_box
        cx0 = max(x0, 0)
        cy0 = max(y0, 0)
        cx1 = min(x1, self.values.shape[1])
        cy1 = min(y1, self.values.shape[0])
        if cx0 >= cx1 or cy0 >= cy1:
            return

        bx0 = cx0 - x0
        by0 = cy0 - y0
        bx1 = bx0 + (cx1 - cx0)
        by1 = by0 + (cy1 - cy0)
        block_view = block[by0:by1, bx0:bx1, :]
        weight_view = _weight_view(weight, block_view.shape[:2])

        self.values[cy0:cy1, cx0:cx1, :] += block_view * weight_view
        self.weights[cy0:cy1, cx0:cx1, :] += weight_view

    def finalize(self, *, fill_value: float = 0.0) -> np.ndarray:
        """Return the normalized stitched output."""

        output = np.full_like(self.values, fill_value, dtype=np.float32)
        np.divide(
            self.values,
            self.weights,
            out=output,
            where=self.weights > 0,
        )
        return output

    def write_tiff(
        self,
        path: str | Path,
        *,
        chunk_rows: int = 1024,
        fill_value: float = 0.0,
        tiff_kwargs: dict | None = None,
    ) -> None:
        """Write the normalized canvas to a BigTIFF without materializing all output."""

        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("Install with the image extra to write TIFF output: pip install -e .[image]") from exc

        output = tifffile.memmap(
            path,
            shape=self.shape,
            dtype=np.uint8,
            photometric="rgb" if self.shape[2] in {3, 4} else "minisblack",
            bigtiff=True,
            **(tiff_kwargs or {}),
        )
        for y0 in range(0, self.shape[0], chunk_rows):
            y1 = min(y0 + chunk_rows, self.shape[0])
            chunk = np.full((y1 - y0, self.shape[1], self.shape[2]), fill_value, dtype=np.float32)
            weights = self.weights[y0:y1]
            np.divide(
                self.values[y0:y1],
                weights,
                out=chunk,
                where=weights > 0,
            )
            output[y0:y1] = np.clip(chunk, 0, 255).astype(np.uint8)
            print(f"  wrote rows {y0:,}-{y1:,} / {self.shape[0]:,}", flush=True)
        output.flush()


def stitch_blocks(
    blocks: list[np.ndarray] | np.ndarray,
    output_boxes: list[Box] | np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Convenience function for stitching blocks in one call."""

    canvas = StitchCanvas(shape)
    for block, output_box in zip(blocks, output_boxes, strict=True):
        canvas.add(np.asarray(block), tuple(int(v) for v in output_box))
    return canvas.finalize()


def _weight_view(weight: np.ndarray | float, shape: tuple[int, int]) -> np.ndarray:
    if np.isscalar(weight):
        return np.full((*shape, 1), float(weight), dtype=np.float32)
    weight_arr = np.asarray(weight, dtype=np.float32)
    if weight_arr.shape == shape:
        return weight_arr[:, :, None]
    if weight_arr.shape == (*shape, 1):
        return weight_arr
    raise ValueError("weight must be scalar, (height, width), or (height, width, 1).")
