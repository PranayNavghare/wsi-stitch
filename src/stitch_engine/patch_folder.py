from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
import math
import re
from pathlib import Path
import tempfile
import time

import numpy as np

from stitch_engine.alignment import AlignmentConfig, align_boxes_from_images
from stitch_engine.stitcher import StitchCanvas


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
XY_PATTERN = re.compile(r"(?:^|_)x(?P<x>-?\d+)_(?:y)(?P<y>-?\d+)(?:_|\.|$)", re.IGNORECASE)
TIFF_COMPRESSION = "jpeg"
TIFF_JPEG_QUALITY = 90
TIFF_TILE_SIZE = 256


@dataclass(frozen=True)
class TiffMetadata:
    mpp_x: float
    mpp_y: float
    objective_power: float


def stitch_patch_folder(
    patch_dir: str | Path,
    output_path: str | Path,
    *,
    cols: int,
    stride_x: int | None = None,
    stride_y: int | None = None,
    overlap_x: int = 0,
    overlap_y: int = 0,
    overlap_percent: float | None = None,
    from_filenames: bool = False,
    progress_every: int = 100,
    mode: str = "average",
    jitter_percent: float = 10.0,
    align_downsample: int = 8,
    align_coarse_downsample: int = 16,
    align_refine_radius: int = 4,
    align_min_score: float = 0.12,
    align_max_pairs: int | None = None,
    align_solver_iterations: int = 200,
    align_base_weight: float = 0.05,
    align_matcher: str = "phase",
    align_diagnostics_dir: str | None = None,
    writer: str = "chunked",
    chunk_size: int = 2048,
    cell_padding: int = 32,
    mpp: float = 0.25,
    mpp_x: float | None = None,
    mpp_y: float | None = None,
    objective_power: float | None = None,
) -> Path:
    """Stitch sorted image patches as a row-major grid preview."""

    total_start = time.perf_counter()
    image_files = sorted(
        path for path in Path(patch_dir).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No patch images found in {patch_dir}.")
    if cols < 1:
        raise ValueError("cols must be positive.")
    if mode not in {"average", "direct", "align"}:
        raise ValueError("mode must be 'average', 'direct', or 'align'.")
    if writer not in {"chunked", "cell", "full"}:
        raise ValueError("writer must be 'chunked', 'cell', or 'full'.")
    print(f"Found {len(image_files):,} patches in {patch_dir}", flush=True)

    first_image = _read_image(image_files[0])
    patch_h, patch_w = first_image.shape[:2]
    channels = 1 if first_image.ndim == 2 else first_image.shape[2]
    print(f"Patch shape: {patch_w} x {patch_h}, channels={channels}", flush=True)
    tiff_metadata = _resolve_tiff_metadata(
        mpp=mpp,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        objective_power=objective_power,
    )
    print(
        "TIFF metadata: "
        f"mpp_x={tiff_metadata.mpp_x:.4f}, "
        f"mpp_y={tiff_metadata.mpp_y:.4f}, "
        f"objective={tiff_metadata.objective_power:g}x",
        flush=True,
    )

    if overlap_percent is not None:
        if overlap_percent < 0 or overlap_percent >= 100:
            raise ValueError("overlap_percent must be in [0, 100).")
        overlap_x = round(patch_w * overlap_percent / 100)
        overlap_y = round(patch_h * overlap_percent / 100)

    if from_filenames:
        boxes = [_box_from_filename(path, patch_w, patch_h) for path in image_files]
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        boxes = [
            (box[0] - min_x, box[1] - min_y, box[2] - min_x, box[3] - min_y)
            for box in boxes
        ]
    else:
        stride_x = patch_w - overlap_x if stride_x is None else stride_x
        stride_y = patch_h - overlap_y if stride_y is None else stride_y
        if stride_x < 1 or stride_y < 1:
            raise ValueError("stride must be positive. Check overlap values.")

        rows = math.ceil(len(image_files) / cols)
        canvas_w = stride_x * (cols - 1) + patch_w
        canvas_h = stride_y * (rows - 1) + patch_h
        boxes = []
        for index in range(len(image_files)):
            row, col = divmod(index, cols)
            x0 = col * stride_x
            y0 = row * stride_y
            boxes.append((x0, y0, x0 + patch_w, y0 + patch_h))
    base_boxes = list(boxes)

    if mode == "align":
        if not from_filenames:
            raise ValueError("align mode currently requires --from-filenames.")
        boxes = align_boxes_from_images(
            image_files,
            boxes,
            patch_shape=(patch_h, patch_w),
            config=AlignmentConfig(
                base_overlap_percent=overlap_percent if overlap_percent is not None else 50.0,
                jitter_percent=jitter_percent,
                downsample=align_downsample,
                coarse_downsample=align_coarse_downsample,
                refine_radius=align_refine_radius,
                min_score=align_min_score,
                max_pairs=align_max_pairs,
                solver_iterations=align_solver_iterations,
                base_weight=align_base_weight,
                matcher=align_matcher,
                diagnostics_dir=align_diagnostics_dir,
            ),
            progress_every=progress_every,
        )


    canvas_w = max(box[2] for box in boxes)
    canvas_h = max(box[3] for box in boxes)

    if mode == "direct":
        if not from_filenames:
            raise ValueError("direct mode currently requires --from-filenames.")
        print(f"Canvas: {canvas_w:,} x {canvas_h:,}", flush=True)
        print("Fast direct mode: writing one non-overlapping grid cell per patch.", flush=True)
        _write_direct_grid_tiff(
            image_files=image_files,
            boxes=boxes,
            output_path=output_path,
            shape=(canvas_h, canvas_w, channels),
            patch_shape=(patch_h, patch_w),
            cell_padding=cell_padding,
            tiff_metadata=tiff_metadata,
            progress_every=progress_every,
        )
        print(f"Total time: {_format_seconds(time.perf_counter() - total_start)}", flush=True)
        print(f"Done: {output_path}", flush=True)
        return Path(output_path)

    if writer == "cell":
        print(f"Canvas: {canvas_w:,} x {canvas_h:,}", flush=True)
        print("Cell writer: using corrected positions without overlap averaging.", flush=True)
        _write_cell_tiff(
            image_files=image_files,
            boxes=boxes,
            base_boxes=base_boxes,
            output_path=output_path,
            shape=(canvas_h, canvas_w, channels),
            patch_shape=(patch_h, patch_w),
            cell_padding=cell_padding,
            tiff_metadata=tiff_metadata,
            progress_every=progress_every,
        )
        print(f"Total time: {_format_seconds(time.perf_counter() - total_start)}", flush=True)
        print(f"Done: {output_path}", flush=True)
        return Path(output_path)

    if writer == "chunked":
        print(f"Canvas: {canvas_w:,} x {canvas_h:,}", flush=True)
        print(f"Chunked accurate writer: chunk_size={chunk_size:,}", flush=True)
        _write_chunked_average_tiff(
            image_files=image_files,
            boxes=boxes,
            output_path=output_path,
            shape=(canvas_h, canvas_w, channels),
            patch_shape=(patch_h, patch_w),
            chunk_size=chunk_size,
            tiff_metadata=tiff_metadata,
            progress_every=progress_every,
        )
        print(f"Total time: {_format_seconds(time.perf_counter() - total_start)}", flush=True)
        print(f"Done: {output_path}", flush=True)
        return Path(output_path)

    with tempfile.TemporaryDirectory(prefix="stitch_engine_") as work_dir:
        temp_gb = _estimate_temp_gb(canvas_w, canvas_h, channels)
        print(f"Canvas: {canvas_w:,} x {canvas_h:,}", flush=True)
        print(f"Temporary work dir: {work_dir}", flush=True)
        print(f"Estimated temp arrays: {temp_gb:.2f} GB before final TIFF", flush=True)
        print("Stitching patches...", flush=True)
        canvas = StitchCanvas((canvas_h, canvas_w, channels), dtype=np.float32, work_dir=work_dir)

        for index, (path, box) in enumerate(zip(image_files, boxes, strict=True), start=1):
            image = _ensure_compatible(_read_image(path), patch_h, patch_w, channels)
            canvas.add(image.astype(np.float32), box)
            if index == 1 or index % progress_every == 0 or index == len(image_files):
                print(f"  stitched {index:,} / {len(image_files):,}: {path.name}", flush=True)

        print(f"Writing {output_path}...", flush=True)
        _write_canvas(output_path, canvas, tiff_metadata=tiff_metadata)
        print(f"Total time: {_format_seconds(time.perf_counter() - total_start)}", flush=True)
        print(f"Done: {output_path}", flush=True)
    return Path(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch a row-major folder of patch images.")
    parser.add_argument("patch_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--cols", type=int, default=None, help="Number of patches per row.")
    parser.add_argument("--stride-x", type=int, default=None)
    parser.add_argument("--stride-y", type=int, default=None)
    parser.add_argument("--overlap-x", type=int, default=0)
    parser.add_argument("--overlap-y", type=int, default=0)
    parser.add_argument(
        "--overlap-percent",
        type=float,
        default=None,
        help="Base overlap percentage for both axes, for example 50 for 1024 patches at stride 512.",
    )
    parser.add_argument(
        "--from-filenames",
        action="store_true",
        help="Read x/y positions from filenames like idx000001_x000512_y000000.tif.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["average", "direct", "align"],
        default="average",
        help="average blends filename placements; direct is fast grid preview; align estimates jitter then blends.",
    )
    parser.add_argument("--jitter-percent", type=float, default=10.0)
    parser.add_argument("--align-downsample", type=int, default=8)
    parser.add_argument("--align-coarse-downsample", type=int, default=16)
    parser.add_argument("--align-refine-radius", type=int, default=4)
    parser.add_argument("--align-min-score", type=float, default=0.12)
    parser.add_argument("--align-max-pairs", type=int, default=None)
    parser.add_argument("--align-solver-iterations", type=int, default=200)
    parser.add_argument("--align-base-weight", type=float, default=0.05)
    parser.add_argument("--align-matcher", choices=["phase", "brute"], default="phase")
    parser.add_argument("--align-diagnostics-dir", type=str, default=None)
    parser.add_argument(
        "--writer",
        choices=["chunked", "cell", "full"],
        default="chunked",
        help="chunked averages overlaps; cell avoids overlap averaging; full uses the older full-canvas accumulator.",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--cell-padding",
        type=int,
        default=32,
        help="Extra pixels copied around each cell for writer=cell to avoid cracks after alignment.",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=0.25,
        help="Default microns per pixel for both axes. Used for QuPath/OpenSlide scale metadata.",
    )
    parser.add_argument("--mpp-x", type=float, default=None, help="Microns per pixel in X.")
    parser.add_argument("--mpp-y", type=float, default=None, help="Microns per pixel in Y.")
    parser.add_argument(
        "--objective-power",
        type=float,
        default=None,
        help="Objective magnification label. If omitted, estimated from MPP.",
    )
    args = parser.parse_args()

    if not args.from_filenames and args.cols is None:
        parser.error("--cols is required unless --from-filenames is used.")

    stitch_patch_folder(
        args.patch_dir,
        args.output_path,
        cols=args.cols or 1,
        stride_x=args.stride_x,
        stride_y=args.stride_y,
        overlap_x=args.overlap_x,
        overlap_y=args.overlap_y,
        overlap_percent=args.overlap_percent,
        from_filenames=args.from_filenames,
        progress_every=args.progress_every,
        mode=args.mode,
        jitter_percent=args.jitter_percent,
        align_downsample=args.align_downsample,
        align_coarse_downsample=args.align_coarse_downsample,
        align_refine_radius=args.align_refine_radius,
        align_min_score=args.align_min_score,
        align_max_pairs=args.align_max_pairs,
        align_solver_iterations=args.align_solver_iterations,
        align_base_weight=args.align_base_weight,
        align_matcher=args.align_matcher,
        align_diagnostics_dir=args.align_diagnostics_dir,
        writer=args.writer,
        chunk_size=args.chunk_size,
        cell_padding=args.cell_padding,
        mpp=args.mpp,
        mpp_x=args.mpp_x,
        mpp_y=args.mpp_y,
        objective_power=args.objective_power,
    )


def _read_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install with the image extra to read/write patch images: pip install -e .[image]") from exc

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _format_seconds(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {sec:.1f}s"
    return f"{sec:.1f}s"


def _write_canvas(path: str | Path, canvas: StitchCanvas, *, tiff_metadata: TiffMetadata) -> None:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        temp_path = _uncompressed_temp_path(path)
        canvas.write_tiff(temp_path, tiff_kwargs=_tiff_writer_kwargs(canvas.shape, tiff_metadata))
        _compress_tiff_output(temp_path, path, tiff_metadata)
        return

    output = np.clip(canvas.finalize(), 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install with the image extra to read/write patch images: pip install -e .[image]") from exc
    Image.fromarray(output).save(path)


def _write_direct_grid_tiff(
    *,
    image_files: list[Path],
    boxes: list[tuple[int, int, int, int]],
    output_path: str | Path,
    shape: tuple[int, int, int],
    patch_shape: tuple[int, int],
    cell_padding: int,
    tiff_metadata: TiffMetadata,
    progress_every: int,
) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Install with the image extra to write TIFF output: pip install -e .[image]") from exc

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("direct mode currently writes .tif/.tiff output only.")
    temp_path = _uncompressed_temp_path(output_path)
    if temp_path.exists():
        temp_path.unlink()

    canvas = tifffile.memmap(
        temp_path,
        shape=shape,
        dtype=np.uint8,
        photometric="rgb" if shape[2] in {3, 4} else "minisblack",
        bigtiff=True,
        **_tiff_writer_kwargs(shape, tiff_metadata),
    )
    canvas[:] = 0

    patch_h, patch_w = patch_shape
    xs = sorted({box[0] for box in boxes})
    ys = sorted({box[1] for box in boxes})
    next_x = {x: xs[index + 1] if index + 1 < len(xs) else x + patch_w for index, x in enumerate(xs)}
    next_y = {y: ys[index + 1] if index + 1 < len(ys) else y + patch_h for index, y in enumerate(ys)}

    print("Writing direct TIFF cells...", flush=True)
    for index, (path, box) in enumerate(zip(image_files, boxes, strict=True), start=1):
        x0, y0, _, _ = box
        x1 = min(next_x[x0], shape[1])
        y1 = min(next_y[y0], shape[0])
        if x0 >= x1 or y0 >= y1:
            continue

        image = _ensure_compatible(_read_image(path), patch_h, patch_w, shape[2])
        cell = image[: y1 - y0, : x1 - x0]
        canvas[y0:y1, x0:x1] = cell

        if index == 1 or index % progress_every == 0 or index == len(image_files):
            print(f"  wrote {index:,} / {len(image_files):,}: {path.name}", flush=True)

    canvas.flush()
    del canvas
    _compress_tiff_output(temp_path, output_path, tiff_metadata)


def _write_cell_tiff(
    *,
    image_files: list[Path],
    boxes: list[tuple[int, int, int, int]],
    base_boxes: list[tuple[int, int, int, int]],
    output_path: str | Path,
    shape: tuple[int, int, int],
    patch_shape: tuple[int, int],
    cell_padding: int,
    tiff_metadata: TiffMetadata,
    progress_every: int,
) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Install with the image extra to write TIFF output: pip install -e .[image]") from exc

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("cell writer currently writes .tif/.tiff output only.")
    temp_path = _uncompressed_temp_path(output_path)
    if temp_path.exists():
        temp_path.unlink()

    canvas_h, canvas_w, channels = shape
    patch_h, patch_w = patch_shape
    output = tifffile.memmap(
        temp_path,
        shape=shape,
        dtype=np.uint8,
        photometric="rgb" if channels in {3, 4} else "minisblack",
        bigtiff=True,
        **_tiff_writer_kwargs(shape, tiff_metadata),
    )
    output[:] = 255

    base_cells = _cell_boxes_from_base_grid(base_boxes)
    print("Writing cell TIFF...", flush=True)
    for index, (path, box, base_box, base_cell) in enumerate(
        zip(image_files, boxes, base_boxes, base_cells, strict=True),
        start=1,
    ):
        crop_x0 = max(0, base_cell[0] - base_box[0] - cell_padding)
        crop_y0 = max(0, base_cell[1] - base_box[1] - cell_padding)
        crop_x1 = min(patch_w, base_cell[2] - base_box[0] + cell_padding)
        crop_y1 = min(patch_h, base_cell[3] - base_box[1] + cell_padding)
        if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
            continue

        dst_x0 = box[0] + crop_x0
        dst_y0 = box[1] + crop_y0
        dst_x1 = box[0] + crop_x1
        dst_y1 = box[1] + crop_y1

        clip_x0 = max(0, dst_x0)
        clip_y0 = max(0, dst_y0)
        clip_x1 = min(canvas_w, dst_x1)
        clip_y1 = min(canvas_h, dst_y1)
        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            continue

        image = _ensure_compatible(_read_image(path), patch_h, patch_w, channels)
        src_x0 = crop_x0 + (clip_x0 - dst_x0)
        src_y0 = crop_y0 + (clip_y0 - dst_y0)
        src_x1 = src_x0 + (clip_x1 - clip_x0)
        src_y1 = src_y0 + (clip_y1 - clip_y0)
        output[clip_y0:clip_y1, clip_x0:clip_x1] = image[src_y0:src_y1, src_x0:src_x1]

        if index == 1 or index % progress_every == 0 or index == len(image_files):
            print(f"  wrote cell {index:,} / {len(image_files):,}: {path.name}", flush=True)

    output.flush()
    del output
    _compress_tiff_output(temp_path, output_path, tiff_metadata)


def _cell_boxes_from_base_grid(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    centers_x = sorted({(box[0] + box[2]) / 2 for box in boxes})
    centers_y = sorted({(box[1] + box[3]) / 2 for box in boxes})
    x_edges = _edges_from_centers(centers_x, min(box[0] for box in boxes), max(box[2] for box in boxes))
    y_edges = _edges_from_centers(centers_y, min(box[1] for box in boxes), max(box[3] for box in boxes))
    x_lookup = {center: (x_edges[index], x_edges[index + 1]) for index, center in enumerate(centers_x)}
    y_lookup = {center: (y_edges[index], y_edges[index + 1]) for index, center in enumerate(centers_y)}

    cells = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        x0, x1 = x_lookup[cx]
        y0, y1 = y_lookup[cy]
        cells.append((x0, y0, x1, y1))
    return cells


def _edges_from_centers(centers: list[float], low: int, high: int) -> list[int]:
    if not centers:
        return [low, high]
    edges = [int(round(low))]
    for left, right in zip(centers, centers[1:], strict=False):
        edges.append(int(round((left + right) / 2)))
    edges.append(int(round(high)))
    return edges


def _write_chunked_average_tiff(
    *,
    image_files: list[Path],
    boxes: list[tuple[int, int, int, int]],
    output_path: str | Path,
    shape: tuple[int, int, int],
    patch_shape: tuple[int, int],
    chunk_size: int,
    tiff_metadata: TiffMetadata,
    progress_every: int,
) -> None:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Install with the image extra to write TIFF output: pip install -e .[image]") from exc

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("chunked writer currently writes .tif/.tiff output only.")
    if chunk_size < 512:
        raise ValueError("chunk_size must be at least 512.")
    temp_path = _uncompressed_temp_path(output_path)
    if temp_path.exists():
        temp_path.unlink()

    canvas_h, canvas_w, channels = shape
    patch_h, patch_w = patch_shape
    output = tifffile.memmap(
        temp_path,
        shape=shape,
        dtype=np.uint8,
        photometric="rgb" if channels in {3, 4} else "minisblack",
        bigtiff=True,
        **_tiff_writer_kwargs(shape, tiff_metadata),
    )

    patch_index = _build_chunk_index(boxes, chunk_size)
    total_chunks_x = math.ceil(canvas_w / chunk_size)
    total_chunks_y = math.ceil(canvas_h / chunk_size)
    total_chunks = total_chunks_x * total_chunks_y
    done_chunks = 0

    print(
        f"Writing chunked averaged TIFF: {total_chunks_y} x {total_chunks_x} chunks",
        flush=True,
    )

    for cy in range(total_chunks_y):
        y0 = cy * chunk_size
        y1 = min(y0 + chunk_size, canvas_h)
        for cx in range(total_chunks_x):
            x0 = cx * chunk_size
            x1 = min(x0 + chunk_size, canvas_w)
            chunk_box = (x0, y0, x1, y1)
            candidates = sorted(patch_index.get((cy, cx), set()))

            values = np.zeros((y1 - y0, x1 - x0, channels), dtype=np.uint16)
            weights = np.zeros((y1 - y0, x1 - x0, 1), dtype=np.uint8)

            for patch_idx in candidates:
                box = boxes[patch_idx]
                ix0 = max(chunk_box[0], box[0])
                iy0 = max(chunk_box[1], box[1])
                ix1 = min(chunk_box[2], box[2])
                iy1 = min(chunk_box[3], box[3])
                if ix0 >= ix1 or iy0 >= iy1:
                    continue

                image = _ensure_compatible(_read_image(image_files[patch_idx]), patch_h, patch_w, channels)
                px0 = ix0 - box[0]
                py0 = iy0 - box[1]
                px1 = px0 + (ix1 - ix0)
                py1 = py0 + (iy1 - iy0)
                cx0 = ix0 - x0
                cy0 = iy0 - y0
                cx1 = cx0 + (ix1 - ix0)
                cy1 = cy0 + (iy1 - iy0)

                values[cy0:cy1, cx0:cx1] += image[py0:py1, px0:px1].astype(np.uint16)
                weights[cy0:cy1, cx0:cx1] += 1

            output[y0:y1, x0:x1] = _finalize_uint8_average(values, weights)
            done_chunks += 1
            if done_chunks == 1 or done_chunks % progress_every == 0 or done_chunks == total_chunks:
                print(
                    f"  wrote chunk {done_chunks:,} / {total_chunks:,} "
                    f"({len(candidates):,} patches)",
                    flush=True,
                )

    output.flush()
    del output
    _compress_tiff_output(temp_path, output_path, tiff_metadata)


def _finalize_uint8_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    averaged = np.zeros(values.shape, dtype=np.uint8)
    valid = weights[:, :, 0] > 0
    if not np.any(valid):
        return averaged
    numerator = values[valid].astype(np.uint32)
    denominator = weights[:, :, 0][valid].astype(np.uint32)[:, None]
    averaged[valid] = ((numerator + denominator // 2) // denominator).astype(np.uint8)
    return averaged


def _resolve_tiff_metadata(
    *,
    mpp: float,
    mpp_x: float | None,
    mpp_y: float | None,
    objective_power: float | None,
) -> TiffMetadata:
    resolved_x = mpp if mpp_x is None else mpp_x
    resolved_y = mpp if mpp_y is None else mpp_y
    if resolved_x <= 0 or resolved_y <= 0:
        raise ValueError("MPP values must be positive.")

    resolved_power = _estimate_objective_power((resolved_x + resolved_y) / 2)
    if objective_power is not None:
        if objective_power <= 0:
            raise ValueError("objective_power must be positive.")
        resolved_power = objective_power

    return TiffMetadata(
        mpp_x=float(resolved_x),
        mpp_y=float(resolved_y),
        objective_power=float(resolved_power),
    )


def _estimate_objective_power(mpp: float) -> float:
    if mpp <= 0:
        return 20.0
    if mpp < 0.35:
        return 40.0
    if mpp < 0.55:
        return 20.0
    if mpp < 1.1:
        return 10.0
    return 5.0


def _tiff_writer_kwargs(shape: tuple[int, int, int], metadata: TiffMetadata) -> dict:
    height, width, _ = shape
    return {
        "resolution": _resolution_from_mpp(metadata.mpp_x, metadata.mpp_y),
        "resolutionunit": "CENTIMETER",
        "description": _image_description(width, height, metadata),
        "metadata": None,
    }


def _resolution_from_mpp(mpp_x: float, mpp_y: float) -> tuple[float, float]:
    # TIFF resolution is pixels per centimeter. MPP is microns per pixel.
    return 10000.0 / mpp_x, 10000.0 / mpp_y


def _image_description(width: int, height: int, metadata: TiffMetadata) -> str:
    mpp = (metadata.mpp_x + metadata.mpp_y) / 2
    return (
        "Aperio Image Library v1.0\r\n"
        f"{width}x{height} [0,0 {width}x{height}] ({height}x{width})"
        f"|AppMag = {metadata.objective_power:g}"
        f"|MPP = {mpp:.4f}"
    )


def _uncompressed_temp_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.uncompressed.tmp{output_path.suffix}")


def _compress_tiff_output(
    uncompressed_path: Path,
    output_path: Path,
    metadata: TiffMetadata,
) -> None:
    print(
        f"Compressing TIFF: {TIFF_COMPRESSION} Q={TIFF_JPEG_QUALITY}, "
        f"tile={TIFF_TILE_SIZE}, pyramid=True",
        flush=True,
    )
    from stitch_engine._vips import configure_project_vips

    configure_project_vips()
    try:
        import pyvips
    except ImportError as exc:
        raise ImportError(
            "Automatic TIFF compression requires pyvips/libvips. "
            "Install pyvips or keep the uncompressed temp TIFF manually."
        ) from exc

    if output_path.exists():
        output_path.unlink()

    gc.collect()
    image = pyvips.Image.new_from_file(str(uncompressed_path), access="sequential")
    if image.bands == 4:
        image = image.flatten(background=[255, 255, 255])
    image = image.cast("uchar")
    image = image.copy(xres=1000.0 / metadata.mpp_x, yres=1000.0 / metadata.mpp_y)
    image.set_type(
        pyvips.GValue.gstr_type,
        "image-description",
        _image_description(image.width, image.height, metadata),
    )
    image.tiffsave(
        str(output_path),
        compression=TIFF_COMPRESSION,
        Q=TIFF_JPEG_QUALITY,
        tile=True,
        tile_width=TIFF_TILE_SIZE,
        tile_height=TIFF_TILE_SIZE,
        pyramid=True,
        depth="onetile",
        bigtiff=True,
    )
    del image
    gc.collect()
    uncompressed_path.unlink(missing_ok=True)


def _build_chunk_index(
    boxes: list[tuple[int, int, int, int]],
    chunk_size: int,
) -> dict[tuple[int, int], set[int]]:
    index: dict[tuple[int, int], set[int]] = {}
    for patch_idx, (x0, y0, x1, y1) in enumerate(boxes):
        cx0 = max(0, x0 // chunk_size)
        cy0 = max(0, y0 // chunk_size)
        cx1 = max(0, (x1 - 1) // chunk_size)
        cy1 = max(0, (y1 - 1) // chunk_size)
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                index.setdefault((cy, cx), set()).add(patch_idx)
    return index


def _box_from_filename(path: Path, patch_w: int, patch_h: int) -> tuple[int, int, int, int]:
    match = XY_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse x/y coordinates from filename: {path.name}")
    x0 = int(match.group("x"))
    y0 = int(match.group("y"))
    return (x0, y0, x0 + patch_w, y0 + patch_h)


def _estimate_temp_gb(width: int, height: int, channels: int) -> float:
    bytes_per_pixel = (channels + 1) * np.dtype(np.float32).itemsize
    return width * height * bytes_per_pixel / 1024**3


def _ensure_compatible(image: np.ndarray, patch_h: int, patch_w: int, channels: int) -> np.ndarray:
    if image.shape[:2] != (patch_h, patch_w):
        raise ValueError("All patch images must have the same width and height.")
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] != channels:
        raise ValueError("All patch images must have the same number of channels.")
    return image


if __name__ == "__main__":
    main()
