from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class AlignmentConfig:
    """Settings for content-based local offset estimation."""

    base_overlap_percent: float = 50.0
    jitter_percent: float = 10.0
    downsample: int = 8
    coarse_downsample: int = 16
    refine_radius: int = 4
    min_overlap_pixels: int = 64
    min_score: float = 0.12
    max_pairs: int | None = None
    solver_iterations: int = 200
    base_weight: float = 0.05
    matcher: str = "phase"
    diagnostics_dir: str | None = None


@dataclass(frozen=True)
class MatchResult:
    dx: int
    dy: int
    score: float
    matched: bool
    skipped: bool = False


@dataclass(frozen=True)
class MatchEdge:
    cell_a: tuple[int, int]
    cell_b: tuple[int, int]
    dx: int
    dy: int
    score: float


def align_boxes_from_images(
    image_files: list[Path],
    base_boxes: list[Box],
    *,
    patch_shape: tuple[int, int],
    config: AlignmentConfig,
    progress_every: int = 100,
) -> list[Box]:
    """Estimate corrected boxes from filename/base boxes and image content."""

    patch_h, patch_w = patch_shape
    grid = _grid_from_boxes(base_boxes)
    base_by_cell = {cell: base_boxes[index] for cell, index in grid.cell_to_index.items()}

    match_count = 0
    fallback_count = 0
    skipped_count = 0
    processed = 0
    edges: list[MatchEdge] = []
    attempts: list[dict[str, object]] = []

    print("Estimating jitter from neighboring overlap content...", flush=True)
    print(
        f"  search: base overlap={config.base_overlap_percent:.1f}%, "
        f"jitter=+/-{config.jitter_percent:.1f}%, "
        f"coarse={config.coarse_downsample}, fine={config.downsample}, "
        f"matcher={config.matcher}",
        flush=True,
    )

    for row in range(grid.rows):
        for col in range(grid.cols):
            cell = (row, col)
            if cell not in grid.cell_to_index:
                continue
            neighbors = ((row, col + 1), (row + 1, col))
            for neighbor in neighbors:
                if neighbor not in grid.cell_to_index:
                    continue
                result = _match_cells(
                    image_files,
                    grid.cell_to_index,
                    cell,
                    neighbor,
                    base_by_cell,
                    patch_shape,
                    config,
                )
                if result.matched:
                    edges.append(
                        MatchEdge(
                            cell_a=cell,
                            cell_b=neighbor,
                            dx=result.dx,
                            dy=result.dy,
                            score=result.score,
                        )
                    )
                    match_count += 1
                    status = "matched"
                elif result.skipped:
                    skipped_count += 1
                    status = "skipped"
                else:
                    fallback_count += 1
                    status = "fallback"
                if config.diagnostics_dir is not None:
                    attempts.append(
                        {
                            "a_row": cell[0],
                            "a_col": cell[1],
                            "b_row": neighbor[0],
                            "b_col": neighbor[1],
                            "status": status,
                            "dx": result.dx,
                            "dy": result.dy,
                            "score": result.score,
                        }
                    )
                processed += 1

                if processed == 1 or processed % progress_every == 0:
                    print(
                        f"  aligned pairs {processed:,}: matched={match_count:,}, "
                        f"skipped={skipped_count:,}, fallback={fallback_count:,}",
                        flush=True,
                    )
                if config.max_pairs is not None and processed >= config.max_pairs:
                    break
            if config.max_pairs is not None and processed >= config.max_pairs:
                break

        if config.max_pairs is not None and processed >= config.max_pairs:
            break

    print(
        f"  solving global layout from {len(edges):,} matched edges...",
        flush=True,
    )
    positions = _solve_positions(
        grid=grid,
        base_by_cell=base_by_cell,
        edges=edges,
        iterations=config.solver_iterations,
        base_weight=config.base_weight,
    )

    if config.diagnostics_dir is not None:
        _write_diagnostics(
            Path(config.diagnostics_dir),
            grid=grid,
            base_by_cell=base_by_cell,
            positions=positions,
            edges=edges,
            attempts=attempts,
        )

    corrected = []
    for index, box in enumerate(base_boxes):
        cell = grid.index_to_cell[index]
        x, y = positions.get(cell, (float(box[0]), float(box[1])))
        x0 = int(round(x))
        y0 = int(round(y))
        corrected.append((x0, y0, x0 + patch_w, y0 + patch_h))

    min_x = min(box[0] for box in corrected)
    min_y = min(box[1] for box in corrected)
    normalized = [
        (box[0] - min_x, box[1] - min_y, box[2] - min_x, box[3] - min_y)
        for box in corrected
    ]
    print(
        f"  alignment done: matched={match_count:,}, "
        f"skipped={skipped_count:,}, fallback={fallback_count:,}",
        flush=True,
    )
    return normalized


@dataclass(frozen=True)
class _Grid:
    rows: int
    cols: int
    cell_to_index: dict[tuple[int, int], int]
    index_to_cell: dict[int, tuple[int, int]]


def _grid_from_boxes(boxes: list[Box]) -> _Grid:
    xs = sorted({box[0] for box in boxes})
    ys = sorted({box[1] for box in boxes})
    x_to_col = {x: index for index, x in enumerate(xs)}
    y_to_row = {y: index for index, y in enumerate(ys)}
    cell_to_index = {}
    index_to_cell = {}
    for index, box in enumerate(boxes):
        cell = (y_to_row[box[1]], x_to_col[box[0]])
        cell_to_index[cell] = index
        index_to_cell[index] = cell
    return _Grid(rows=len(ys), cols=len(xs), cell_to_index=cell_to_index, index_to_cell=index_to_cell)


def _write_diagnostics(
    path: Path,
    *,
    grid: _Grid,
    base_by_cell: dict[tuple[int, int], Box],
    positions: dict[tuple[int, int], tuple[float, float]],
    edges: list[MatchEdge],
    attempts: list[dict[str, object]],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "alignment_edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["a_row", "a_col", "b_row", "b_col", "status", "dx", "dy", "score"],
        )
        writer.writeheader()
        writer.writerows(attempts)

    components = _components_from_edges(grid, edges)
    with (path / "alignment_positions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row",
                "col",
                "component",
                "base_x",
                "base_y",
                "solved_x",
                "solved_y",
                "delta_x",
                "delta_y",
                "degree",
            ],
        )
        writer.writeheader()
        degrees = _edge_degrees(edges)
        for index, cell in grid.index_to_cell.items():
            base = base_by_cell[cell]
            solved = positions[cell]
            writer.writerow(
                {
                    "row": cell[0],
                    "col": cell[1],
                    "component": components.get(cell, -1),
                    "base_x": base[0],
                    "base_y": base[1],
                    "solved_x": round(solved[0], 3),
                    "solved_y": round(solved[1], 3),
                    "delta_x": round(solved[0] - base[0], 3),
                    "delta_y": round(solved[1] - base[1], 3),
                    "degree": degrees.get(cell, 0),
                }
            )

    print(f"  alignment diagnostics: {path}", flush=True)


def _components_from_edges(grid: _Grid, edges: list[MatchEdge]) -> dict[tuple[int, int], int]:
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {
        cell: [] for cell in grid.cell_to_index
    }
    for edge in edges:
        adjacency[edge.cell_a].append(edge.cell_b)
        adjacency[edge.cell_b].append(edge.cell_a)

    components: dict[tuple[int, int], int] = {}
    component_id = 0
    for cell in adjacency:
        if cell in components:
            continue
        stack = [cell]
        components[cell] = component_id
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in components:
                    components[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1
    return components


def _edge_degrees(edges: list[MatchEdge]) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for edge in edges:
        degrees[edge.cell_a] = degrees.get(edge.cell_a, 0) + 1
        degrees[edge.cell_b] = degrees.get(edge.cell_b, 0) + 1
    return degrees


def _solve_positions(
    *,
    grid: _Grid,
    base_by_cell: dict[tuple[int, int], Box],
    edges: list[MatchEdge],
    iterations: int,
    base_weight: float,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Relax matched neighbor offsets into one globally consistent layout."""

    positions = {
        cell: np.array([float(box[0]), float(box[1])], dtype=np.float64)
        for cell, box in base_by_cell.items()
    }
    base_positions = {cell: pos.copy() for cell, pos in positions.items()}
    if not edges:
        return {cell: (float(pos[0]), float(pos[1])) for cell, pos in positions.items()}

    constraints: dict[tuple[int, int], list[tuple[tuple[int, int], np.ndarray, float]]] = {
        cell: [] for cell in positions
    }
    for edge in edges:
        offset = np.array([float(edge.dx), float(edge.dy)], dtype=np.float64)
        weight = max(0.05, float(edge.score))
        constraints[edge.cell_a].append((edge.cell_b, offset, weight))
        constraints[edge.cell_b].append((edge.cell_a, -offset, weight))

    anchor = min(positions)
    anchor_pos = base_positions[anchor].copy()

    for _ in range(max(1, iterations)):
        next_positions: dict[tuple[int, int], np.ndarray] = {}
        for cell, current in positions.items():
            total = base_positions[cell] * base_weight
            weight_sum = base_weight
            for other, offset, weight in constraints.get(cell, []):
                total += (positions[other] - offset) * weight
                weight_sum += weight
            next_positions[cell] = total / weight_sum if weight_sum > 0 else current

        drift = next_positions[anchor] - anchor_pos
        positions = {cell: pos - drift for cell, pos in next_positions.items()}

    return {cell: (float(pos[0]), float(pos[1])) for cell, pos in positions.items()}


def _match_cells(
    image_files: list[Path],
    cell_to_index: dict[tuple[int, int], int],
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    base_by_cell: dict[tuple[int, int], Box],
    patch_shape: tuple[int, int],
    config: AlignmentConfig,
) -> MatchResult:
    base_a = base_by_cell[cell_a]
    base_b = base_by_cell[cell_b]

    if config.matcher == "phase":
        result = _phase_match_cells(
            image_files=image_files,
            cell_to_index=cell_to_index,
            cell_a=cell_a,
            cell_b=cell_b,
            base_a=base_a,
            base_b=base_b,
            patch_shape=patch_shape,
            config=config,
        )
        if result.matched or result.skipped:
            return result
    elif config.matcher != "brute":
        raise ValueError("matcher must be 'phase' or 'brute'.")

    coarse = max(config.downsample, config.coarse_downsample)
    coarse_a = _read_gray_small(image_files[cell_to_index[cell_a]], coarse)
    coarse_b = _read_gray_small(image_files[cell_to_index[cell_b]], coarse)
    coarse_expected_dx = round((base_b[0] - base_a[0]) / coarse)
    coarse_expected_dy = round((base_b[1] - base_a[1]) / coarse)
    coarse_jitter = max(1, round(max(patch_shape) * config.jitter_percent / 100 / coarse))
    coarse_min_overlap = max(4, round(config.min_overlap_pixels / coarse))

    coarse_result = _search_translation(
        coarse_a,
        coarse_b,
        expected_dx=coarse_expected_dx,
        expected_dy=coarse_expected_dy,
        jitter=coarse_jitter,
        min_overlap=coarse_min_overlap,
    )

    fine_a = _read_gray_small(image_files[cell_to_index[cell_a]], config.downsample)
    fine_b = _read_gray_small(image_files[cell_to_index[cell_b]], config.downsample)
    fine_expected_dx = round(coarse_result.dx * coarse / config.downsample)
    fine_expected_dy = round(coarse_result.dy * coarse / config.downsample)
    fine_min_overlap = max(4, round(config.min_overlap_pixels / config.downsample))
    fine_jitter = max(1, config.refine_radius)

    fine_result = _search_translation(
        fine_a,
        fine_b,
        expected_dx=fine_expected_dx,
        expected_dy=fine_expected_dy,
        jitter=fine_jitter,
        min_overlap=fine_min_overlap,
    )
    if fine_result.score < 0:
        return MatchResult(
            dx=base_b[0] - base_a[0],
            dy=base_b[1] - base_a[1],
            score=fine_result.score,
            matched=False,
            skipped=True,
        )
    if fine_result.score < config.min_score:
        return MatchResult(
            dx=base_b[0] - base_a[0],
            dy=base_b[1] - base_a[1],
            score=fine_result.score,
            matched=False,
        )
    return MatchResult(
        dx=fine_result.dx * config.downsample,
        dy=fine_result.dy * config.downsample,
        score=fine_result.score,
        matched=True,
    )


def _phase_match_cells(
    *,
    image_files: list[Path],
    cell_to_index: dict[tuple[int, int], int],
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    base_a: Box,
    base_b: Box,
    patch_shape: tuple[int, int],
    config: AlignmentConfig,
) -> MatchResult:
    image_a = _read_gray_small(image_files[cell_to_index[cell_a]], config.downsample)
    image_b = _read_gray_small(image_files[cell_to_index[cell_b]], config.downsample)
    expected_dx = round((base_b[0] - base_a[0]) / config.downsample)
    expected_dy = round((base_b[1] - base_a[1]) / config.downsample)
    max_shift = max(1, round(max(patch_shape) * config.jitter_percent / 100 / config.downsample))
    min_overlap = max(4, round(config.min_overlap_pixels / config.downsample))

    overlap = _overlap_views(image_a, image_b, expected_dx, expected_dy, min_overlap)
    if overlap is None:
        return MatchResult(base_b[0] - base_a[0], base_b[1] - base_a[1], -1.0, False, True)
    ref, moving = overlap
    if float(np.std(ref)) < 3.0 or float(np.std(moving)) < 3.0:
        return MatchResult(base_b[0] - base_a[0], base_b[1] - base_a[1], -1.0, False, True)

    shift_y, shift_x, score = _phase_shift(ref, moving)
    if abs(shift_x) > max_shift or abs(shift_y) > max_shift:
        return MatchResult(base_b[0] - base_a[0], base_b[1] - base_a[1], score, False)
    if score < config.min_score:
        return MatchResult(base_b[0] - base_a[0], base_b[1] - base_a[1], score, False)

    corrected_dx = round((expected_dx + shift_x) * config.downsample)
    corrected_dy = round((expected_dy + shift_y) * config.downsample)
    return MatchResult(corrected_dx, corrected_dy, score, True)


def _overlap_views(
    image_a: np.ndarray,
    image_b: np.ndarray,
    dx: int,
    dy: int,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = image_a.shape
    ax0 = max(0, dx)
    ay0 = max(0, dy)
    ax1 = min(w, dx + w)
    ay1 = min(h, dy + h)
    overlap_w = ax1 - ax0
    overlap_h = ay1 - ay0
    if overlap_w < min_overlap or overlap_h < min_overlap:
        return None
    bx0 = ax0 - dx
    by0 = ay0 - dy
    return (
        image_a[ay0:ay1, ax0:ax1],
        image_b[by0 : by0 + overlap_h, bx0 : bx0 + overlap_w],
    )


def _phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int, float]:
    ref = _windowed_normalized(reference)
    mov = _windowed_normalized(moving)
    product = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(product)
    product /= np.maximum(magnitude, 1e-12)
    corr = np.fft.ifft2(product).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shift_y = int(peak[0])
    shift_x = int(peak[1])
    if shift_y > corr.shape[0] // 2:
        shift_y -= corr.shape[0]
    if shift_x > corr.shape[1] // 2:
        shift_x -= corr.shape[1]
    score = float(corr[peak])
    return shift_y, shift_x, score


def _windowed_normalized(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    arr = arr - float(np.mean(arr))
    std = float(np.std(arr))
    if std > 1e-6:
        arr = arr / std
    wy = np.hanning(arr.shape[0]).astype(np.float32)
    wx = np.hanning(arr.shape[1]).astype(np.float32)
    return arr * wy[:, None] * wx[None, :]


def _read_gray_small(path: Path, downsample: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install with the image extra to align patch images: pip install -e .[image]") from exc

    with Image.open(path) as image:
        gray = image.convert("L")
        if downsample > 1:
            size = (max(1, gray.width // downsample), max(1, gray.height // downsample))
            gray = gray.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32)


def _search_translation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    expected_dx: int,
    expected_dy: int,
    jitter: int,
    min_overlap: int,
) -> MatchResult:
    best = MatchResult(dx=expected_dx, dy=expected_dy, score=-1.0, matched=False)
    for dy in range(expected_dy - jitter, expected_dy + jitter + 1):
        for dx in range(expected_dx - jitter, expected_dx + jitter + 1):
            score = _overlap_score(image_a, image_b, dx, dy, min_overlap)
            if score > best.score:
                best = MatchResult(dx=dx, dy=dy, score=score, matched=True)
    return best


def _overlap_score(
    image_a: np.ndarray,
    image_b: np.ndarray,
    dx: int,
    dy: int,
    min_overlap: int,
) -> float:
    h, w = image_a.shape
    ax0 = max(0, dx)
    ay0 = max(0, dy)
    ax1 = min(w, dx + w)
    ay1 = min(h, dy + h)
    overlap_w = ax1 - ax0
    overlap_h = ay1 - ay0
    if overlap_w < min_overlap or overlap_h < min_overlap:
        return -1.0

    bx0 = ax0 - dx
    by0 = ay0 - dy
    a = image_a[ay0:ay1, ax0:ax1]
    b = image_b[by0 : by0 + overlap_h, bx0 : bx0 + overlap_w]

    if a.size == 0 or b.size == 0:
        return -1.0
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 3.0 or b_std < 3.0:
        return -1.0

    a_norm = (a - float(np.mean(a))) / a_std
    b_norm = (b - float(np.mean(b))) / b_std
    return float(np.mean(a_norm * b_norm))
