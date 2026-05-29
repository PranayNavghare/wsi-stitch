# New Stitch Project Report

## Goal

New Stitch is a metadata-light patch stitching engine for large microscopy or PCB-style image mosaics. The main goal is to build a large image from overlapping patches when the nominal grid is known, but the real capture positions may contain vibration or stage jitter.

The current priority is:

- handle consistent patch sizes detected from input patches, with 1024 x 1024 used as the main test case
- correct variable overlap caused by camera/stage jitter
- avoid blurry overlap averaging when alignment is good
- keep memory use low enough for very large mosaics
- write TIFF output that can be opened in QuPath

## Current Inputs

The engine can work from a folder of image patches.

The preferred patch filenames contain base coordinates:

```text
idx000000_x000000_y000000.tif
idx000001_x000512_y000000.tif
idx000002_x001024_y000000.tif
```

These x/y values are treated as the rough acquisition grid. They do not need to be perfect when alignment mode is enabled.

## Main Components

### `patch_folder.py`

Command-line entry point for stitching a patch folder.

Responsibilities:

- discover image patches
- parse filename coordinates
- build base patch boxes
- call alignment when requested
- select output writer
- write TIFF metadata fields such as MPP and objective power

Important modes:

- `average`: blends overlaps
- `direct`: fast grid preview
- `align`: estimates jitter from neighboring overlap content

Important writers:

- `chunked`: averages overlaps chunk-by-chunk to avoid huge temporary arrays
- `cell`: writes one corrected central cell per patch to avoid blur
- `full`: older full-canvas accumulator

### `alignment.py`

Content-based jitter correction.

Responsibilities:

- build a grid from base x/y boxes
- compare neighboring right/down patch pairs
- estimate local offsets using phase correlation or brute-force matching
- skip blank/low-texture overlaps
- solve a global corrected layout from matched neighbor edges
- optionally export diagnostics CSVs

The default matcher is phase correlation.

### `stitcher.py`

General accumulation canvas for overlap averaging.

Responsibilities:

- add blocks into a large canvas
- accumulate pixel values and weights
- write a normalized TIFF without materializing the full final image at once

This path is useful but slower and can require large temporary storage.

## Algorithm Summary

The stitcher works in four stages.

1. Build a rough layout from filename coordinates.

   The filename x/y values provide the base grid. For 1024 px patches with 50% overlap, the expected stride is about 512 px.

2. Match neighboring overlaps.

   In alignment mode, the engine compares neighboring patch overlaps and estimates how much the second patch should move relative to the first patch.

   Example:

   ```text
   filename says B is dx=512, dy=0 from A
   matcher finds B is dx=519, dy=3 from A
   ```

3. Solve global positions.

   The engine does not simply chain patches row-by-row. It solves a global layout from all reliable matched edges, while keeping filename coordinates as weak anchors. This reduces drift and keeps blank regions stable.

4. Write final TIFF.

   The best current visual output is from the `cell` writer after alignment. It avoids averaging two slightly shifted patches, which is what caused blurry or doubled tissue structures.

## Tested Results

The test dataset had:

- 12,544 patches
- patch size: 1024 x 1024 in this dataset; the engine detects patch dimensions from the first patch
- base overlap: 50%
- jitter search: +/-25%
- output canvas around 57,5xx x 57,6xx pixels

Observed timings on the server:

- older full averaging path: around 1 hour or more
- chunked average writer: around 23 minutes
- phase alignment with chunked writer and 8000 max pairs: around 8 minutes
- phase alignment with full pair scan and cell writer: around 9-10 minutes

Best command used:

```bash
time python -m stitch_engine.patch_folder "/home/kanchenjunga/Downloads/stitch/patches" "stitched_cell.tif" \
  --from-filenames \
  --mode align \
  --align-matcher phase \
  --overlap-percent 50 \
  --jitter-percent 25 \
  --align-downsample 4 \
  --align-min-score 0.005 \
  --align-base-weight 0.01 \
  --align-solver-iterations 300 \
  --writer cell \
  --cell-padding 192 \
  --progress-every 500
```

The output became visually sharp and mostly correct. A few small local mismatches can still appear near blank or low-texture regions.

## Runtime Parameters

The engine is usually called with filename coordinates, alignment mode, phase matching, and the cell writer:

```bash
time python -m stitch_engine.patch_folder "/home/kanchenjunga/Downloads/stitch/patches" "stitched_cell.tif" \
  --from-filenames \
  --mode align \
  --align-matcher phase \
  --overlap-percent 50 \
  --jitter-percent 25 \
  --align-downsample 4 \
  --align-min-score 0.005 \
  --align-base-weight 0.01 \
  --align-solver-iterations 300 \
  --writer cell \
  --cell-padding 192 \
  --mpp 0.307 \
  --objective-power 40 \
  --progress-every 500
```

Parameter meanings:

| Parameter | Purpose |
| --- | --- |
| `patch_dir` | Folder containing image patches. |
| `output_path` | Final stitched TIFF path. |
| `--from-filenames` | Read rough x/y patch positions from filenames. |
| `--mode align` | Estimate camera/stage jitter from overlap content before writing. |
| `--align-matcher phase` | Use phase correlation for local offset estimation. |
| `--overlap-percent 50` | Base overlap used when patches were generated. The pixel stride is derived from the detected patch size. For 1024 px patches, this means about 512 px stride. |
| `--jitter-percent 25` | Search range around the base overlap. This allows large camera/stage vibration. |
| `--align-downsample 4` | Downsample factor for fine alignment. Smaller is more accurate but slower. |
| `--align-min-score 0.005` | Minimum confidence for accepting a match. Lower accepts more pairs but may allow weak matches. |
| `--align-base-weight 0.01` | Weak anchor to filename coordinates during global solve. Lower lets pixel matches move patches more. |
| `--align-solver-iterations 300` | Number of global layout relaxation iterations. |
| `--writer cell` | Writes one corrected central cell per patch instead of averaging overlaps. This avoids blur. |
| `--cell-padding 192` | Extra pixels copied around each cell to reduce cracks/seams after alignment. |
| `--mpp 0.307` | Microns per pixel for QuPath scale metadata. Must come from source metadata or calibration. |
| `--objective-power 40` | Magnification label written to TIFF metadata. |
| `--progress-every 500` | Console progress frequency. |

Compression is not passed as a command-line parameter. It is automatic for TIFF output. After writing the temporary uncompressed TIFF, the engine creates a compressed tiled pyramidal BigTIFF using JPEG quality 90.

## Why Blank Patches Cause Problems

Blank or mostly white patches contain little useful signal. In those regions, image matching cannot reliably tell whether a patch shifted by 0 px, 10 px, or 100 px because the overlap looks almost the same.

The current alignment engine skips many blank pairs. That is correct behavior, but it means those regions rely more heavily on the base filename grid and nearby solved tissue regions.

For PCB images, alignment should usually be easier because tracks, pads, silkscreen, and solder mask texture provide stronger visual features. Large uniform PCB areas can still have the same issue.

## Metadata Status

The engine currently supports MPP and objective-power arguments:

```bash
--mpp 0.307 --objective-power 40
```

If no MPP is passed, the default is:

```text
0.25 um/px
```

Objective power is estimated from MPP when not passed:

```text
mpp < 0.35 -> 40x
mpp < 0.55 -> 20x
mpp < 1.1  -> 10x
else       -> 5x
```

Important note: the engine cannot calculate the true MPP from image pixels alone. MPP must come from original WSI metadata, QuPath image properties, scanner settings, or a calibration target.

Current status: final TIFF conversion is handled by VIPS, which writes resolution metadata and an Aperio-style image description. QuPath metadata recognition should still be verified on each output by checking Magnification, Pixel width, Pixel height, Server type, and Pyramid levels.

## Compression

Compression is now treated as a built-in output feature rather than a separate CLI option.

The current TIFF workflow is:

```text
patches -> temporary uncompressed TIFF -> compressed tiled pyramidal BigTIFF
```

The final conversion uses VIPS with these defaults:

```text
compression: JPEG
quality: 90
tile size: 256 x 256
pyramid: true
pyramid depth: onetile
SubIFD pyramid: disabled, because it caused QuPath crashes in testing
BigTIFF: true
```

This keeps the stitching/writing code simple while producing a much smaller QuPath-friendly final file. The temporary uncompressed TIFF is deleted after the compressed file is written. During writing, disk usage temporarily includes both the uncompressed temporary file and the compressed final file.

This means the image extra now requires `pyvips` and a working system `libvips` installation:

```bash
pip install -e ".[image]"
```

Recommended future comparison metrics:

- file size
- compression ratio
- write/compression time
- read/open time
- MAE / RMSE / PSNR / SSIM against an uncompressed baseline

Weissman score may be used for compression efficiency comparisons, but it does not measure image fidelity.

## Recommended Metadata Flow

For real use, metadata should be stored when patches are generated.

Recommended `patch_meta.json`:

```json
{
  "mpp_x": 0.307,
  "mpp_y": 0.307,
  "objective_power": 40,
  "source_width": 57523,
  "source_height": 57686,
  "patch_size": 1024,
  "overlap_percent": 50
}
```

The stitcher should eventually load metadata in this order:

1. CLI arguments
2. `patch_meta.json`
3. first patch TIFF metadata
4. default fallback

## Reference Repositories Reviewed

Several external repositories were reviewed as references:

- `tiatoolbox`: useful WSI IO and tiling concepts
- `CLAM`: useful patch coordinate conventions, but license/use constraints matter
- `histolab`: clean patch extraction style
- `PathEX` and `PathoPatcher`: less directly useful for jitter stitching
- `ashlar`: highly relevant for phase correlation and global mosaic solving
- `MIST`: highly relevant for global optimization and pair filtering concepts
- `scikit-image`: useful reference for phase correlation methods

The current implementation is custom and focused on this project.

## Cleanup Recommendations

Keep:

- `new stitch/`
- top-level `.gitignore`
- top-level `.git/` if this workspace should remain version-controlled

Inside `new stitch/`, keep:

- `src/`
- `tests/`
- `pyproject.toml`
- `README.md`
- `PROJECT_REPORT.md`

Usually delete or move out:

- `new stitch/venv/` because it can be recreated
- `new stitch/patches/` if the data is copied elsewhere or too large
- `new stitch/tests/__pycache__/`

Keep `tools/generate_patches.py` if it is still used to create test patches. Otherwise archive it with the dataset.

## Next Work

Highest priority:

- finalize QuPath/OpenSlide metadata recognition
- add `patch_meta.json` support
- keep improving cell writer edge handling for the few remaining local mismatches

Performance target:

- current practical result: around 10 minutes for 12,544 patches
- desired target: 5-10 minutes
- likely improvements: faster final writing, cached patch reads, parallel read/write, and possibly tile-based output instead of patch-by-patch output
