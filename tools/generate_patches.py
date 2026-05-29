import argparse
import os
import random
import pyvips


# ============================================================
# WINDOWS VIPS
# ============================================================

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vips_bin = os.path.join(repo_root, "vips-dev-8.15", "bin")

if os.path.exists(vips_bin):
    os.environ["PATH"] = vips_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(vips_bin)


# ============================================================
# IMAGE PREP
# ============================================================

def prepare_image(image):
    image = image.cast("uchar")

    if image.bands == 4:
        image = image.flatten(background=[255, 255, 255])

    if image.bands > 4:
        image = image.extract_band(0, n=3)

    return image


# ============================================================
# GENERATE PATCHES
# ============================================================

def generate_patches(
    image_path,
    output_dir,
    patch_size=1024,
    overlap_percent=0.50,
    level=0,
    compression="jpeg",
    quality=90,
    jitter_percent=0.00,
    seed=None,
):
    print("\nLoading source image...")

    if seed is not None:
        random.seed(seed)

    image = pyvips.Image.new_from_file(
        image_path,
        level=level
    )

    image = prepare_image(image)

    width = image.width
    height = image.height

    stride = int(round(patch_size * (1.0 - overlap_percent)))
    max_jitter = int(round(patch_size * jitter_percent))

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nImage: {width} x {height}")
    print(f"Patch size: {patch_size}")
    print(f"Overlap: {overlap_percent:.0%}")
    print(f"Base stride: {stride}px")
    print(f"Jitter: ±{max_jitter}px ({jitter_percent:.0%} of patch size)")

    count = 0
    row = 0

    base_y = 0

    while base_y < height:

        if base_y + patch_size > height:
            base_y = height - patch_size

        base_x = 0
        col = 0

        while base_x < width:

            if base_x + patch_size > width:
                base_x = width - patch_size

            # ------------------------------------------------------------
            # Unknown jitter simulation:
            # base_x/base_y = camera/stage commanded position
            # actual_x/actual_y = real crop position after hidden jitter
            # Filename stores ONLY base_x/base_y.
            # ------------------------------------------------------------
            jitter_x = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
            jitter_y = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0

            actual_x = base_x + jitter_x
            actual_y = base_y + jitter_y

            actual_x = max(0, min(actual_x, width - patch_size))
            actual_y = max(0, min(actual_y, height - patch_size))

            patch = image.crop(
                actual_x,
                actual_y,
                patch_size,
                patch_size
            )

            # Filename stores base/camera coordinate only.
            # Stitcher does not know actual jitter.
            filename = (
                f"idx{count:06d}_"
                f"x{base_x:06d}_"
                f"y{base_y:06d}.tif"
            )

            save_path = os.path.join(
                output_dir,
                filename
            )

            patch.tiffsave(
                save_path,
                tile=True,
                tile_width=256,
                tile_height=256,
                compression=compression,
                Q=quality,
                bigtiff=True
            )

            count += 1
            col += 1

            if count % 100 == 0:
                print(f"  generated {count} patches")

            if base_x == width - patch_size:
                break

            base_x += stride

        row += 1

        if base_y == height - patch_size:
            break

        base_y += stride


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--compression", default="jpeg")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--level", type=int, default=0)

    parser.add_argument(
        "--jitter",
        type=float,
        default=0.00,
        help="Jitter as fraction of patch size. Example: 0.10 means ±10%."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable patch generation."
    )

    args = parser.parse_args()

    generate_patches(
        image_path=args.image,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap_percent=args.overlap,
        compression=args.compression,
        quality=args.quality,
        level=args.level,
        jitter_percent=args.jitter,
        seed=args.seed,
    )