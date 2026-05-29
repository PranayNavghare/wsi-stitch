from __future__ import annotations

import os
from pathlib import Path


def configure_project_vips() -> None:
    project_root = Path(__file__).resolve().parents[2]
    vips_bin = project_root / "vips-dev-8.15" / "bin"

    if not vips_bin.exists():
        return

    os.environ["PATH"] = str(vips_bin) + os.pathsep + os.environ.get("PATH", "")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(vips_bin))