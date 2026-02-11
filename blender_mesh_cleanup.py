from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _candidate_blender_paths() -> list[str]:
    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        return [env_path]

    which_path = shutil.which("blender")
    if which_path:
        return [which_path]

    # Common Windows install paths
    candidates: list[str] = []
    program_files = os.environ.get("ProgramFiles", r"C:\\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")

    candidates.extend(
        [
            os.path.join(program_files, "Blender Foundation", "Blender", "blender.exe"),
            os.path.join(program_files_x86, "Blender Foundation", "Blender", "blender.exe"),
        ]
    )

    return candidates


def _find_blender_executable() -> Optional[str]:
    for path in _candidate_blender_paths():
        if path and os.path.exists(path):
            return path
    return None


def cleanup_glb_default(
    input_glb_path: str,
    *,
    target_tris: int = 15000,
    merge_dist: float = 0.0005,
    auto_smooth_angle_deg: float = 45.0,
    limited_dissolve_angle_deg: float = 1.0,
    max_decimate_iters: int = 3,
    output_glb_path: Optional[str] = None,
) -> str:
    """Run a deterministic Blender cleanup pipeline on a GLB and write *_clean.glb.

    Best-effort: if Blender is not available or the cleanup fails, returns the original path.
    """

    input_path = Path(input_glb_path)
    if not input_path.exists():
        return input_glb_path

    blender_exe = _find_blender_executable()
    if not blender_exe:
        print("[WARN] Blender not found. Set BLENDER_PATH or add Blender to PATH to enable default GLB cleanup.")
        return input_glb_path

    script_path = Path(__file__).parent / "blender_scripts" / "cleanup_glb.py"
    if not script_path.exists():
        print(f"[WARN] Missing Blender cleanup script at: {script_path}")
        return input_glb_path

    if output_glb_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
    else:
        output_path = Path(output_glb_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        blender_exe,
        "-b",
        "--factory-startup",
        "-P",
        str(script_path),
        "--",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--target-tris",
        str(int(target_tris)),
        "--merge-dist",
        str(float(merge_dist)),
        "--auto-smooth-angle",
        str(float(auto_smooth_angle_deg)),
        "--limited-dissolve-angle",
        str(float(limited_dissolve_angle_deg)),
        "--max-decimate-iters",
        str(int(max_decimate_iters)),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Blender prints a lot of noise to stderr; still useful for debugging.
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("[WARN] Blender mesh cleanup failed; returning original GLB.")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return input_glb_path

    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)

    print("[WARN] Blender cleanup produced no output; returning original GLB.")
    return input_glb_path
