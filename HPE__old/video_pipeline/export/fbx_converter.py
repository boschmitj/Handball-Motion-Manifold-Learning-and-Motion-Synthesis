from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class BlenderFBXConverter:
    """Converts BVH animation files to FBX through Blender CLI."""

    def __init__(self, blender_executable: str = "blender", script_path: Path | None = None) -> None:
        self.blender_executable = blender_executable
        self.script_path = script_path or (Path(__file__).resolve().parent / "blender_bvh_to_fbx.py")

    def _is_windows_blender(self) -> bool:
        exe = self.blender_executable.lower()
        return exe.endswith(".exe") or bool(re.match(r"^/mnt/[a-z]/", exe))

    def _translate_for_blender(self, path: Path) -> str:
        if not self._is_windows_blender():
            return str(path)

        wslpath = shutil.which("wslpath")
        if not wslpath:
            raise RuntimeError("wslpath is required to run Windows Blender from WSL")

        proc = subprocess.run(
            [wslpath, "-w", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to translate path for Windows Blender: {path}\n{proc.stderr.strip()}"
            )
        return proc.stdout.strip()

    def _build_command(self, bvh_path: Path, fbx_path: Path) -> list[str]:
        script_arg = self._translate_for_blender(self.script_path.resolve())
        bvh_arg = self._translate_for_blender(bvh_path)
        fbx_arg = self._translate_for_blender(fbx_path)

        return [
            self.blender_executable,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            script_arg,
            "--",
            bvh_arg,
            fbx_arg,
        ]

    def convert(self, bvh_path: Path, fbx_path: Path) -> Path:
        bvh_path = bvh_path.resolve()
        fbx_path = fbx_path.resolve()

        if not bvh_path.exists():
            raise FileNotFoundError(f"BVH file not found: {bvh_path}")
        if bvh_path.suffix.lower() not in {".bvh", ".bhv"}:
            raise ValueError(f"Expected .bvh/.bhv input file, got: {bvh_path}")
        if not self.script_path.exists():
            raise FileNotFoundError(f"Blender conversion script not found: {self.script_path}")

        fbx_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._build_command(bvh_path, fbx_path)

        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Blender executable not found: {self.blender_executable}. "
                "Install Blender or pass --blender_executable /abs/path/to/blender"
            ) from exc

        if process.returncode != 0:
            raise RuntimeError(
                "Blender BVH->FBX conversion failed. "
                f"Command: {' '.join(command)}\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )

        if not fbx_path.exists():
            raise RuntimeError(f"Blender finished without creating FBX file: {fbx_path}")

        return fbx_path
