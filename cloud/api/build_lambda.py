#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAVI Agent API — empaquetador del zip de Lambda (`api/build/savi-api.zip`).

Copia `handler.py`, `savi_api/` (incluyendo `knowledge/`) y `static/`, e instala
`anthropic` (única dependencia externa permitida) para el runtime de Lambda
(manylinux2014 x86_64, cpython 3.12) usando el pip del venv de `cloud/.venv`.
`boto3`/`botocore` NO se incluyen (los provee el runtime de Lambda).

Uso:
    python api/build_lambda.py
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

log = logging.getLogger("savi.build_lambda")

API_DIR = Path(__file__).resolve().parent
CLOUD_ROOT = API_DIR.parent
BUILD_DIR = API_DIR / "build"
STAGE_DIR = BUILD_DIR / "stage"
ZIP_PATH = BUILD_DIR / "savi-api.zip"

VENV_PIP = CLOUD_ROOT / ".venv" / "bin" / "pip"

# Patrones a excluir siempre al copiar código propio o dependencias instaladas.
_EXCLUDE_DIR_NAMES = {"__pycache__", "tests", "test"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
_EXCLUDE_TOP_LEVEL_DEPS = {"boto3", "botocore", "s3transfer", "jmespath"}


def _should_skip_dir(dirname: str) -> bool:
    return dirname in _EXCLUDE_DIR_NAMES


def _copy_tree_filtered(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            if _should_skip_dir(item.name):
                continue
            _copy_tree_filtered(item, dst / item.name)
        else:
            if item.suffix in _EXCLUDE_SUFFIXES:
                continue
            shutil.copy2(item, dst / item.name)


def stage_source(stage_dir: Path) -> None:
    """Copia handler.py + savi_api/ (con knowledge/) + static/ a `stage_dir`."""
    handler_py = API_DIR / "handler.py"
    if not handler_py.is_file():
        raise FileNotFoundError(f"Falta {handler_py}")
    shutil.copy2(handler_py, stage_dir / "handler.py")

    savi_api_dir = API_DIR / "savi_api"
    if not savi_api_dir.is_dir():
        raise FileNotFoundError(
            f"Falta {savi_api_dir} (otro agente debería haberlo escrito: store/inference/retrieval/tools)"
        )
    _copy_tree_filtered(savi_api_dir, stage_dir / "savi_api")

    static_dir = API_DIR / "static"
    if static_dir.is_dir():
        _copy_tree_filtered(static_dir, stage_dir / "static")
    else:
        log.warning("No existe %s (otro agente debería haberlo escrito); se omite", static_dir)


def install_anthropic(stage_dir: Path, pip_path: Path = VENV_PIP) -> None:
    """Instala `anthropic` para Lambda python3.12 manylinux2014 x86_64, sin boto3/botocore."""
    pip_cmd = str(pip_path) if pip_path.is_file() else sys.executable
    args = [pip_cmd] if pip_path.is_file() else [pip_cmd, "-m", "pip"]
    args += [
        "install",
        "--platform", "manylinux2014_x86_64",
        "--implementation", "cp",
        "--python-version", "3.12",
        "--only-binary=:all:",
        "--target", str(stage_dir),
        "anthropic",
    ]
    log.info("Instalando anthropic para Lambda: %s", " ".join(args))
    subprocess.run(args, check=True)

    # El runtime de Lambda ya trae boto3/botocore; no los empaquetamos (pesan mucho
    # y pueden traer una versión incompatible con la del runtime).
    for dep_name in _EXCLUDE_TOP_LEVEL_DEPS:
        for path in stage_dir.glob(f"{dep_name}*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def zip_stage(stage_dir: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.suffix in _EXCLUDE_SUFFIXES or path.parent.name in _EXCLUDE_DIR_NAMES:
                continue
            if path.name == "RECORD" and path.parent.name.endswith(".dist-info"):
                continue
            arcname = path.relative_to(stage_dir)
            zf.write(path, arcname=arcname)
    return zip_path.stat().st_size


def build(clean: bool = True) -> Path:
    if clean and STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    stage_source(STAGE_DIR)
    install_anthropic(STAGE_DIR)
    size_bytes = zip_stage(STAGE_DIR, ZIP_PATH)

    size_mb = size_bytes / (1024 * 1024)
    log.info("Zip generado: %s (%.2f MB)", ZIP_PATH, size_mb)
    if size_mb >= 50:
        raise RuntimeError(f"El zip supera 50 MB ({size_mb:.2f} MB) — límite de Lambda sin capas")
    return ZIP_PATH


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        zip_path = build()
    except Exception as exc:  # noqa: BLE001
        log.error("Falló el empaquetado: %s", exc)
        return 1
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"OK: {zip_path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
