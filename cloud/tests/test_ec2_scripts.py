# -*- coding: utf-8 -*-
"""
Simulación local de ec2/run_pipeline.sh con binarios falsos (aws, curl, shutdown,
python) en el PATH. NUNCA toca AWS real, NUNCA lee ~/.aws ni /tmp/savi_aws.env,
NUNCA puede disparar un `shutdown` real: el directorio con los binarios falsos
va PRIMERO en PATH y, además, el script expone el apagado vía la variable
overridable SAVI_SHUTDOWN_CMD, que aquí se fija explícitamente a la ruta del
`shutdown` falso (defensa en profundidad, no sólo el orden del PATH).

Escenarios cubiertos (spec del integrador):
  1. Sin tag SaviRunId           → se apaga, el pipeline (python) NO se invoca.
  2. _SUCCESS.json ya existe     → se omite el pipeline (idempotencia), se apaga.
  3. Camino normal               → python se invoca con los args EXACTOS del
                                    contrato, el log se sube a S3, se apaga.
  4. SaviNoShutdown=true         → nunca se llama a shutdown.
  5. (bonus) bootstrap.sh ausente .bootstrapped → se invoca antes del pipeline.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

RUN_PIPELINE_SH = Path(__file__).resolve().parent.parent / "ec2" / "run_pipeline.sh"

FAKE_AWS = r"""#!/bin/bash
echo "aws $*" >> "$FAKE_CALLS/aws.log"
if [ "$1 $2" = "s3api head-object" ]; then
  if [ "${SUCCESS_EXISTS:-0}" = "1" ]; then
    exit 0
  else
    exit 254
  fi
fi
exit 0
"""

FAKE_CURL = r"""#!/bin/bash
echo "curl $*" >> "$FAKE_CALLS/curl.log"
for a in "$@"; do
  if [[ "$a" == *"/latest/api/token" ]]; then
    printf '%s' "FAKETOKEN"
    exit 0
  fi
done
for a in "$@"; do
  case "$a" in
    *"/tags/instance/SaviRunId")
      if [ -n "${TAG_SAVIRUNID:-}" ]; then printf '%s\n200' "$TAG_SAVIRUNID"; else printf '\n404'; fi
      exit 0 ;;
    *"/tags/instance/SaviInputKey")
      if [ -n "${TAG_SAVIINPUTKEY:-}" ]; then printf '%s\n200' "$TAG_SAVIINPUTKEY"; else printf '\n404'; fi
      exit 0 ;;
    *"/tags/instance/SaviNoShutdown")
      if [ -n "${TAG_SAVINOSHUTDOWN:-}" ]; then printf '%s\n200' "$TAG_SAVINOSHUTDOWN"; else printf '\n404'; fi
      exit 0 ;;
  esac
done
printf ''
exit 0
"""

FAKE_SHUTDOWN = r"""#!/bin/bash
echo "shutdown $*" >> "$FAKE_CALLS/shutdown.log"
exit 0
"""

FAKE_SYSTEMCTL = r"""#!/bin/bash
echo "systemctl $*" >> "$FAKE_CALLS/systemctl.log"
exit 0
"""

FAKE_PYTHON = r"""#!/bin/bash
echo "python $*" >> "$FAKE_CALLS/python.log"
exit "${PYTHON_EXIT_CODE:-0}"
"""

FAKE_BOOTSTRAP = r"""#!/bin/bash
echo "bootstrap $*" >> "$FAKE_CALLS/bootstrap.log"
touch "$SAVI_ROOT/.bootstrapped"
exit 0
"""


def _write_exec(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path: Path):
    """Arma PATH con binarios falsos + un SAVI_ROOT/env file aislados."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    _write_exec(fakebin / "aws", FAKE_AWS)
    _write_exec(fakebin / "curl", FAKE_CURL)
    _write_exec(fakebin / "shutdown", FAKE_SHUTDOWN)
    _write_exec(fakebin / "systemctl", FAKE_SYSTEMCTL)
    _write_exec(fakebin / "python", FAKE_PYTHON)

    calls = tmp_path / "calls"
    calls.mkdir()

    savi_root = tmp_path / "savi_root"
    savi_root.mkdir()
    (savi_root / ".bootstrapped").touch()  # bootstrap ya hecho salvo que el test diga lo contrario
    _write_exec(savi_root / "bootstrap.sh", FAKE_BOOTSTRAP)

    log_dir = tmp_path / "var_log_savi"
    log_dir.mkdir()

    env_file = tmp_path / "savi.env"
    env_file.write_text(
        "RAW_BUCKET=raw-bucket\nPROCESSED_BUCKET=processed-bucket\nREGION=us-east-1\n",
        encoding="utf-8",
    )

    # PATH: fakebin primero, PATH real del sistema después (para date/tee/timeout/mkdir/bash).
    # Nunca se incluye nada que resuelva a un `shutdown` real antes que el falso.
    env = {
        "PATH": f"{fakebin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "SAVI_ROOT": str(savi_root),
        "SAVI_ENV_FILE": str(env_file),
        "SAVI_LOG_DIR": str(log_dir),
        "SAVI_PYTHON_BIN": str(fakebin / "python"),
        "SAVI_TIMEOUT_SECS": "10",
        "SAVI_MAX_LOOPS": "3",
        "FAKE_CALLS": str(calls),
        # Defensa en profundidad además del orden del PATH: shutdown real inalcanzable.
        "SAVI_SHUTDOWN_CMD": f"{fakebin / 'shutdown'} -h now",
    }
    return {
        "env": env,
        "calls": calls,
        "savi_root": savi_root,
        "log_dir": log_dir,
    }


def _run(sandbox, extra_env: dict[str, str] | None = None):
    env = dict(sandbox["env"])
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(RUN_PIPELINE_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc


def _log(sandbox, name: str) -> str:
    p = sandbox["calls"] / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ════════════════════════════════════════════════════════════════════
def test_sin_tag_runid_apaga_y_no_corre_python(sandbox):
    proc = _run(sandbox)
    assert proc.returncode == 0, proc.stderr

    python_log = _log(sandbox, "python.log")
    assert python_log == "", f"no debía invocarse python, log={python_log!r}"

    shutdown_log = _log(sandbox, "shutdown.log")
    assert "shutdown -h now" in shutdown_log, f"debía apagarse, log={shutdown_log!r}"


def test_success_existente_omite_pipeline(sandbox):
    proc = _run(sandbox, {
        "TAG_SAVIRUNID": "run-ya-hecho",
        "TAG_SAVIINPUTKEY": "s3://raw-bucket/input/ames.txt",
        "SUCCESS_EXISTS": "1",
    })
    assert proc.returncode == 0, proc.stderr

    aws_log = _log(sandbox, "aws.log")
    assert "s3api head-object --bucket processed-bucket --key runs/run-ya-hecho/_SUCCESS.json" in aws_log

    python_log = _log(sandbox, "python.log")
    assert python_log == "", f"ya existía _SUCCESS.json, no debía correr python: {python_log!r}"

    shutdown_log = _log(sandbox, "shutdown.log")
    assert "shutdown -h now" in shutdown_log


def test_camino_normal_invoca_python_con_args_exactos(sandbox):
    proc = _run(sandbox, {
        "TAG_SAVIRUNID": "20261002T210501Z-ameshousing",
        "TAG_SAVIINPUTKEY": "s3://raw-bucket/input/AmesHousing.txt",
        "SUCCESS_EXISTS": "0",
    })
    assert proc.returncode == 0, proc.stderr

    python_log = _log(sandbox, "python.log")
    expected_args = (
        "python /savi_root/code/savi_cpu_pipeline.py "
        "--input s3://raw-bucket/input/AmesHousing.txt "
        "--reference s3://raw-bucket/reference/ "
        "--output s3://processed-bucket/runs/20261002T210501Z-ameshousing/ "
        "--run-id 20261002T210501Z-ameshousing"
    ).replace("/savi_root", str(sandbox["savi_root"]))
    assert expected_args in python_log, f"args inesperados: {python_log!r}"

    aws_log = _log(sandbox, "aws.log")
    assert "s3 sync s3://raw-bucket/code/cpu/" in aws_log
    assert "s3 cp" in aws_log and "runs/20261002T210501Z-ameshousing/logs/ec2_pipeline.log" in aws_log

    log_file = sandbox["log_dir"] / "pipeline.log"
    assert log_file.exists()
    assert "pipeline CPU terminó OK" in log_file.read_text(encoding="utf-8")

    shutdown_log = _log(sandbox, "shutdown.log")
    assert "shutdown -h now" in shutdown_log


def test_savi_no_shutdown_true_no_apaga(sandbox):
    proc = _run(sandbox, {
        "TAG_SAVIRUNID": "run-debug",
        "TAG_SAVIINPUTKEY": "s3://raw-bucket/input/ames.txt",
        "SUCCESS_EXISTS": "0",
        "TAG_SAVINOSHUTDOWN": "true",
    })
    assert proc.returncode == 0, proc.stderr

    python_log = _log(sandbox, "python.log")
    assert "run-debug" in python_log

    shutdown_log = _log(sandbox, "shutdown.log")
    assert shutdown_log == "", f"SaviNoShutdown=true: no debía apagarse, log={shutdown_log!r}"


def test_bootstrap_se_invoca_si_falta_marcador(sandbox):
    (sandbox["savi_root"] / ".bootstrapped").unlink()
    proc = _run(sandbox, {
        "TAG_SAVIRUNID": "run-boot",
        "TAG_SAVIINPUTKEY": "s3://raw-bucket/input/ames.txt",
        "SUCCESS_EXISTS": "0",
    })
    assert proc.returncode == 0, proc.stderr

    bootstrap_log = _log(sandbox, "bootstrap.log")
    assert bootstrap_log != "", "bootstrap.sh debía invocarse al faltar .bootstrapped"

    python_log = _log(sandbox, "python.log")
    assert "run-boot" in python_log, "tras el bootstrap el pipeline debía correr igual"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
