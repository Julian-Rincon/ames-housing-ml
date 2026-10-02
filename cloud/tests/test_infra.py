# -*- coding: utf-8 -*-
"""
Tests de `infra/deploy.py` y `infra/iam/*.json`.

Todos son tests sin red: funciones puras (tar, zip, user-data, configs de
notificación) y un subprocess con `--dry-run` que no debe necesitar
credenciales AWS ni hacer ninguna llamada real.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

CLOUD_ROOT = Path(__file__).resolve().parent.parent
INFRA_DIR = CLOUD_ROOT / "infra"
IAM_DIR = INFRA_DIR / "iam"

sys.path.insert(0, str(INFRA_DIR))

import deploy  # noqa: E402  (import tras ajustar sys.path)


# --------------------------------------------------------------------------- #
# sourcedir.tar.gz
# --------------------------------------------------------------------------- #

def test_build_sourcedir_tar_has_files_at_root(tmp_path):
    (tmp_path / "savi_gpu_sagemaker.py").write_text("# gpu script\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("# utils\n", encoding="utf-8")

    tar_bytes = deploy.build_sourcedir_tar_bytes(tmp_path)

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        names = sorted(tar.getnames())
        assert names == ["savi_gpu_sagemaker.py", "utils.py"]
        # Deben estar en la raíz del tar (sin subdirectorios).
        for name in names:
            assert "/" not in name


def test_build_sourcedir_tar_missing_file_raises(tmp_path):
    (tmp_path / "utils.py").write_text("# utils\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        deploy.build_sourcedir_tar_bytes(tmp_path)


# --------------------------------------------------------------------------- #
# zip de lambdas
# --------------------------------------------------------------------------- #

def test_zip_single_file_contains_only_that_file(tmp_path):
    module = tmp_path / "lambda_start_ec2.py"
    module.write_text("def handler(event, context):\n    return {}\n", encoding="utf-8")

    zip_bytes = deploy.zip_single_file(module)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert zf.namelist() == ["lambda_start_ec2.py"]


def test_zip_single_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        deploy.zip_single_file(tmp_path / "no_existe.py")


# --------------------------------------------------------------------------- #
# user-data: sustitución de placeholders
# --------------------------------------------------------------------------- #

def test_render_user_data_substitutes_all_placeholders(tmp_path):
    template = tmp_path / "user-data.sh.tpl"
    template.write_text(
        "#!/bin/bash\n"
        "RAW=__RAW_BUCKET__\n"
        "PROCESSED=__PROCESSED_BUCKET__\n"
        "REGION=__REGION__\n"
        "LOG_GROUP=__LOG_GROUP__\n",
        encoding="utf-8",
    )

    rendered = deploy.render_user_data(
        template,
        {
            "__RAW_BUCKET__": "savi-raw-006840014780",
            "__PROCESSED_BUCKET__": "savi-processed-006840014780",
            "__REGION__": "us-east-1",
            "__LOG_GROUP__": "/savi/ec2-cpu-engine",
        },
    )

    assert "savi-raw-006840014780" in rendered
    assert "savi-processed-006840014780" in rendered
    assert "us-east-1" in rendered
    assert "/savi/ec2-cpu-engine" in rendered
    assert "__" not in rendered.replace("__bin", "")  # no quedan placeholders sin reemplazar


def test_render_user_data_missing_template_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        deploy.render_user_data(tmp_path / "no_existe.sh.tpl", {"__X__": "y"})


# --------------------------------------------------------------------------- #
# configuración de notificaciones S3
# --------------------------------------------------------------------------- #

def test_raw_notification_config_structure():
    cfg = deploy.build_raw_notification_config("arn:aws:lambda:us-east-1:123:function:savi-start-ec2")
    rules = cfg["LambdaFunctionConfigurations"][0]["Filter"]["Key"]["FilterRules"]
    assert {"Name": "prefix", "Value": "input/"} in rules
    assert cfg["LambdaFunctionConfigurations"][0]["Events"] == ["s3:ObjectCreated:*"]


def test_processed_notification_config_structure():
    cfg = deploy.build_processed_notification_config("arn:aws:lambda:us-east-1:123:function:savi-start-sagemaker")
    rules = cfg["LambdaFunctionConfigurations"][0]["Filter"]["Key"]["FilterRules"]
    assert {"Name": "prefix", "Value": "runs/"} in rules
    assert {"Name": "suffix", "Value": "_SUCCESS.json"} in rules


def test_processed_lifecycle_config_has_both_rules():
    cfg = deploy.processed_lifecycle_config()
    ids = {rule["ID"] for rule in cfg["Rules"]}
    assert "savi-abort-incomplete-multipart-1d" in ids
    assert "savi-expire-runs-30d" in ids
    expire_rule = next(r for r in cfg["Rules"] if r["ID"] == "savi-expire-runs-30d")
    assert expire_rule["Filter"]["Prefix"] == "runs/"
    assert expire_rule["Expiration"]["Days"] == 30


# --------------------------------------------------------------------------- #
# helpers puros
# --------------------------------------------------------------------------- #

def test_bucket_names():
    raw, processed = deploy.bucket_names("006840014780")
    assert raw == "savi-raw-006840014780"
    assert processed == "savi-processed-006840014780"


def test_lab_role_arn():
    assert deploy.lab_role_arn("006840014780") == "arn:aws:iam::006840014780:role/LabRole"


# --------------------------------------------------------------------------- #
# dry-run sin credenciales (subprocess)
# --------------------------------------------------------------------------- #

def test_dry_run_without_credentials_exits_zero():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("AWS_") and k != "AWS_PROFILE"
    }
    env["PATH"] = os.environ.get("PATH", "")

    result = subprocess.run(
        [sys.executable, str(INFRA_DIR / "deploy.py"), "--dry-run", "--account-id", "123456789012"],
        cwd=str(CLOUD_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "DRY RUN" in result.stdout
    assert "123456789012" in result.stdout


# --------------------------------------------------------------------------- #
# IAM JSONs
# --------------------------------------------------------------------------- #

IAM_FILES = sorted(IAM_DIR.glob("*.json"))


def test_iam_dir_has_expected_files():
    names = {p.name for p in IAM_FILES}
    expected = {
        "ec2_trust_policy.json", "ec2_role_policy.json",
        "lambda1_trust_policy.json", "lambda1_role_policy.json",
        "lambda2_trust_policy.json", "lambda2_role_policy.json",
        "sagemaker_trust_policy.json", "sagemaker_role_policy.json",
    }
    assert expected.issubset(names)


@pytest.mark.parametrize("path", IAM_FILES, ids=lambda p: p.name)
def test_iam_json_is_valid_with_correct_version(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["Version"] == "2012-10-17"
    assert "Statement" in data and len(data["Statement"]) >= 1
