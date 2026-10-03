# -*- coding: utf-8 -*-
"""
SAVI v2 · Cloud — Lambda 2 (`savi-start-sagemaker`).

Disparada por la escritura de `savi-processed-<acct>/runs/<run_id>/_SUCCESS.json`
(el último artefacto que deja el motor CPU en EC2). Lee el `_SUCCESS.json`,
valida `status == "SUCCEEDED"` y lanza el Training Job de SageMaker (motor GPU
Double DQN) con la cadena de fallback Spot → On-Demand → tipo/instancia alterna.

Ver CONTRACT.md → "Lambda 2 → SageMaker".
"""
from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("savi.lambda_start_sagemaker")
logger.setLevel(logging.INFO)

_SUCCESS_SUFFIX = "_SUCCESS.json"
_RUNS_PREFIX = "runs/"

_JOB_NAME_MAX_LEN = 63
_JOB_NAME_PREFIX = "savi-dqn-"

# Códigos/errores de ClientError que disparan el siguiente escalón del fallback.
# AccessDeniedException: AWS Academy Learner Lab niega por política IAM explícita los
# tipos GPU de SageMaker (p.ej. ml.g4dn.xlarge) aunque la cuota exista → bajar a CPU.
_CAPACITY_ERROR_CODES = {"ResourceLimitExceeded", "CapacityError", "AccessDeniedException"}
_SPOT_VALIDATION_HINTS = ("spot", "instance type")


def _sagemaker_client():
    """Cliente SageMaker perezoso (permite inyectar un stub en tests)."""
    return boto3.client("sagemaker")


def _s3_client():
    """Cliente S3 perezoso (permite inyectar un stub en tests)."""
    return boto3.client("s3")


def extract_run_id_from_key(key: str) -> str | None:
    """
    Extrae el run_id de una key `runs/<run_id>/_SUCCESS.json`.
    Devuelve None si la key no corresponde al patrón esperado.
    """
    if not key.startswith(_RUNS_PREFIX) or not key.endswith(_SUCCESS_SUFFIX):
        return None
    remainder = key[len(_RUNS_PREFIX):]
    parts = remainder.split("/")
    # Debe ser exactamente runs/<run_id>/_SUCCESS.json (2 segmentos tras el prefijo).
    if len(parts) != 2 or not parts[0]:
        return None
    return parts[0]


def sanitize_job_name(run_id: str) -> str:
    """
    Construye un nombre de training job válido y único:
    `savi-dqn-<run_id-sanitizado>`, ≤63 chars, cumpliendo
    `^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}`.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9-]+", "-", run_id)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    if not sanitized:
        sanitized = "run"
    max_suffix_len = _JOB_NAME_MAX_LEN - len(_JOB_NAME_PREFIX)
    sanitized = sanitized[:max_suffix_len].strip("-") or "run"
    name = f"{_JOB_NAME_PREFIX}{sanitized}"
    # Garantiza que empiece con alfanumérico (el prefijo ya lo asegura) y longitud final.
    return name[:_JOB_NAME_MAX_LEN]


def build_training_job_request(run_id: str, instance_type: str, image_uri: str, use_spot: bool, env: dict) -> dict:
    """
    Función pura: construye el dict de `create_training_job` según CONTRACT.md.

    `env` espera las claves: ROLE_ARN, CODE_S3_URI, PROCESSED_BUCKET,
    MAX_RUNTIME (int/str), EPOCHS (int/str), REGION.
    """
    processed_bucket = env["PROCESSED_BUCKET"]
    code_s3_uri = env["CODE_S3_URI"]
    max_runtime = int(env["MAX_RUNTIME"])
    epochs = int(env["EPOCHS"])
    region = env["REGION"]
    role_arn = env["ROLE_ARN"]

    job_name = sanitize_job_name(run_id)
    processed_s3_uri = f"s3://{processed_bucket}/{_RUNS_PREFIX}{run_id}/"
    output_s3_uri = f"s3://{processed_bucket}/{_RUNS_PREFIX}{run_id}/sagemaker/"
    checkpoint_s3_uri = f"s3://{processed_bucket}/{_RUNS_PREFIX}{run_id}/checkpoints/"

    hyperparameters = {
        "sagemaker_program": json.dumps("savi_gpu_sagemaker.py"),
        "sagemaker_submit_directory": json.dumps(code_s3_uri),
        "sagemaker_region": json.dumps(region),
        "sagemaker_container_log_level": json.dumps(20),
        "epochs": json.dumps(epochs),
        "run-id": json.dumps(run_id),
        "publish-prefix": json.dumps(f"s3://{processed_bucket}/{_RUNS_PREFIX}{run_id}/"),
    }

    request = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {
                "ChannelName": "processed",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": processed_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": output_s3_uri},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
        "Tags": [
            {"Key": "Project", "Value": "SAVI"},
            {"Key": "RunId", "Value": run_id},
        ],
    }

    if use_spot:
        request["EnableManagedSpotTraining"] = True
        request["StoppingCondition"]["MaxWaitTimeInSeconds"] = 2 * max_runtime
        request["CheckpointConfig"] = {
            "S3Uri": checkpoint_s3_uri,
            "LocalPath": "/opt/ml/checkpoints",
        }

    return request


def _is_capacity_or_spot_error(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    code = error.get("Code", "")
    if code in _CAPACITY_ERROR_CODES:
        return True
    if code == "ValidationException":
        message = error.get("Message", "").lower()
        return any(hint in message for hint in _SPOT_VALIDATION_HINTS)
    return False


def _is_already_exists(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") == "ResourceInUse"


def _create_with_fallback(sm, run_id: str, env: dict) -> dict:
    """
    Intenta crear el training job siguiendo la cadena de fallback:
    1. Spot, INSTANCE_TYPE (GPU) (si USE_SPOT=true).
    2. On-Demand, INSTANCE_TYPE (GPU).
    3. Spot, FALLBACK_INSTANCE_TYPE con CPU_IMAGE_URI (si USE_SPOT=true).
    4. On-Demand, FALLBACK_INSTANCE_TYPE con CPU_IMAGE_URI.
    `ResourceInUse` en cualquier intento se trata como éxito idempotente.
    """
    use_spot_env = str(env.get("USE_SPOT", "true")).lower() == "true"
    instance_type = env["INSTANCE_TYPE"]
    image_uri = env["IMAGE_URI"]
    fallback_instance_type = env["FALLBACK_INSTANCE_TYPE"]
    cpu_image_uri = env["CPU_IMAGE_URI"]

    attempts = []
    if use_spot_env:
        attempts.append((instance_type, image_uri, True))
    attempts.append((instance_type, image_uri, False))
    if use_spot_env:
        attempts.append((fallback_instance_type, cpu_image_uri, True))
    attempts.append((fallback_instance_type, cpu_image_uri, False))

    last_exc = None
    for i, (inst_type, img_uri, use_spot) in enumerate(attempts):
        request = build_training_job_request(run_id, inst_type, img_uri, use_spot, env)
        try:
            logger.info(
                "run_id=%s | intento %d/%d → job=%s instance=%s spot=%s",
                run_id, i + 1, len(attempts), request["TrainingJobName"], inst_type, use_spot,
            )
            sm.create_training_job(**request)
            logger.info("run_id=%s | training job creado: %s", run_id, request["TrainingJobName"])
            return {
                "run_id": run_id,
                "training_job_name": request["TrainingJobName"],
                "instance_type": inst_type,
                "use_spot": use_spot,
                "status": "created",
                "attempt": i + 1,
            }
        except ClientError as exc:
            last_exc = exc
            if _is_already_exists(exc):
                logger.info(
                    "run_id=%s | training job %s ya existe (ResourceInUse) → idempotente, no es error",
                    run_id, request["TrainingJobName"],
                )
                return {
                    "run_id": run_id,
                    "training_job_name": request["TrainingJobName"],
                    "instance_type": inst_type,
                    "use_spot": use_spot,
                    "status": "already_exists",
                    "attempt": i + 1,
                }
            if _is_capacity_or_spot_error(exc) and i < len(attempts) - 1:
                logger.warning(
                    "run_id=%s | intento %d falló (%s) → probando siguiente escalón del fallback",
                    run_id, i + 1, exc.response.get("Error", {}).get("Code"),
                )
                continue
            logger.error("run_id=%s | fallo irrecuperable creando el training job: %s", run_id, exc)
            raise

    # No debería llegar aquí, pero por completitud:
    raise last_exc if last_exc else RuntimeError(f"run_id={run_id} | no se pudo crear el training job")


def _read_success_record(s3, bucket: str, key: str) -> dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    return json.loads(body)


def _handle_record(sm, s3, record: dict, env: dict) -> dict | None:
    bucket = record["s3"]["bucket"]["name"]
    raw_key = record["s3"]["object"]["key"]
    key = unquote_plus(raw_key)

    run_id = extract_run_id_from_key(key)
    if run_id is None:
        logger.info("Ignorando key fuera de alcance: %s", key)
        return None

    logger.info("run_id=%s | leyendo _SUCCESS.json de s3://%s/%s", run_id, bucket, key)
    success = _read_success_record(s3, bucket, key)

    status = success.get("status")
    if status != "SUCCEEDED":
        logger.warning("run_id=%s | status='%s' (≠ SUCCEEDED) → no se lanza SageMaker", run_id, status)
        return {"run_id": run_id, "status": "skipped", "reason": f"status={status}"}

    return _create_with_fallback(sm, run_id, env)


def _env_from_os() -> dict:
    return {
        "ROLE_ARN": os.environ["SAGEMAKER_ROLE_ARN"],
        "IMAGE_URI": os.environ["IMAGE_URI"],
        "CPU_IMAGE_URI": os.environ["CPU_IMAGE_URI"],
        "INSTANCE_TYPE": os.environ["INSTANCE_TYPE"],
        "FALLBACK_INSTANCE_TYPE": os.environ["FALLBACK_INSTANCE_TYPE"],
        "USE_SPOT": os.environ.get("USE_SPOT", "true"),
        "CODE_S3_URI": os.environ["CODE_S3_URI"],
        "PROCESSED_BUCKET": os.environ["PROCESSED_BUCKET"],
        "MAX_RUNTIME": os.environ["MAX_RUNTIME"],
        "EPOCHS": os.environ["EPOCHS"],
        "REGION": os.environ.get("AWS_REGION", boto3.session.Session().region_name),
    }


def handler(event, context):
    env = _env_from_os()
    sm = _sagemaker_client()
    s3 = _s3_client()

    results = []
    for record in event.get("Records", []):
        outcome = _handle_record(sm, s3, record, env)
        if outcome is not None:
            results.append(outcome)

    summary = {"processed": len(results), "results": results}
    logger.info("Resumen de ejecución: %s", summary)
    return summary
