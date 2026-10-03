# -*- coding: utf-8 -*-
"""
SAVI v2 · Cloud — Lambda 1 (`savi-start-ec2`).

Disparada por eventos S3 PutObject en `savi-raw-<acct>/input/`. Construye el
`run_id`, etiqueta la instancia EC2 con `SaviInputKey`/`SaviRunId` y arranca
(o reactiva) el motor CPU según el estado actual de la instancia.

Ver CONTRACT.md → "Lambda 1 → EC2 (vía tags de la instancia)".
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3

logger = logging.getLogger("savi.lambda_start_ec2")
logger.setLevel(logging.INFO)

# Sufijos de archivo que disparan el pipeline (case-insensitive).
_VALID_SUFFIXES = (".csv", ".txt")
_INPUT_PREFIX = "input/"

# Estados EC2 que terminan la ejecución sin posibilidad de arranque.
_TERMINAL_STATES = {"terminated", "shutting-down"}

_WAITER_DELAY_SECONDS = 5
_WAITER_MAX_ATTEMPTS = 40


def _ec2_client():
    """Cliente EC2 perezoso (permite inyectar un stub en tests)."""
    return boto3.client("ec2")


def _sanitize_stem(stem: str, max_len: int = 20) -> str:
    """Sanitiza el stem del archivo a [a-z0-9-], recortado a `max_len` caracteres."""
    lowered = stem.lower()
    sanitized = re.sub(r"[^a-z0-9-]+", "-", lowered)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized[:max_len] or "run"


def build_run_id(key: str, now: datetime | None = None) -> str:
    """
    Genera el run_id exactamente como indica el contrato:
    `<UTC %Y%m%dT%H%M%SZ>-<stem del archivo sanitizado a [a-z0-9-], máx 20 chars>`.
    Ej: `20261002T210501Z-ameshousing`.
    """
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    filename = key.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f"{timestamp}-{_sanitize_stem(stem)}"


def _is_relevant_key(key: str) -> bool:
    """Ignora keys fuera de `input/` o sin sufijo .csv/.txt (case-insensitive)."""
    if not key.startswith(_INPUT_PREFIX):
        return False
    # Debe quedar un nombre de archivo real tras el prefijo (no un "directorio").
    if key == _INPUT_PREFIX:
        return False
    return key.lower().endswith(_VALID_SUFFIXES)


def _describe_instance_state(ec2, instance_id: str) -> str:
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    reservations = resp.get("Reservations", [])
    if not reservations or not reservations[0].get("Instances"):
        raise RuntimeError(f"No se encontró la instancia {instance_id}")
    return reservations[0]["Instances"][0]["State"]["Name"]


def _tag_instance(ec2, instance_id: str, input_key: str, raw_bucket: str, run_id: str) -> None:
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            {"Key": "SaviInputKey", "Value": f"s3://{raw_bucket}/{input_key}"},
            {"Key": "SaviRunId", "Value": run_id},
        ],
    )


def _handle_record(ec2, record: dict, instance_id: str, raw_bucket: str) -> dict | None:
    raw_key = record["s3"]["object"]["key"]
    key = unquote_plus(raw_key)

    if not _is_relevant_key(key):
        logger.info("Ignorando key fuera de alcance: %s", key)
        return None

    run_id = build_run_id(key)
    logger.info("run_id=%s | key=%s | iniciando procesamiento de evento S3", run_id, key)

    _tag_instance(ec2, instance_id, key, raw_bucket, run_id)
    logger.info("run_id=%s | tags SaviInputKey/SaviRunId escritos en %s", run_id, instance_id)

    state = _describe_instance_state(ec2, instance_id)
    logger.info("run_id=%s | estado actual de %s: %s", run_id, instance_id, state)

    if state == "stopped":
        ec2.start_instances(InstanceIds=[instance_id])
        logger.info("run_id=%s | instancia %s estaba 'stopped' → start_instances emitido", run_id, instance_id)
        action = "started"
    elif state == "stopping":
        logger.info("run_id=%s | instancia %s estaba 'stopping' → esperando 'instance_stopped'", run_id, instance_id)
        waiter = ec2.get_waiter("instance_stopped")
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": _WAITER_DELAY_SECONDS, "MaxAttempts": _WAITER_MAX_ATTEMPTS},
        )
        ec2.start_instances(InstanceIds=[instance_id])
        logger.info("run_id=%s | instancia %s detenida → start_instances emitido", run_id, instance_id)
        action = "waited_then_started"
    elif state in ("pending", "running"):
        logger.info(
            "run_id=%s | instancia %s ya está '%s' → sólo se actualizaron los tags "
            "(el script del EC2 re-chequea el tag antes de apagarse)",
            run_id, instance_id, state,
        )
        action = "tags_only"
    elif state in _TERMINAL_STATES:
        raise RuntimeError(
            f"run_id={run_id} | la instancia {instance_id} está en estado terminal '{state}'; "
            "no se puede arrancar el pipeline. Revisa la instancia manualmente."
        )
    else:
        raise RuntimeError(f"run_id={run_id} | estado EC2 desconocido: {state}")

    return {"run_id": run_id, "key": key, "instance_state": state, "action": action}


def handler(event, context):
    instance_id = os.environ["INSTANCE_ID"]
    raw_bucket = os.environ["RAW_BUCKET"]
    ec2 = _ec2_client()

    results = []
    for record in event.get("Records", []):
        outcome = _handle_record(ec2, record, instance_id, raw_bucket)
        if outcome is not None:
            results.append(outcome)

    summary = {"processed": len(results), "results": results}
    logger.info("Resumen de ejecución: %s", summary)
    return summary
