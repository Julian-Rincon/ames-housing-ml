#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAVI Cloud — Destructor de infraestructura (boto3), idempotente.

Elimina todo lo que crea `infra/deploy.py`, identificando recursos por nombre/tag
(no depende de que exista `deployment.json`, aunque lo usa si está disponible
para conocer el bucket/IDs más rápido). Pensado para no dejar nada corriendo
y así no gastar el presupuesto del AWS Academy Learner Lab.

Requiere `--yes` para ejecutar cualquier borrado real. `--keep-buckets` evita
tocar los buckets S3 (útil si querés conservar los datos/artefactos).

Uso:
    python infra/teardown.py --yes
    python infra/teardown.py --yes --keep-buckets --region us-east-1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]

INFRA_DIR = Path(__file__).resolve().parent
DEPLOYMENT_JSON = INFRA_DIR / "deployment.json"

REGION_DEFAULT = "us-east-1"
EC2_LOG_GROUP = "/savi/ec2-cpu-engine"
EC2_TAG_NAME = "savi-cpu-engine"
SG_NAME = "savi-cpu-engine-sg"
LAMBDA1_NAME = "savi-start-ec2"
LAMBDA2_NAME = "savi-start-sagemaker"
LAMBDA1_LOG_GROUP = f"/aws/lambda/{LAMBDA1_NAME}"
LAMBDA2_LOG_GROUP = f"/aws/lambda/{LAMBDA2_NAME}"

log = logging.getLogger("savi.teardown")


def bucket_names(account_id: str) -> tuple[str, str]:
    return f"savi-raw-{account_id}", f"savi-processed-{account_id}"


class SaviTeardown:
    def __init__(self, region: str, account_id: Optional[str], keep_buckets: bool) -> None:
        self.region = region
        self.keep_buckets = keep_buckets
        self._session = boto3.Session(region_name=region)
        self._clients: dict[str, Any] = {}
        self.account_id = account_id or self._client("sts").get_caller_identity()["Account"]

    def _client(self, service: str):
        if service not in self._clients:
            self._clients[service] = self._session.client(service, region_name=self.region)
        return self._clients[service]

    # -- notificaciones + lambdas --------------------------------------------- #

    def remove_bucket_notifications(self, bucket: str) -> None:
        s3 = self._client("s3")
        try:
            s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration={})
            log.info("Notificaciones removidas de '%s'", bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchBucket",):
                log.info("Bucket '%s' no existe, nada que limpiar en notificaciones", bucket)
            else:
                raise

    def delete_lambda(self, name: str) -> None:
        lam = self._client("lambda")
        try:
            lam.delete_function(FunctionName=name)
            log.info("Lambda '%s' eliminada", name)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                log.info("Lambda '%s' no existía", name)
            else:
                raise

    def delete_log_group(self, name: str) -> None:
        logs = self._client("logs")
        try:
            logs.delete_log_group(logGroupName=name)
            log.info("Log group '%s' eliminado", name)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                log.info("Log group '%s' no existía", name)
            else:
                raise

    # -- EC2 ------------------------------------------------------------------- #

    def terminate_ec2(self) -> None:
        ec2 = self._client("ec2")
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [EC2_TAG_NAME]},
                {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
            ]
        )
        instance_ids = [i["InstanceId"] for r in resp["Reservations"] for i in r["Instances"]]
        if not instance_ids:
            log.info("No hay instancias '%s' para terminar", EC2_TAG_NAME)
            return
        ec2.terminate_instances(InstanceIds=instance_ids)
        log.info("Terminando instancias: %s (esperando...)", instance_ids)
        ec2.get_waiter("instance_terminated").wait(InstanceIds=instance_ids)
        log.info("Instancias terminadas: %s", instance_ids)

    def delete_security_group(self) -> None:
        ec2 = self._client("ec2")
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
        if not vpcs:
            log.info("No hay VPC default, nada que hacer con el SG")
            return
        vpc_id = vpcs[0]["VpcId"]
        sgs = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc_id]}]
        )["SecurityGroups"]
        if not sgs:
            log.info("Security group '%s' no existe", SG_NAME)
            return
        sg_id = sgs[0]["GroupId"]
        try:
            ec2.delete_security_group(GroupId=sg_id)
            log.info("Security group '%s' (%s) eliminado", SG_NAME, sg_id)
        except ClientError as exc:
            log.warning(
                "No se pudo eliminar el SG %s (%s): %s — puede seguir en uso por una instancia reciente",
                SG_NAME, sg_id, exc,
            )

    # -- buckets ----------------------------------------------------------------- #

    def empty_and_delete_bucket(self, bucket: str) -> None:
        s3 = self._client("s3")
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket"):
                log.info("Bucket '%s' no existe", bucket)
                return
            raise

        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket):
            to_delete = [
                {"Key": v["Key"], "VersionId": v["VersionId"]}
                for v in page.get("Versions", []) + page.get("DeleteMarkers", [])
            ]
            if to_delete:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete, "Quiet": True})
                log.info("Borrados %s objetos/versiones de '%s'", len(to_delete), bucket)

        s3.delete_bucket(Bucket=bucket)
        log.info("Bucket '%s' eliminado", bucket)


def run(args: argparse.Namespace) -> None:
    teardown = SaviTeardown(region=args.region, account_id=args.account_id, keep_buckets=args.keep_buckets)
    raw_bucket, processed_bucket = bucket_names(teardown.account_id)

    log.info("== SAVI Cloud: destrucción de infraestructura (cuenta %s, región %s) ==", teardown.account_id, args.region)

    teardown.remove_bucket_notifications(raw_bucket)
    teardown.remove_bucket_notifications(processed_bucket)

    teardown.delete_lambda(LAMBDA1_NAME)
    teardown.delete_lambda(LAMBDA2_NAME)
    teardown.delete_log_group(LAMBDA1_LOG_GROUP)
    teardown.delete_log_group(LAMBDA2_LOG_GROUP)

    teardown.terminate_ec2()
    teardown.delete_security_group()
    teardown.delete_log_group(EC2_LOG_GROUP)

    if args.keep_buckets:
        log.info("--keep-buckets: se conservan %s y %s", raw_bucket, processed_bucket)
    else:
        teardown.empty_and_delete_bucket(raw_bucket)
        teardown.empty_and_delete_bucket(processed_bucket)

    log.info("Destrucción completa.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Destruye la infraestructura AWS de SAVI Cloud")
    parser.add_argument("--region", default=REGION_DEFAULT)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--keep-buckets", action="store_true", help="No borra los buckets S3")
    parser.add_argument("--yes", action="store_true", required=True, help="Confirmar el borrado (obligatorio)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        log.error("Destrucción abortada: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
