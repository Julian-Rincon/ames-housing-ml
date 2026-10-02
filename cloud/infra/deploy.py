#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAVI Cloud — Desplegador de infraestructura (boto3), idempotente.

Lee CONTRACT.md como fuente de verdad. Crea/actualiza buckets S3, grupos de logs
de CloudWatch, el security group del motor EC2, la propia instancia EC2 y las
dos Lambdas (`savi-start-ec2`, `savi-start-sagemaker`), junto con sus permisos
de invocación y la configuración de notificaciones S3.

Diseñado para correr en AWS Academy Learner Lab: usa `LabRole`/`LabInstanceProfile`
por defecto (no se pueden crear roles IAM nuevos salvo que se pase --create-roles,
lo cual casi seguro fallará en Learner Lab por permisos).

Uso típico:
    python infra/deploy.py --region us-east-1
    python infra/deploy.py --dry-run --account-id 006840014780
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from string import Template
from typing import Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, WaiterError
except ImportError:  # pragma: no cover - boto3 es requisito duro para correr de verdad
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]
    WaiterError = Exception  # type: ignore[assignment,misc]


# --------------------------------------------------------------------------- #
# Constantes y rutas (ver CONTRACT.md)
# --------------------------------------------------------------------------- #

INFRA_DIR = Path(__file__).resolve().parent
CLOUD_ROOT = INFRA_DIR.parent
LAMBDAS_DIR = CLOUD_ROOT / "lambdas"
EC2_DIR = CLOUD_ROOT / "ec2"

REGION_DEFAULT = "us-east-1"
AMI_SSM_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
LOG_RETENTION_DAYS = 14

EC2_LOG_GROUP = "/savi/ec2-cpu-engine"
EC2_TAG_NAME = "savi-cpu-engine"
SG_NAME = "savi-cpu-engine-sg"

LAMBDA1_NAME = "savi-start-ec2"
LAMBDA2_NAME = "savi-start-sagemaker"
LAMBDA1_FILE = LAMBDAS_DIR / "lambda_start_ec2.py"
LAMBDA2_FILE = LAMBDAS_DIR / "lambda_start_sagemaker.py"
LAMBDA1_HANDLER = "lambda_start_ec2.handler"
LAMBDA2_HANDLER = "lambda_start_sagemaker.handler"
LAMBDA1_LOG_GROUP = f"/aws/lambda/{LAMBDA1_NAME}"
LAMBDA2_LOG_GROUP = f"/aws/lambda/{LAMBDA2_NAME}"

IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker"
)
CPU_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker"
)

SAGEMAKER_INSTANCE_TYPE = "ml.g4dn.xlarge"
SAGEMAKER_FALLBACK_INSTANCE_TYPE = "ml.m5.xlarge"

DEPLOYMENT_JSON = INFRA_DIR / "deployment.json"

log = logging.getLogger("savi.deploy")


# --------------------------------------------------------------------------- #
# Funciones puras (testeables sin red / sin AWS)
# --------------------------------------------------------------------------- #

def build_sourcedir_tar_bytes(cloud_root: Path) -> bytes:
    """
    Construye en memoria `sourcedir.tar.gz` con `savi_gpu_sagemaker.py` y
    `utils.py` en la RAÍZ del tar (requisito del SDK de SageMaker para
    `sagemaker_submit_directory`).
    """
    gpu_script = cloud_root / "savi_gpu_sagemaker.py"
    utils_py = cloud_root / "utils.py"
    for p in (gpu_script, utils_py):
        if not p.is_file():
            raise FileNotFoundError(f"No se encontró {p} (requerido para sourcedir.tar.gz)")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(gpu_script, arcname="savi_gpu_sagemaker.py")
        tar.add(utils_py, arcname="utils.py")
    return buf.getvalue()


def zip_single_file(file_path: Path, arcname: Optional[str] = None) -> bytes:
    """Empaqueta un único archivo .py en un zip válido para Lambda (código en memoria)."""
    if not file_path.is_file():
        raise FileNotFoundError(f"No se encontró el módulo Lambda: {file_path}")
    arcname = arcname or file_path.name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=arcname)
    return buf.getvalue()


def render_user_data(template_path: Path, replacements: dict[str, str]) -> str:
    """
    Sustituye los placeholders __RAW_BUCKET__, __PROCESSED_BUCKET__, __REGION__,
    __LOG_GROUP__ (u otros que vengan en `replacements`) en el template de user-data.
    Sustitución simple de texto (los placeholders usan `__NOMBRE__`, no sintaxis
    `string.Template`), por lo que se hace con `str.replace` en cadena.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"No se encontró el template de user-data: {template_path}")
    text = template_path.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    missing = [ph for ph in replacements if ph in text]
    if missing:
        log.warning("Placeholders no reemplazados en user-data (¿cambiaron nombres?): %s", missing)
    return text


def build_raw_notification_config(lambda1_arn: str) -> dict:
    """Configuración de notificación del bucket raw → Lambda 1, prefix `input/`."""
    return {
        "LambdaFunctionConfigurations": [
            {
                "Id": "savi-input-triggers-start-ec2",
                "LambdaFunctionArn": lambda1_arn,
                "Events": ["s3:ObjectCreated:*"],
                "Filter": {
                    "Key": {
                        "FilterRules": [
                            {"Name": "prefix", "Value": "input/"},
                        ]
                    }
                },
            }
        ]
    }


def build_processed_notification_config(lambda2_arn: str) -> dict:
    """Configuración de notificación del bucket processed → Lambda 2, prefix/suffix de _SUCCESS.json."""
    return {
        "LambdaFunctionConfigurations": [
            {
                "Id": "savi-success-triggers-start-sagemaker",
                "LambdaFunctionArn": lambda2_arn,
                "Events": ["s3:ObjectCreated:*"],
                "Filter": {
                    "Key": {
                        "FilterRules": [
                            {"Name": "prefix", "Value": "runs/"},
                            {"Name": "suffix", "Value": "_SUCCESS.json"},
                        ]
                    }
                },
            }
        ]
    }


def processed_lifecycle_config() -> dict:
    """Reglas de ciclo de vida del bucket processed: abort multipart 1d, expirar runs/ a 30d."""
    return {
        "Rules": [
            {
                "ID": "savi-abort-incomplete-multipart-1d",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            },
            {
                "ID": "savi-expire-runs-30d",
                "Status": "Enabled",
                "Filter": {"Prefix": "runs/"},
                "Expiration": {"Days": 30},
            },
        ]
    }


def lab_role_arn(account_id: str) -> str:
    return f"arn:aws:iam::{account_id}:role/LabRole"


def bucket_names(account_id: str) -> tuple[str, str]:
    return f"savi-raw-{account_id}", f"savi-processed-{account_id}"


# --------------------------------------------------------------------------- #
# Deployer
# --------------------------------------------------------------------------- #

class SaviDeployer:
    def __init__(
        self,
        region: str,
        account_id: Optional[str],
        dry_run: bool,
        data_dir: Path,
        skip_upload_data: bool,
        create_roles: bool,
    ) -> None:
        self.region = region
        self.dry_run = dry_run
        self.data_dir = data_dir
        self.skip_upload_data = skip_upload_data
        self.create_roles = create_roles
        self._account_id_override = account_id
        self._session = None
        self._clients: dict[str, Any] = {}
        self.state: dict[str, Any] = {"region": region}

    # -- infraestructura de soporte ---------------------------------------- #

    def _client(self, service: str):
        if self.dry_run:
            raise RuntimeError(
                f"intento de crear cliente boto3 ({service}) en modo --dry-run: "
                "esto nunca debe pasar, es un bug del deployer"
            )
        if service not in self._clients:
            if self._session is None:
                self._session = boto3.Session(region_name=self.region)
            self._clients[service] = self._session.client(service, region_name=self.region)
        return self._clients[service]

    def resolve_account_id(self) -> str:
        if self._account_id_override:
            log.info("Usando --account-id provisto: %s (no se llama a STS)", self._account_id_override)
            return self._account_id_override
        if self.dry_run:
            raise SystemExit(
                "--dry-run sin credenciales requiere --account-id (no se puede llamar a STS)."
            )
        sts = self._client("sts")
        account_id = sts.get_caller_identity()["Account"]
        log.info("Cuenta AWS detectada vía STS: %s", account_id)
        return account_id

    # -- paso b: buckets ---------------------------------------------------- #

    def ensure_buckets(self, raw_bucket: str, processed_bucket: str) -> None:
        for bucket in (raw_bucket, processed_bucket):
            if self.dry_run:
                log.info(
                    "[dry-run] crear/asegurar bucket '%s' (PublicAccessBlock=todo true, "
                    "SSE-S3, BucketOwnerEnforced)",
                    bucket,
                )
                continue
            s3 = self._client("s3")
            exists = True
            try:
                s3.head_bucket(Bucket=bucket)
                log.info("Bucket '%s' ya existe", bucket)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code in ("404", "NoSuchBucket") or status == 404:
                    exists = False
                else:
                    raise
            if not exists:
                kwargs: dict[str, Any] = {"Bucket": bucket}
                if self.region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
                s3.create_bucket(**kwargs)
                log.info("Bucket '%s' creado", bucket)

            s3.put_public_access_block(
                Bucket=bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            s3.put_bucket_ownership_controls(
                Bucket=bucket,
                OwnershipControls={"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]},
            )
            s3.put_bucket_encryption(
                Bucket=bucket,
                ServerSideEncryptionConfiguration={
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                            "BucketKeyEnabled": True,
                        }
                    ]
                },
            )
            log.info("Bucket '%s': PublicAccessBlock + BucketOwnerEnforced + SSE-S3 aplicados", bucket)

        if self.dry_run:
            log.info("[dry-run] aplicar lifecycle en '%s': abort multipart 1d, expirar runs/ a 30d", processed_bucket)
            return
        s3 = self._client("s3")
        s3.put_bucket_lifecycle_configuration(
            Bucket=processed_bucket, LifecycleConfiguration=processed_lifecycle_config()
        )
        log.info("Lifecycle aplicado en '%s'", processed_bucket)

    # -- paso c: código y datos de referencia -------------------------------- #

    def upload_code_and_reference(self, raw_bucket: str, processed_bucket: str) -> None:
        code_cpu_files = [
            (CLOUD_ROOT / "utils.py", "code/cpu/utils.py"),
            (CLOUD_ROOT / "savi_cpu_pipeline.py", "code/cpu/savi_cpu_pipeline.py"),
            (CLOUD_ROOT / "requirements-ec2.txt", "code/cpu/requirements-ec2.txt"),
            (EC2_DIR / "run_pipeline.sh", "code/cpu/run_pipeline.sh"),
        ]
        for local_path, key in code_cpu_files:
            self._upload_file_or_warn(local_path, raw_bucket, key)

        if self.dry_run:
            log.info(
                "[dry-run] construir sourcedir.tar.gz (savi_gpu_sagemaker.py + utils.py en raíz) "
                "y subir a s3://%s/code/sourcedir.tar.gz",
                processed_bucket,
            )
        else:
            try:
                tar_bytes = build_sourcedir_tar_bytes(CLOUD_ROOT)
            except FileNotFoundError as exc:
                log.warning("No se pudo construir sourcedir.tar.gz: %s", exc)
            else:
                s3 = self._client("s3")
                s3.put_object(Bucket=processed_bucket, Key="code/sourcedir.tar.gz", Body=tar_bytes)
                log.info("sourcedir.tar.gz subido a s3://%s/code/sourcedir.tar.gz", processed_bucket)

        if self.skip_upload_data:
            log.info("--skip-upload-data: se omite la subida de archivos de referencia")
            return

        reference_dir = self.data_dir / "reference"
        if not reference_dir.is_dir():
            log.warning("No existe el directorio de referencia: %s (se omite)", reference_dir)
            return
        for ref_file in sorted(reference_dir.iterdir()):
            if ref_file.is_file():
                self._upload_file_or_warn(ref_file, raw_bucket, f"reference/{ref_file.name}")

    def _upload_file_or_warn(self, local_path: Path, bucket: str, key: str) -> None:
        if not local_path.is_file():
            log.warning(
                "Archivo no encontrado (puede que otro agente aún no lo haya escrito): %s "
                "→ se omite subida de s3://%s/%s",
                local_path, bucket, key,
            )
            return
        if self.dry_run:
            log.info("[dry-run] subir %s → s3://%s/%s", local_path, bucket, key)
            return
        s3 = self._client("s3")
        s3.upload_file(str(local_path), bucket, key)
        log.info("Subido %s → s3://%s/%s", local_path, bucket, key)

    # -- paso d: log groups --------------------------------------------------- #

    def ensure_log_groups(self) -> None:
        log_groups = [EC2_LOG_GROUP, LAMBDA1_LOG_GROUP, LAMBDA2_LOG_GROUP]
        if self.dry_run:
            for lg in log_groups:
                log.info("[dry-run] asegurar log group '%s' con retención %sd", lg, LOG_RETENTION_DAYS)
            return
        logs = self._client("logs")
        for lg in log_groups:
            try:
                logs.create_log_group(logGroupName=lg)
                log.info("Log group '%s' creado", lg)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
                    log.info("Log group '%s' ya existe", lg)
                else:
                    raise
            logs.put_retention_policy(logGroupName=lg, retentionInDays=LOG_RETENTION_DAYS)

    # -- paso e: security group ----------------------------------------------- #

    def ensure_security_group(self) -> str:
        if self.dry_run:
            log.info(
                "[dry-run] asegurar SG '%s' en la VPC default: sin ingress, egress sólo TCP 443 0.0.0.0/0",
                SG_NAME,
            )
            return "sg-DRYRUN"

        ec2 = self._client("ec2")
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
        if not vpcs:
            raise RuntimeError("No se encontró una VPC default en la región; no se puede crear el SG")
        vpc_id = vpcs[0]["VpcId"]

        existing = ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [SG_NAME]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )["SecurityGroups"]
        if existing:
            sg_id = existing[0]["GroupId"]
            log.info("Security group '%s' ya existe: %s", SG_NAME, sg_id)
        else:
            resp = ec2.create_security_group(
                GroupName=SG_NAME,
                Description="SAVI CPU engine: sin ingress, egress solo HTTPS",
                VpcId=vpc_id,
                TagSpecifications=[
                    {"ResourceType": "security-group", "Tags": [{"Key": "Project", "Value": "SAVI"}]}
                ],
            )
            sg_id = resp["GroupId"]
            log.info("Security group '%s' creado: %s", SG_NAME, sg_id)

        # Revocar el egress "allow-all" por defecto (sólo existe en SGs recién creados).
        try:
            ec2.revoke_security_group_egress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "-1",
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    }
                ],
            )
            log.info("Egress allow-all por defecto revocado en %s", sg_id)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "InvalidPermission.NotFound":
                log.info("El SG %s ya no tenía el egress allow-all por defecto", sg_id)
            else:
                raise

        try:
            ec2.authorize_security_group_egress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 443,
                        "ToPort": 443,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    }
                ],
            )
            log.info("Egress TCP 443 0.0.0.0/0 autorizado en %s", sg_id)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "InvalidPermission.Duplicate":
                log.info("El SG %s ya tenía la regla de egress TCP 443", sg_id)
            else:
                raise

        self.state["vpc_id"] = vpc_id
        return sg_id

    # -- paso f: instancia EC2 ------------------------------------------------ #

    def ensure_ec2_instance(self, sg_id: str, raw_bucket: str, processed_bucket: str) -> str:
        if self.dry_run:
            log.info(
                "[dry-run] buscar instancia tag Name=%s no terminada; si no existe, run_instances "
                "t3.medium AMI=SSM:%s IamInstanceProfile=LabInstanceProfile SG=%s subred default pública, "
                "EBS 20GB gp3 cifrado, IMDSv2 requerido + tags en metadata, InstanceInitiatedShutdownBehavior=stop",
                EC2_TAG_NAME, AMI_SSM_PARAM, sg_id,
            )
            return "i-DRYRUN"

        ec2 = self._client("ec2")
        non_terminal = ["pending", "running", "shutting-down", "stopping", "stopped"]
        existing = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [EC2_TAG_NAME]},
                {"Name": "instance-state-name", "Values": non_terminal},
            ]
        )
        instances = [i for r in existing["Reservations"] for i in r["Instances"]]
        if instances:
            instance_id = instances[0]["InstanceId"]
            log.info(
                "Instancia '%s' ya existe: %s (estado=%s). Si cambiaste user-data.sh.tpl, NO se reaplica "
                "(sólo corre en el primer boot); recreá la instancia manualmente si hace falta.",
                EC2_TAG_NAME, instance_id, instances[0]["State"]["Name"],
            )
            return instance_id

        ssm = self._client("ssm")
        ami_id = ssm.get_parameter(Name=AMI_SSM_PARAM)["Parameter"]["Value"]
        log.info("AMI resuelta vía SSM: %s", ami_id)

        # No todas las AZ ofrecen t3.medium (p.ej. us-east-1e) → filtrar por oferta real
        offered = {
            o["Location"]
            for o in ec2.describe_instance_type_offerings(
                LocationType="availability-zone",
                Filters=[{"Name": "instance-type", "Values": ["t3.medium"]}],
            )["InstanceTypeOfferings"]
        }
        subnets = [
            s for s in ec2.describe_subnets(
                Filters=[{"Name": "default-for-az", "Values": ["true"]}]
            )["Subnets"]
            if s["AvailabilityZone"] in offered
        ]
        if not subnets:
            raise RuntimeError("No hay subred default en una AZ que ofrezca t3.medium")
        subnets.sort(key=lambda s: s["AvailabilityZone"])
        subnet_id = subnets[0]["SubnetId"]
        log.info("Subred elegida: %s (%s)", subnet_id, subnets[0]["AvailabilityZone"])

        template_path = EC2_DIR / "user-data.sh.tpl"
        try:
            user_data = render_user_data(
                template_path,
                {
                    "__RAW_BUCKET__": raw_bucket,
                    "__PROCESSED_BUCKET__": processed_bucket,
                    "__REGION__": self.region,
                    "__LOG_GROUP__": EC2_LOG_GROUP,
                },
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Falta ec2/user-data.sh.tpl (otro agente debería haberlo escrito): {exc}"
            ) from exc

        resp = ec2.run_instances(
            ImageId=ami_id,
            InstanceType="t3.medium",
            MinCount=1,
            MaxCount=1,
            IamInstanceProfile={"Name": "LabInstanceProfile"},
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": subnet_id,
                    "Groups": [sg_id],
                    "AssociatePublicIpAddress": True,
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": 20,
                        "VolumeType": "gp3",
                        "Encrypted": True,
                        "DeleteOnTermination": True,
                    },
                }
            ],
            MetadataOptions={
                "HttpTokens": "required",
                "InstanceMetadataTags": "enabled",
            },
            InstanceInitiatedShutdownBehavior="stop",
            UserData=user_data,
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": EC2_TAG_NAME},
                        {"Key": "Project", "Value": "SAVI"},
                    ],
                }
            ],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        log.info(
            "Instancia '%s' lanzada: %s (NO se espera a que termine el bootstrap, se apaga sola)",
            EC2_TAG_NAME, instance_id,
        )
        return instance_id

    # -- paso g: lambdas ------------------------------------------------------- #

    def ensure_lambda(
        self,
        name: str,
        file_path: Path,
        handler: str,
        timeout: int,
        memory: int,
        env_vars: dict[str, str],
        role_arn: str,
    ) -> str:
        if self.dry_run:
            log.info(
                "[dry-run] crear/actualizar Lambda '%s' (handler=%s, timeout=%ss, memoria=%sMB, env=%s)",
                name, handler, timeout, memory, sorted(env_vars),
            )
            return f"arn:aws:lambda:{self.region}:DRYRUN:function:{name}"

        if not file_path.is_file():
            raise RuntimeError(f"Falta el módulo Lambda {file_path} (otro agente debería haberlo escrito)")

        lam = self._client("lambda")
        zip_bytes = zip_single_file(file_path)

        exists = True
        try:
            lam.get_function(FunctionName=name)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                exists = False
            else:
                raise

        if exists:
            lam.update_function_code(FunctionName=name, ZipFile=zip_bytes)
            self._wait_lambda(lam, name, "function_updated")
            lam.update_function_configuration(
                FunctionName=name,
                Runtime="python3.12",
                Role=role_arn,
                Handler=handler,
                Timeout=timeout,
                MemorySize=memory,
                Environment={"Variables": env_vars},
            )
            self._wait_lambda(lam, name, "function_updated")
            log.info("Lambda '%s' actualizada", name)
        else:
            self._create_lambda_with_retry(
                lam, name, zip_bytes, handler, timeout, memory, env_vars, role_arn
            )
            self._wait_lambda(lam, name, "function_active_v2")
            log.info("Lambda '%s' creada", name)

        return lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]

    @staticmethod
    def _wait_lambda(lam, name: str, waiter_name: str) -> None:
        try:
            lam.get_waiter(waiter_name).wait(FunctionName=name)
        except WaiterError as exc:
            log.warning("Waiter '%s' no confirmó estado estable para '%s': %s", waiter_name, name, exc)

    def _create_lambda_with_retry(
        self, lam, name: str, zip_bytes: bytes, handler: str, timeout: int,
        memory: int, env_vars: dict[str, str], role_arn: str,
        max_attempts: int = 6, base_delay: float = 5.0,
    ) -> None:
        """
        Reintenta create_function ante InvalidParameterValueException con mensaje
        "cannot be assumed" — eventual consistency típica tras crear/usar un rol IAM.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                lam.create_function(
                    FunctionName=name,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler=handler,
                    Code={"ZipFile": zip_bytes},
                    Timeout=timeout,
                    MemorySize=memory,
                    Environment={"Variables": env_vars},
                    Publish=False,
                )
                return
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code")
                msg = exc.response.get("Error", {}).get("Message", "")
                retryable = code == "InvalidParameterValueException" and "cannot be assumed" in msg
                if retryable and attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning(
                        "create_function('%s') falló por propagación IAM (intento %s/%s), "
                        "reintentando en %.0fs: %s",
                        name, attempt, max_attempts, delay, msg,
                    )
                    time.sleep(delay)
                    continue
                raise

    # -- paso h: permisos + notificaciones ------------------------------------ #

    def ensure_invoke_permission(self, function_name: str, bucket: str, account_id: str) -> None:
        if self.dry_run:
            log.info(
                "[dry-run] add_permission: s3.amazonaws.com puede invocar '%s' desde bucket '%s'",
                function_name, bucket,
            )
            return
        lam = self._client("lambda")
        try:
            lam.add_permission(
                FunctionName=function_name,
                StatementId=f"AllowS3Invoke-{bucket}",
                Action="lambda:InvokeFunction",
                Principal="s3.amazonaws.com",
                # Sin SourceAccount: en AWS Academy Learner Lab esa condición hace que S3
                # nunca logre invocar la Lambda (verificado 2026-10-02). El bucket es
                # nuestro y su ARN ya restringe el origen.
                SourceArn=f"arn:aws:s3:::{bucket}",
            )
            log.info("Permiso de invocación S3→'%s' agregado", function_name)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceConflictException":
                log.info("El permiso de invocación S3→'%s' ya existía", function_name)
            else:
                raise

    def ensure_bucket_notification(self, bucket: str, config: dict) -> None:
        if self.dry_run:
            log.info("[dry-run] put_bucket_notification_configuration en '%s': %s", bucket, json.dumps(config))
            return
        s3 = self._client("s3")
        s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=config)
        log.info("Notificación S3 configurada en '%s'", bucket)


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> dict[str, Any]:
    deployer = SaviDeployer(
        region=args.region,
        account_id=args.account_id,
        dry_run=args.dry_run,
        data_dir=Path(args.data_dir),
        skip_upload_data=args.skip_upload_data,
        create_roles=args.create_roles,
    )

    log.info("== SAVI Cloud: despliegue de infraestructura ==")
    log.info("Región: %s | dry-run: %s", args.region, args.dry_run)

    account_id = deployer.resolve_account_id()
    raw_bucket, processed_bucket = bucket_names(account_id)
    role_arn = lab_role_arn(account_id)

    log.info("Cuenta: %s | bucket raw: %s | bucket processed: %s", account_id, raw_bucket, processed_bucket)
    log.info("Rol usado para EC2/Lambdas: %s", role_arn)
    if args.create_roles:
        log.warning(
            "--create-roles fue pasado: se intentará crear roles least-privilege desde infra/iam/*.json. "
            "En AWS Academy Learner Lab esto casi seguro falla por iam:CreateRole denegado; "
            "en ese caso se recomienda NO usar esta bandera y confiar en LabRole/LabInstanceProfile."
        )

    deployer.ensure_buckets(raw_bucket, processed_bucket)
    deployer.upload_code_and_reference(raw_bucket, processed_bucket)
    deployer.ensure_log_groups()
    sg_id = deployer.ensure_security_group()
    instance_id = deployer.ensure_ec2_instance(sg_id, raw_bucket, processed_bucket)

    lambda1_env = {"INSTANCE_ID": instance_id, "RAW_BUCKET": raw_bucket}
    lambda2_env = {
        "SAGEMAKER_ROLE_ARN": role_arn,
        "IMAGE_URI": IMAGE_URI,
        "CPU_IMAGE_URI": CPU_IMAGE_URI,
        "INSTANCE_TYPE": SAGEMAKER_INSTANCE_TYPE,
        "FALLBACK_INSTANCE_TYPE": SAGEMAKER_FALLBACK_INSTANCE_TYPE,
        "USE_SPOT": "true",
        "CODE_S3_URI": f"s3://{processed_bucket}/code/sourcedir.tar.gz",
        "PROCESSED_BUCKET": processed_bucket,
        "MAX_RUNTIME": "3600",
        "EPOCHS": "150",
    }

    lambda1_arn = deployer.ensure_lambda(
        LAMBDA1_NAME, LAMBDA1_FILE, LAMBDA1_HANDLER, timeout=300, memory=128,
        env_vars=lambda1_env, role_arn=role_arn,
    )
    lambda2_arn = deployer.ensure_lambda(
        LAMBDA2_NAME, LAMBDA2_FILE, LAMBDA2_HANDLER, timeout=60, memory=256,
        env_vars=lambda2_env, role_arn=role_arn,
    )

    deployer.ensure_invoke_permission(LAMBDA1_NAME, raw_bucket, account_id)
    deployer.ensure_invoke_permission(LAMBDA2_NAME, processed_bucket, account_id)
    deployer.ensure_bucket_notification(raw_bucket, build_raw_notification_config(lambda1_arn))
    deployer.ensure_bucket_notification(processed_bucket, build_processed_notification_config(lambda2_arn))

    deployment_state = {
        "account_id": account_id,
        "region": args.region,
        "dry_run": args.dry_run,
        "raw_bucket": raw_bucket,
        "processed_bucket": processed_bucket,
        "role_arn": role_arn,
        "security_group_id": sg_id,
        "instance_id": instance_id,
        "lambda1_name": LAMBDA1_NAME,
        "lambda1_arn": lambda1_arn,
        "lambda2_name": LAMBDA2_NAME,
        "lambda2_arn": lambda2_arn,
        "log_groups": [EC2_LOG_GROUP, LAMBDA1_LOG_GROUP, LAMBDA2_LOG_GROUP],
        "sample_trigger_command": (
            f"aws s3 cp data/input/AmesHousing.txt "
            f"s3://{raw_bucket}/input/AmesHousing.txt"
        ),
    }
    return deployment_state


def write_summary(state: dict[str, Any], dry_run: bool) -> None:
    if not dry_run:
        DEPLOYMENT_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info("Estado del despliegue escrito en %s", DEPLOYMENT_JSON)

    print("\n" + "=" * 70)
    print("SAVI Cloud — resumen de despliegue" + (" (DRY RUN, nada fue creado)" if dry_run else ""))
    print("=" * 70)
    for k in (
        "account_id", "region", "raw_bucket", "processed_bucket", "role_arn",
        "security_group_id", "instance_id", "lambda1_arn", "lambda2_arn",
    ):
        print(f"  {k}: {state.get(k)}")
    print("\nPara disparar el pipeline manualmente:")
    print(f"  {state['sample_trigger_command']}")
    print("=" * 70 + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Despliega la infraestructura AWS de SAVI Cloud")
    parser.add_argument("--region", default=REGION_DEFAULT, help="Región AWS (default: us-east-1)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Imprime el plan completo, no hace ninguna llamada mutante (ni de lectura) a AWS",
    )
    parser.add_argument(
        "--account-id", default=None,
        help="Forzar el account id (requerido para --dry-run sin credenciales configuradas)",
    )
    parser.add_argument(
        "--data-dir", default=str(CLOUD_ROOT / "data"),
        help="Directorio con input/AmesHousing.txt y reference/* (default: data/)",
    )
    parser.add_argument(
        "--skip-upload-data", action="store_true",
        help="No sube los archivos de reference/ (el código sí se sube siempre)",
    )
    parser.add_argument(
        "--create-roles", action="store_true",
        help="Intenta crear roles IAM least-privilege desde infra/iam/*.json "
        "(default OFF: Learner Lab deniega iam:CreateRole, se usa LabRole/LabInstanceProfile)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        state = run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - reporte claro al usuario final
        log.error("Despliegue abortado: %s", exc)
        return 1
    write_summary(state, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
