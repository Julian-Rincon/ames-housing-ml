#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAVI Cloud — guarda la clave de la API de Anthropic en SSM Parameter Store.

Lee la clave por stdin con `getpass` (nunca se imprime en pantalla, nunca se
acepta por argv para que no quede en el historial de la shell ni en `ps`) y la
guarda como SecureString en `/savi/anthropic_api_key` (sobrescribiendo si ya
existe). La Lambda `savi-api` la lee en frío vía `SAVI_SSM_KEY_PARAM`.

Uso:
    python infra/set_llm_key.py
    python infra/set_llm_key.py --region us-east-1 --param-name /savi/anthropic_api_key
"""
from __future__ import annotations

import argparse
import getpass
import logging
import sys
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]

REGION_DEFAULT = "us-east-1"
DEFAULT_PARAM_NAME = "/savi/anthropic_api_key"

log = logging.getLogger("savi.set_llm_key")


def put_ssm_secure_string(ssm, param_name: str, value: str) -> None:
    ssm.put_parameter(Name=param_name, Value=value, Type="SecureString", Overwrite=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarda la clave de Anthropic en SSM Parameter Store (SecureString)"
    )
    parser.add_argument("--region", default=REGION_DEFAULT)
    parser.add_argument("--param-name", default=DEFAULT_PARAM_NAME)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if boto3 is None:
        log.error("boto3 no está instalado")
        return 1

    api_key = getpass.getpass(f"Clave de Anthropic (se guardará en SSM {args.param_name}, no se mostrará): ")
    if not api_key or not api_key.strip():
        log.error("No se ingresó ninguna clave; se cancela")
        return 1

    try:
        session = boto3.Session(region_name=args.region)
        ssm = session.client("ssm", region_name=args.region)
        put_ssm_secure_string(ssm, args.param_name, api_key.strip())
    except ClientError as exc:
        log.error("No se pudo guardar el parámetro en SSM: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Error inesperado: %s", exc)
        return 1

    log.info("Clave guardada en SSM SecureString '%s' (región %s)", args.param_name, args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
