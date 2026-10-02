#!/bin/bash
# SAVI v2 · run_pipeline.sh — corre en CADA boot vía systemd (savi-pipeline.service).
#
# Cost guard estricto: pase lo que pase (éxito, error, timeout, instancia sin
# tags) la instancia TERMINA apagada, salvo tag SaviNoShutdown=true. Por eso
# NO se usa `set -e`: un `exit` temprano por un comando fallido saltaría pasos
# de limpieza/log antes de llegar al trap. Los errores se manejan a mano y el
# trap EXIT es la única vía de apagado, así se dispare desde donde se dispare.
set -uo pipefail

# ── Rutas/overrides (permiten correr este script sin privilegios en tests) ──
SAVI_ROOT="${SAVI_ROOT:-/opt/savi}"
SAVI_ENV_FILE="${SAVI_ENV_FILE:-/etc/savi.env}"
SAVI_LOG_DIR="${SAVI_LOG_DIR:-/var/log/savi}"
SAVI_PYTHON_BIN="${SAVI_PYTHON_BIN:-${SAVI_ROOT}/venv/bin/python}"
SAVI_TIMEOUT_SECS="${SAVI_TIMEOUT_SECS:-3600}"
SAVI_MAX_LOOPS="${SAVI_MAX_LOOPS:-3}"
SAVI_IMDS_BASE="${SAVI_IMDS_BASE:-http://169.254.169.254}"
# Comando de apagado, overridable a propósito para pruebas: nunca se debe
# invocar el `shutdown` real fuera de una EC2 (ver tests/test_ec2_scripts.py).
# shellcheck disable=SC2206
SAVI_SHUTDOWN_CMD=(${SAVI_SHUTDOWN_CMD:-shutdown -h now})

mkdir -p "$SAVI_LOG_DIR"
LOG_FILE="${SAVI_LOG_DIR}/pipeline.log"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"; }

# shellcheck disable=SC1090
if [ -f "$SAVI_ENV_FILE" ]; then
  source "$SAVI_ENV_FILE"
fi
: "${RAW_BUCKET:?falta RAW_BUCKET (revisar ${SAVI_ENV_FILE})}"
: "${PROCESSED_BUCKET:?falta PROCESSED_BUCKET (revisar ${SAVI_ENV_FILE})}"
: "${REGION:?falta REGION (revisar ${SAVI_ENV_FILE})}"

RUN_ID_FOR_TRAP=""

imds_token() {
  curl -sS -X PUT "${SAVI_IMDS_BASE}/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" --max-time 5
}

# Lee un tag de instancia vía IMDSv2. 404 (tag ausente) o cualquier respuesta
# != 200 → devuelve rc=1 y stdout vacío (el llamador decide qué hacer).
get_tag() {
  local name="$1" token resp http_code body
  token="$(imds_token)"
  resp="$(curl -sS -o - -w $'\n%{http_code}' \
    -H "X-aws-ec2-metadata-token: ${token}" \
    "${SAVI_IMDS_BASE}/latest/meta-data/tags/instance/${name}" --max-time 5)"
  http_code="${resp##*$'\n'}"
  body="${resp%$'\n'*}"
  if [ "$http_code" = "200" ]; then
    printf '%s' "$body"
    return 0
  fi
  return 1
}

# ── Cost guard final: se ejecuta SIEMPRE al salir del script (EXIT trap) ──
cleanup() {
  local rc=$?
  log "cleanup: rc=${rc} run_id=${RUN_ID_FOR_TRAP:-<ninguno>}"

  if [ -n "${RUN_ID_FOR_TRAP:-}" ] && [ -f "$LOG_FILE" ]; then
    aws s3 cp "$LOG_FILE" "s3://${PROCESSED_BUCKET}/runs/${RUN_ID_FOR_TRAP}/logs/ec2_pipeline.log" --region "$REGION" \
      || log "ADVERTENCIA: no se pudo subir el log final a S3"
  fi

  local no_shutdown=""
  no_shutdown="$(get_tag SaviNoShutdown || true)"
  if [ "$no_shutdown" = "true" ]; then
    log "SaviNoShutdown=true → NO se apaga la instancia"
  else
    log "Apagando instancia (cost guard)"
    "${SAVI_SHUTDOWN_CMD[@]}" || true
  fi
  exit "$rc"
}
trap cleanup EXIT

# ── Procesa un run_id: idempotencia, sync de código, pipeline, subida de log ──
process_run() {
  local run_id="$1" input_key="$2" rc=0
  RUN_ID_FOR_TRAP="$run_id"
  log "=== procesando run_id=${run_id} input=${input_key} ==="

  if aws s3api head-object --bucket "$PROCESSED_BUCKET" --key "runs/${run_id}/_SUCCESS.json" --region "$REGION" >/dev/null 2>&1; then
    log "run_id=${run_id} ya tiene _SUCCESS.json → se omite (idempotencia)"
    return 0
  fi

  log "sincronizando código desde s3://${RAW_BUCKET}/code/cpu/"
  if ! aws s3 sync "s3://${RAW_BUCKET}/code/cpu/" "${SAVI_ROOT}/code/" --region "$REGION"; then
    log "ADVERTENCIA: aws s3 sync de código falló, se intenta igual con lo que haya en ${SAVI_ROOT}/code/"
  fi

  log "ejecutando pipeline CPU (timeout ${SAVI_TIMEOUT_SECS}s)"
  {
    timeout "$SAVI_TIMEOUT_SECS" "$SAVI_PYTHON_BIN" "${SAVI_ROOT}/code/savi_cpu_pipeline.py" \
      --input "$input_key" \
      --reference "s3://${RAW_BUCKET}/reference/" \
      --output "s3://${PROCESSED_BUCKET}/runs/${run_id}/" \
      --run-id "$run_id"
  } 2>&1 | while IFS= read -r line; do
      printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line"
    done | tee -a "$LOG_FILE"
  rc="${PIPESTATUS[0]}"
  if [ "$rc" -eq 0 ]; then
    log "pipeline CPU terminó OK (run_id=${run_id})"
  else
    log "pipeline CPU terminó con rc=${rc} (run_id=${run_id})"
  fi

  log "subiendo log a s3://${PROCESSED_BUCKET}/runs/${run_id}/logs/ec2_pipeline.log"
  aws s3 cp "$LOG_FILE" "s3://${PROCESSED_BUCKET}/runs/${run_id}/logs/ec2_pipeline.log" --region "$REGION" \
    || log "ADVERTENCIA: no se pudo subir el log de este run"

  return "$rc"
}

# ── Bucle principal: hasta SAVI_MAX_LOOPS veces, por si durante el procesamiento
#    aparece un SaviRunId nuevo (Lambda 1 reusa la instancia sin pasar por un
#    stop/start). El tag se relee en cada vuelta. ──
loop_count=0
last_run_id=""
while [ "$loop_count" -lt "$SAVI_MAX_LOOPS" ]; do
  loop_count=$((loop_count + 1))

  run_id="$(get_tag SaviRunId || true)"
  if [ -z "$run_id" ]; then
    log "sin tag SaviRunId → nada que procesar"
    break
  fi
  if [ "$run_id" = "$last_run_id" ]; then
    log "run_id=${run_id} sin cambios respecto a la vuelta anterior → fin del polling"
    break
  fi

  input_key="$(get_tag SaviInputKey || true)"
  if [ -z "$input_key" ]; then
    log "tag SaviRunId=${run_id} presente pero falta SaviInputKey → se omite"
    break
  fi

  if [ ! -f "${SAVI_ROOT}/.bootstrapped" ]; then
    log "bootstrap no realizado todavía → ejecutando ${SAVI_ROOT}/bootstrap.sh"
    if ! "${SAVI_ROOT}/bootstrap.sh"; then
      log "ERROR: bootstrap.sh falló, no se puede procesar run_id=${run_id}"
      break
    fi
  fi

  process_run "$run_id" "$input_key"
  last_run_id="$run_id"
done

log "ciclo de procesamiento terminado (vueltas=${loop_count})"
# Al llegar aquí el script termina (exit 0 implícito) y el trap EXIT (cleanup)
# sube el log final y apaga la instancia salvo SaviNoShutdown=true.
