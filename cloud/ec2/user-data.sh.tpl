#!/bin/bash
# SAVI v2 · cloud-init user-data — EC2 "savi-cpu-engine" (Amazon Linux 2023 x86_64)
#
# Se ejecuta UNA SOLA VEZ, en el primer boot de la instancia (comportamiento
# estándar de cloud-init: usa el instance-id para no repetirse en los
# stop/start del Learner Lab). Dentro de este arranque:
#   1. Escribe /etc/savi.env con la config fija (buckets, región, log group).
#   2. Escribe /opt/savi/bootstrap.sh (idempotente, lo vuelve a correr
#      run_pipeline.sh si /opt/savi/.bootstrapped no existe).
#   3. Instala y habilita el unit systemd savi-pipeline.service, que SÍ corre
#      en CADA boot (incluidos los reinicios del Learner Lab).
#   4. Descarga run_pipeline.sh desde S3 (fuente de verdad versionada ahí).
#   5. Arranca el servicio sin bloquear el resto del cloud-init.
#
# El deploy script sustituye estos placeholders con str.replace() antes de
# pasar el archivo como user-data:
RAW_BUCKET="__RAW_BUCKET__"
PROCESSED_BUCKET="__PROCESSED_BUCKET__"
REGION="__REGION__"
LOG_GROUP="__LOG_GROUP__"

# set -e: cualquier fallo debe cortar el user-data y disparar el cost guard
# (shutdown) en vez de dejar la instancia a medio configurar corriendo 24/7.
set -euo pipefail

SAVI_ROOT="/opt/savi"
LOG_DIR="/var/log/savi"
LOG_FILE="${LOG_DIR}/user-data.log"
mkdir -p "$SAVI_ROOT" "$LOG_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"; }

imds_token() {
  curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" --max-time 5
}

# Lee un tag de la instancia vía IMDSv2. 404 (o cualquier respuesta != 200) =
# tag ausente → devuelve vacío y rc != 0 (nunca corta el script: se usa con `|| true`).
get_tag() {
  local name="$1" token resp http_code body
  token="$(imds_token)"
  resp="$(curl -sS -o - -w $'\n%{http_code}' \
    -H "X-aws-ec2-metadata-token: ${token}" \
    "http://169.254.169.254/latest/meta-data/tags/instance/${name}" --max-time 5)"
  http_code="${resp##*$'\n'}"
  body="${resp%$'\n'*}"
  if [ "$http_code" = "200" ]; then
    printf '%s' "$body"
    return 0
  fi
  return 1
}

# ── Cost guard: CUALQUIER fallo del user-data termina apagando la instancia,
#    salvo que la instancia tenga el tag SaviNoShutdown=true (sólo depuración).
cleanup() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    log "user-data falló con rc=${rc}"
    local no_shutdown=""
    no_shutdown="$(get_tag SaviNoShutdown || true)"
    if [ "$no_shutdown" = "true" ]; then
      log "SaviNoShutdown=true → NO se apaga la instancia pese al fallo"
    else
      log "Apagando instancia (cost guard de user-data)"
      shutdown -h now || true
    fi
  fi
}
trap cleanup EXIT

log "=== SAVI user-data: inicio (primer boot) ==="

# 1) Config fija para todos los scripts posteriores (bootstrap.sh, run_pipeline.sh)
cat > /etc/savi.env <<EOF
RAW_BUCKET=${RAW_BUCKET}
PROCESSED_BUCKET=${PROCESSED_BUCKET}
REGION=${REGION}
LOG_GROUP=${LOG_GROUP}
EOF
log "escrito /etc/savi.env"

# 2) bootstrap.sh idempotente: instala paquetes, crea el venv, instala deps,
#    configura y arranca el agente de CloudWatch, y marca .bootstrapped.
#    Heredoc con comillas ('BOOTSTRAP_EOF') para que NO se expandan variables
#    aquí: bootstrap.sh las resuelve él mismo al correr, leyendo /etc/savi.env.
cat > "${SAVI_ROOT}/bootstrap.sh" <<'BOOTSTRAP_EOF'
#!/bin/bash
# SAVI v2 · bootstrap idempotente del EC2 (lo invoca run_pipeline.sh si hace falta).
set -uo pipefail

SAVI_ROOT="${SAVI_ROOT:-/opt/savi}"
SAVI_ENV_FILE="${SAVI_ENV_FILE:-/etc/savi.env}"
LOG_DIR="${SAVI_LOG_DIR:-/var/log/savi}"
mkdir -p "$LOG_DIR"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "[bootstrap] $*" | tee -a "${LOG_DIR}/bootstrap.log"; }

# shellcheck disable=SC1090
if [ -f "$SAVI_ENV_FILE" ]; then
  source "$SAVI_ENV_FILE"
fi
: "${RAW_BUCKET:?falta RAW_BUCKET en ${SAVI_ENV_FILE}}"
: "${REGION:?falta REGION en ${SAVI_ENV_FILE}}"
: "${LOG_GROUP:?falta LOG_GROUP en ${SAVI_ENV_FILE}}"

log "dnf install python3.12 (fallback python3.11) + agente de CloudWatch"
dnf install -y amazon-cloudwatch-agent || log "ADVERTENCIA: no se pudo instalar amazon-cloudwatch-agent"
if dnf install -y python3.12 python3.12-pip; then
  PY=python3.12
elif dnf install -y python3.11 python3.11-pip; then
  PY=python3.11
else
  log "ERROR: no hay python3.12 ni python3.11 disponibles en dnf"
  exit 1
fi

if [ ! -x "${SAVI_ROOT}/venv/bin/python" ]; then
  log "creando venv en ${SAVI_ROOT}/venv con ${PY}"
  "$PY" -m venv "${SAVI_ROOT}/venv" || exit 1
fi

mkdir -p "${SAVI_ROOT}/code"
log "descargando requirements-ec2.txt desde s3://${RAW_BUCKET}/code/cpu/"
aws s3 cp "s3://${RAW_BUCKET}/code/cpu/requirements-ec2.txt" "${SAVI_ROOT}/requirements-ec2.txt" --region "$REGION"

log "instalando dependencias Python en el venv"
"${SAVI_ROOT}/venv/bin/pip" install --upgrade pip || exit 1
"${SAVI_ROOT}/venv/bin/pip" install -r "${SAVI_ROOT}/requirements-ec2.txt" || exit 1

# Config + arranque del agente de CloudWatch: generamos el JSON aquí porque
# necesita el instance-id real (nombre del log stream) y REGION/LOG_GROUP,
# que sólo se conocen en este boot. Ver ec2/cloudwatch-agent.json en el repo
# para la forma documentada de esta misma config.
TOKEN="$(curl -sS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' --max-time 5)"
INSTANCE_ID="$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/meta-data/instance-id --max-time 5)"

cat > "${SAVI_ROOT}/cloudwatch-agent.json" <<EOF
{
  "agent": {"metrics_collection_interval": 60, "region": "${REGION}", "logfile": "/opt/aws/amazon-cloudwatch-agent/logs/amazon-cloudwatch-agent.log"},
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/savi/*.log",
            "log_group_name": "${LOG_GROUP}",
            "log_stream_name": "${INSTANCE_ID}",
            "timestamp_format": "%Y-%m-%dT%H:%M:%SZ",
            "retention_in_days": 14
          }
        ]
      }
    }
  }
}
EOF

log "configurando y arrancando el agente de CloudWatch"
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s -c "file:${SAVI_ROOT}/cloudwatch-agent.json" \
  || log "ADVERTENCIA: agente CloudWatch no arrancó (los logs igual se suben a S3)"

touch "${SAVI_ROOT}/.bootstrapped"
log "bootstrap completado"
BOOTSTRAP_EOF
chmod +x "${SAVI_ROOT}/bootstrap.sh"
log "escrito ${SAVI_ROOT}/bootstrap.sh"

# 3) systemd unit: corre run_pipeline.sh en CADA boot (Type=oneshot, habilitado).
cat > /etc/systemd/system/savi-pipeline.service <<'UNIT_EOF'
[Unit]
Description=SAVI v2 - motor CPU pipeline (EC2)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/opt/savi/run_pipeline.sh
TimeoutStartSec=4200
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
UNIT_EOF
log "escrito /etc/systemd/system/savi-pipeline.service"

systemctl daemon-reload
systemctl enable savi-pipeline.service
log "servicio savi-pipeline.service habilitado (corre en cada boot)"

# 4) run_pipeline.sh es la fuente de verdad versionada en S3 (no se embebe
#    aquí): así se puede corregir sin regenerar el user-data / recrear la EC2.
log "descargando run_pipeline.sh desde s3://${RAW_BUCKET}/code/cpu/"
aws s3 cp "s3://${RAW_BUCKET}/code/cpu/run_pipeline.sh" "${SAVI_ROOT}/run_pipeline.sh" --region "$REGION"
chmod +x "${SAVI_ROOT}/run_pipeline.sh"

# 5) Pre-bootstrap en el primer boot: así la primera corrida real no paga los
#    3-5 min de dnf+pip. Si falla, el trap apaga y run_pipeline.sh lo reintenta.
log "ejecutando bootstrap.sh (dnf + venv + pip + CloudWatch agent)"
"${SAVI_ROOT}/bootstrap.sh"

# 6) Primer arranque del servicio (sin tags → se apaga solo, queda listo en 'stopped').
systemctl start --no-block savi-pipeline.service
log "savi-pipeline.service arrancado (--no-block)"

log "=== SAVI user-data: fin (OK) ==="
