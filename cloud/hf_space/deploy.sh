#!/usr/bin/env bash
# Publica la demo en Hugging Face Spaces (gratis). Requisito único: `hf auth login` una vez.
#   ./deploy.sh                 -> crea/actualiza <tu-usuario>/savi-agent
#   SPACE=usuario/nombre ./deploy.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_HF="$(hf auth whoami 2>/dev/null | grep -v -i 'hint\|update' | head -1 | awk '{print $NF}')"
[ -n "$USER_HF" ] && [ "$USER_HF" != "in" ] || { echo "No hay sesión: corre 'hf auth login' (token con permiso write)"; exit 1; }
SPACE="${SPACE:-$USER_HF/savi-agent}"
"$HERE/build.sh"
hf repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok >/dev/null
hf upload "$SPACE" "$HERE/dist" . --repo-type space --commit-message "Deploy SAVI agent"
SUB="$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
URL="https://$SUB.hf.space"
echo "Space: https://huggingface.co/spaces/$SPACE"
echo "Demo:  $URL   (el primer build tarda 1-3 min)"
echo "$URL" > "$HERE/.space_url"

# Keep-alive: los Spaces gratis se duermen tras 48 h sin visitas. La VM gratuita de GCP
# (nexus-cloud-node, vía Tailscale) visita /api/health cada 12 h. Idempotente.
VM="${SAVI_KEEPALIVE_HOST:-JulianRincon@100.70.251.71}"
if ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$HOME/.ssh/google_compute_engine" "$VM" true 2>/dev/null; then
  ssh -o BatchMode=yes -i "$HOME/.ssh/google_compute_engine" "$VM" \
    "(crontab -l 2>/dev/null | grep -v savi-keepalive; echo '17 */12 * * * curl -fsS -m 60 $URL/api/health >/dev/null 2>&1 # savi-keepalive') | crontab -"
  echo "Keep-alive instalado en la VM (cada 12 h)."
else
  echo "AVISO: no pude entrar a la VM ($VM); instala el keep-alive a mano:"
  echo "  17 */12 * * * curl -fsS -m 60 $URL/api/health >/dev/null 2>&1 # savi-keepalive"
fi

# Enlaces: homepage del repo en GitHub y botón "Live demo" de SAVI en el portafolio.
OLD_URL="https://5w442qdw5roag3chtm6esbohfu0mwaap.lambda-url.us-east-1.on.aws/"
gh repo edit Julian-Rincon/ames-housing-ml --homepage "$URL" >/dev/null 2>&1 && echo "Homepage del repo actualizada."
PORT="${PORTFOLIO_DIR:-$HOME/Documentos/Proyectos/Julian-Rincon.github.io}"
if [ -f "$PORT/index.html" ]; then
  python3 - "$PORT/index.html" "$OLD_URL" "$URL/" <<'PY'
import sys, pathlib
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
s = p.read_text()
marker = '<!-- savi-demo -->'
if old in s:
    s = s.replace(old, new)
elif marker in s:
    s = s.replace(marker, f'<a href="{new}" target="_blank" rel="noopener" class="project-link">Live demo →</a>')
p.write_text(s)
PY
  git -C "$PORT" commit -qam "Point SAVI live demo to the Hugging Face Space" && git -C "$PORT" push -q && echo "Portafolio actualizado y publicado."
fi

# README del repo: apuntar la demo al Space.
REPO="$(cd "$HERE/../.." && pwd)"
if grep -q "$OLD_URL" "$REPO/README.md"; then
  sed -i "s#$OLD_URL#$URL/#g" "$REPO/README.md"
  git -C "$REPO" commit -qam "Point live demo to the Hugging Face Space" && git -C "$REPO" push -q && echo "README de SAVI actualizado."
fi
