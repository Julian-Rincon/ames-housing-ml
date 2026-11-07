#!/usr/bin/env bash
# Publica la demo en Hugging Face Spaces como Space ESTÁTICO (gratis, no se duerme; la API
# corre en el navegador con Pyodide). Requisito único: `hf auth login` una vez.
#   ./deploy.sh                 -> crea/actualiza <tu-usuario>/savi-agent
#   SPACE=usuario/nombre ./deploy.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_HF="$(hf auth whoami 2>/dev/null | grep -v -i 'hint\|update' | head -1 | awk '{print $NF}' | sed 's/^user=//I')"
[ -n "$USER_HF" ] && [ "$USER_HF" != "in" ] || { echo "No hay sesión: corre 'hf auth login' (token con permiso write)"; exit 1; }
SPACE="${SPACE:-$USER_HF/savi-agent}"
"$HERE/build.sh"
hf spaces info "$SPACE" >/dev/null 2>&1 || hf repos create "$SPACE" --repo-type space --space-sdk static >/dev/null
# upload_folder en vez de `hf upload`: el CLI intenta crear el repo y HF responde 402.
python3 -c 'import sys; from huggingface_hub import HfApi; HfApi().upload_folder(repo_id=sys.argv[1], repo_type="space", folder_path=sys.argv[2], commit_message="Deploy SAVI agent (static, Pyodide)", delete_patterns=["style.css"])' "$SPACE" "$HERE/dist"
SUB="$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
URL="https://$SUB.static.hf.space"
echo "Space: https://huggingface.co/spaces/$SPACE"
echo "Demo:  $URL   (la primera carga baja Pyodide, ~10 s)"
echo "$URL" > "$HERE/.space_url"


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
s = s.replace(old, new)
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
