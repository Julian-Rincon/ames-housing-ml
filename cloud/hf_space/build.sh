#!/usr/bin/env bash
# Arma cloud/hf_space/dist como Space ESTÁTICO (gratis, sin servidor, no se duerme):
# la API de Python corre en el navegador con Pyodide. dist/ = index.html + app.zip + README.md
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CLOUD="$(dirname "$HERE")"
DIST="$HERE/dist"
RAG="${SAVI_RAG_DIR:-$CLOUD/out/local_run/rag}"
SERVING="${SAVI_SERVING_DIR:-$CLOUD/out/model/serving}"
for f in "$RAG/parcels.json.gz" "$RAG/avm_model.json" "$SERVING/policy.json"; do
  [ -f "$f" ] || { echo "Falta $f: corre el pipeline CPU/GPU o ajusta SAVI_RAG_DIR/SAVI_SERVING_DIR" >&2; exit 1; }
done
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/data"
cp "$CLOUD/api/handler.py" "$STAGE/"
cp -r "$CLOUD/api/savi_api" "$CLOUD/api/static" "$STAGE/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
cp -r "$RAG" "$STAGE/data/rag" && cp -r "$SERVING" "$STAGE/data/serving"
rm -rf "$DIST" && mkdir -p "$DIST"
# el zip se escribe FUERA de la carpeta comprimida (si no, se mete a sí mismo a medio escribir)
python3 -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])" "$DIST/app" "$STAGE"
python3 - "$CLOUD/api/static/index.html" "$HERE/pyodide_shim.html" "$DIST/index.html" <<'PY'
import sys, pathlib
page, shim, out = (pathlib.Path(a) for a in sys.argv[1:])
s = page.read_text()
assert s.count("</head>") == 1, "index.html sin </head> único"
pathlib.Path(out).write_text(s.replace("</head>", shim.read_text() + "</head>"))
PY
cp "$HERE/README.space.md" "$DIST/README.md"
echo "dist listo: $(du -sh "$DIST" | cut -f1)"
