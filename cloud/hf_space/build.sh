#!/usr/bin/env bash
# Arma cloud/hf_space/dist: código de la API + artefactos del modelo, listo para subir al Space.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CLOUD="$(dirname "$HERE")"
DIST="$HERE/dist"
RAG="${SAVI_RAG_DIR:-$CLOUD/out/local_run/rag}"
SERVING="${SAVI_SERVING_DIR:-$CLOUD/out/model/serving}"
for f in "$RAG/parcels.json.gz" "$RAG/avm_model.json" "$SERVING/policy.json"; do
  [ -f "$f" ] || { echo "Falta $f: corre el pipeline CPU/GPU o ajusta SAVI_RAG_DIR/SAVI_SERVING_DIR" >&2; exit 1; }
done
rm -rf "$DIST" && mkdir -p "$DIST/data"
cp "$CLOUD/api/handler.py" "$CLOUD/api/local_server.py" "$DIST/"
cp -r "$CLOUD/api/savi_api" "$CLOUD/api/static" "$DIST/"
find "$DIST" -name __pycache__ -prune -exec rm -rf {} +
cp -r "$RAG" "$DIST/data/rag" && cp -r "$SERVING" "$DIST/data/serving"
cp "$HERE/Dockerfile" "$DIST/Dockerfile"
cp "$HERE/README.space.md" "$DIST/README.md"
echo "dist listo: $(du -sh "$DIST" | cut -f1)"
