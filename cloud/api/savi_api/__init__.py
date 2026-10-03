# -*- coding: utf-8 -*-
"""
SAVI Agent API — núcleo RAG puro-Python (sin numpy/pandas/torch/xgboost).

Submódulos:
  store.py       carga de artefactos (local SAVI_DATA_DIR o S3) + índices en memoria
  inference.py   reinferencia exacta del AVM (XGBoost), K-Means, estado DQN y MLP (DQN)
  retrieval.py   consultas de parcelas/ventas/zonas/mercado y BM25 sobre knowledge/
  tools.py       TOOLS (formato Anthropic) + run_tool(name, input)
"""
