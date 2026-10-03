# -*- coding: utf-8 -*-
"""Agrega `api/` a sys.path para poder hacer `import handler` y `import savi_api.*`."""
from __future__ import annotations

import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
