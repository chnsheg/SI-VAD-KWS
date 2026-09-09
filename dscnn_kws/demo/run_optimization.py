"""Standalone launcher for the two-recording optimization command."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dscnn_kws.demo.optimization_cli import main


raise SystemExit(main())
