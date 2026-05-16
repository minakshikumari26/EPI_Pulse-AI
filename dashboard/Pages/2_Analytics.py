"""EpiPulse AI analytics page wrapper."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = Path(__file__).resolve().parents[1]

for p in [str(PROJECT_ROOT), str(DASHBOARD)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import app as analytics_app

analytics_app.main()
