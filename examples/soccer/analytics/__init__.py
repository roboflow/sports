# analytics package — player-motion analytics for the sports soccer example

import sys
from pathlib import Path

# The `sports` library is not pip-installed in this checkout; it lives at the
# repository root (…/sports/sports/). Ensure that root is importable no matter
# what the current working directory is when running analytics/main.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if (_REPO_ROOT / "sports").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
