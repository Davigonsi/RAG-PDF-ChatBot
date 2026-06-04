"""Pytest configuration: add project root to sys.path for absolute imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
