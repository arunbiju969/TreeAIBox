"""Forestry AI Suite package (modularized from treeAIbox_Enhancer.py)."""

import sys
from pathlib import Path

# Ensure the parent directory (scripts/visualization) is in sys.path so internal
# imports (e.g. `from forestry_suite.window...`) resolve seamlessly.
_pkg_parent = str(Path(__file__).resolve().parent.parent)
# Ensure the repository root (TreeAIBox) is in sys.path so root packages
# (e.g. `scripts`, `modules`, `treeaibox`) resolve seamlessly.
_repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)

for _p in [_repo_root, _pkg_parent]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from forestry_suite.window import TreeVisualizerGUI

__all__ = ["TreeVisualizerGUI"]
