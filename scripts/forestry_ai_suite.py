#!/usr/bin/env python3
"""
Forestry AI Suite - Main Application Launcher.

Multi-model segmentation visualization & analytics for TreeAIBox, TreeLearn, and ForestFormer.
"""
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
_vis_dir = str(Path(__file__).resolve().parent / "visualization")

for _p in [_repo_root, _vis_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from forestry_suite.app import main

if __name__ == "__main__":
    main()
