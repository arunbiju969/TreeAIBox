#!/usr/bin/env python3
"""
Forestry AI Suite - entry point.

Multi-model segmentation visualization & analytics for TreeAIBox, TreeLearn,
and ForestFormer. The implementation lives in the `forestry_suite` package
(modularized from this former single-file application).
"""
import os
import sys
from pathlib import Path

# Ensure safe UTF-8 terminal I/O on Windows without charmap encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure both repository root and scripts/visualization are in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
_vis_dir = str(Path(__file__).resolve().parent)

for _p in [_repo_root, _vis_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from forestry_suite.app import main

if __name__ == "__main__":
    main()
