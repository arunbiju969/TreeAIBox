#!/usr/bin/env python3
"""
TreeAIBox - Main Application Entry Point.

Modularized implementation lives in the `treeaibox` package.
This module maintains backward-compatibility for all existing launchers and scripts.
"""

import sys
import os

# QtWebEngineWidgets must be imported before any QApplication is instantiated
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Re-export core classes for backward compatibility
from treeaibox.bridge import (
    CC,
    PYCC_AVAILABLE,
    ProcessEvents,
    Worker,
    WebEnginePage,
)
from treeaibox.models import (
    get_model_storage_dir,
    get_log_storage_dir,
    check_model_exists,
    download_model_file,
    load_model_zoo,
)
from treeaibox.point_cloud import (
    StandalonePointCloud,
    ScalarFieldWrapper,
    load_point_cloud_file,
    save_point_cloud_to_las,
)
from treeaibox.web_interface import WebInterface
from treeaibox.app import TreeAIBoxWeb, main

__all__ = [
    "CC",
    "PYCC_AVAILABLE",
    "ProcessEvents",
    "Worker",
    "WebEnginePage",
    "StandalonePointCloud",
    "ScalarFieldWrapper",
    "load_point_cloud_file",
    "save_point_cloud_to_las",
    "get_model_storage_dir",
    "get_log_storage_dir",
    "check_model_exists",
    "download_model_file",
    "load_model_zoo",
    "WebInterface",
    "TreeAIBoxWeb",
    "main",
]

if __name__ == "__main__":
    main()
