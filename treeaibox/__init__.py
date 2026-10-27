"""TreeAIBox Standalone and CloudCompare Plugin Application Package."""

# Ensure QtWebEngineWidgets is imported before any QApplication instance is created
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    pass

from treeaibox.app import TreeAIBoxWeb, main

__all__ = ["TreeAIBoxWeb", "main"]
