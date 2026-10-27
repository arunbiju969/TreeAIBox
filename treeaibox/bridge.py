"""
CloudCompare (PyCC) API bridge, threading worker, and WebEngine page helpers.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

# Detect CloudCompare plugin environment
try:
    import pycc
    import cccorelib

    CC = pycc.GetInstance()
    PYCC_AVAILABLE = True
    print("[INFO] CloudCompare API loaded successfully (Plugin Mode)")
except ImportError:
    PYCC_AVAILABLE = False
    CC = None
    print("[INFO] TreeAIBox running in Standalone Mode (CloudCompare not required)")
except Exception as e:
    PYCC_AVAILABLE = False
    CC = None
    print(f"[INFO] TreeAIBox running in Standalone Mode ({e})")


class ProcessEvents:
    """Helper to process Qt events periodically during intensive operations."""

    def __init__(self, msec=100):
        self.msec = msec
        self.last_time = 0

    def __call__(self, progress):
        import time
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self.last_time > self.msec:
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
            self.last_time = current_time
        return progress


class Worker(QThread):
    """Worker thread to run time-consuming tasks asynchronously in the background."""

    progressUpdated = pyqtSignal(int)
    finished = pyqtSignal(bool, object)
    errorOccurred = pyqtSignal(str)

    def __init__(self, task_function, *args, **kwargs):
        super().__init__()
        self.task_function = task_function
        self.args = args
        self.kwargs = kwargs
        self.result = None

    def run(self):
        try:
            if 'progress_callback' in self.kwargs:
                original_callback = self.kwargs['progress_callback']

                def thread_safe_callback(progress):
                    self.progressUpdated.emit(progress)
                    return original_callback(progress)

                self.kwargs['progress_callback'] = thread_safe_callback

            self.result = self.task_function(*self.args, **self.kwargs)
            self.finished.emit(True, self.result)
        except Exception as e:
            self.errorOccurred.emit(str(e))
            self.finished.emit(False, None)


class WebEnginePage(QWebEngineView):
    """Configured QWebEngineView for rendering the TreeAIBox web UI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

    def createWindow(self, _):
        return self
