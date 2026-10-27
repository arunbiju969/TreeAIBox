"""Application-wide logging, lazy log files, and Qt exception capture."""

import sys
import os
import logging
import traceback
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMessageBox


class LazyFileHandler(logging.Handler):
    """File handler that creates its log file only when the first record is emitted.

    Clean runs leave no empty files behind in scripts/error_logs.
    """
    def __init__(self, log_dir):
        super().__init__()
        self._log_dir = log_dir
        self._stream = None
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            if self._stream is None:
                os.makedirs(self._log_dir, exist_ok=True)
                log_file = os.path.join(self._log_dir, f'errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
                self._stream = open(log_file, 'a', encoding='utf-8')
            self._stream.write(self.format(record) + '\n')
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None
            super().close()


def setup_exception_handler():
    """Set up global exception handler to capture unhandled errors including PyQt exceptions.

    The log file under scripts/error_logs is created lazily: only runs that
    actually log something produce a file. Console output stays at INFO.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'error_logs')

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    file_handler = LazyFileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        """Global exception handler for uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.error(f"Uncaught exception:\n{error_msg}")
        print(f"\n{'='*60}\n[Error] UNCAUGHT ERROR:\n{error_msg}\n{'='*60}\n", file=sys.stderr)

    sys.excepthook = handle_exception


class ErrorCapturingQApplication(QApplication):
    """Custom QApplication that captures and logs exceptions from the event loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception_logged = False

    def notify(self, receiver, event):
        """Override notify to catch exceptions in PyQt slots and event handlers."""
        try:
            return super().notify(receiver, event)
        except Exception as e:
            error_msg = f"""
{'='*60}
[Error] PyQt EVENT LOOP ERROR:
Exception in {receiver.__class__.__name__}.{event.__class__.__name__}
{'='*60}
{traceback.format_exc()}
{'='*60}
"""
            print(error_msg, file=sys.stderr)
            logging.error(f"PyQt event loop exception: {error_msg}")

            # Also show a GUI message box if possible
            try:
                QMessageBox.critical(None, "Error", f"An error occurred:\n\n{str(e)}\n\nCheck error logs for details.")
            except Exception:
                pass

            return False
