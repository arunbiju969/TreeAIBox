"""
TreeAIBox Main Application Window and Entry Point.
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QUrl
from PyQt6.QtWebChannel import QWebChannel

from treeaibox.bridge import WebEnginePage
from treeaibox.web_interface import WebInterface


class TreeAIBoxWeb(QMainWindow):
    """Main application window for TreeAIBox (PyQt6 + WebEngine + JS WebChannel)."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("TreeAIBox")
        self.resize(1200, 900)

        # Create web view
        self.web_view = WebEnginePage(self)
        self.setCentralWidget(self.web_view)

        # Set up web channel for JavaScript communication
        self.channel = QWebChannel()
        self.web_interface = WebInterface(self)
        self.channel.registerObject("backend", self.web_interface)
        self.web_view.page().setWebChannel(self.channel)

        # Load HTML content
        self.loadHtmlContent()

    def closeEvent(self, event):
        """Handle window close event with proper cleanup of background workers."""
        if hasattr(self.web_interface, 'workers'):
            for worker in self.web_interface.workers:
                if worker.isRunning():
                    worker.terminate()
                    worker.wait(3000)
            self.web_interface.workers.clear()

        app = QApplication.instance()
        if app is not None:
            app.quit()
            QApplication.exit()

    def loadHtmlContent(self):
        """Load the HTML UI file."""
        repo_root = Path(__file__).resolve().parent.parent
        html_path = repo_root / "treeaibox_ui.html"

        if html_path.exists():
            self.web_view.load(QUrl.fromLocalFile(str(html_path)))
        else:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TreeAIBox</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .error { color: red; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>TreeAIBox</h1>
                <p class="error">Error: UI file not found.</p>
                <p>Please ensure that treeaibox_ui.html is in the repository root directory.</p>
            </body>
            </html>
            """
            self.web_view.setHtml(html_content)


def main():
    """Application main entry point."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = TreeAIBoxWeb()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
