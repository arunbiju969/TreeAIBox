"""Application entry point for the Forestry AI Suite."""

import sys
import time
from forestry_suite.error_handling import setup_exception_handler, ErrorCapturingQApplication
from forestry_suite.window import TreeVisualizerGUI

def main():
    """Main application entry point."""
    import time
    
    # Set up error handling before creating GUI
    setup_exception_handler()
    
    startup_start = time.perf_counter()
    print(f"[STARTUP] {time.strftime('%H:%M:%S')} - Starting application...")
    
    # Use custom QApplication that captures PyQt exceptions
    app = ErrorCapturingQApplication(sys.argv)
    print(f"[STARTUP] {time.strftime('%H:%M:%S')} - QApplication created ({time.perf_counter() - startup_start:.2f}s)")

    # Set application properties
    app.setApplicationName("Forestry AI Suite")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ForestryAI")
    print(f"[STARTUP] {time.strftime('%H:%M:%S')} - App properties set ({time.perf_counter() - startup_start:.2f}s)")

    # Create and show main window maximized
    window_start = time.perf_counter()
    print(f"[STARTUP] {time.strftime('%H:%M:%S')} - Creating TreeVisualizerGUI...")
    window = TreeVisualizerGUI()
    print(f"[STARTUP] {time.strftime('%H:%M:%S')} - TreeVisualizerGUI created ({time.perf_counter() - window_start:.2f}s)")
    
    window.showMaximized()
    print(f"[STARTUP] {time.strftime('%H:%M:%S')} - Window shown ({time.perf_counter() - startup_start:.2f}s total)")

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
