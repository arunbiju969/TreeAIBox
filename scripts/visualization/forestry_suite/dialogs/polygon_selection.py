"""Interactive 2D polygon selection dialog (matplotlib canvas)."""

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PolygonSelectionDialog(QDialog):
    """Dialog for 2D polygon selection on flattened point cloud."""
    
    polygon_completed = pyqtSignal(np.ndarray)  # Signal emitted when polygon is completed
    
    def __init__(self, points_3d, itc_values, parent=None):
        super().__init__(parent)
        self.points_3d = points_3d  # Full 3D points
        self.itc_values = itc_values  # Corresponding ITC values
        self.view_mode = 'top'  # 'top', 'bottom', 'front', 'back', 'left', 'right'
        self.polygon_points = []
        self.polygon_lines = []
        self._polygon_finished = False
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("2D Polygon Selection")
        # Make window full screen for better visualization
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)
        self.setMinimumSize(screen.width(), screen.height())
        
        layout = QVBoxLayout(self)
        
        # View selection
        view_group = QHBoxLayout()
        view_label = QLabel("View:")
        view_label.setStyleSheet("font-weight: bold;")
        view_group.addWidget(view_label)
        
        self.view_button_group = QButtonGroup(self)
        
        self.top_view_radio = QRadioButton("Top (XY)")
        self.top_view_radio.setChecked(True)
        self.top_view_radio.toggled.connect(self.on_view_changed)
        self.view_button_group.addButton(self.top_view_radio)
        view_group.addWidget(self.top_view_radio)
        
        self.bottom_view_radio = QRadioButton("Bottom (XY)")
        self.bottom_view_radio.toggled.connect(self.on_view_changed)
        self.view_button_group.addButton(self.bottom_view_radio)
        view_group.addWidget(self.bottom_view_radio)
        
        self.front_view_radio = QRadioButton("Front (XZ)")
        self.front_view_radio.toggled.connect(self.on_view_changed)
        self.view_button_group.addButton(self.front_view_radio)
        view_group.addWidget(self.front_view_radio)
        
        self.back_view_radio = QRadioButton("Back (XZ)")
        self.back_view_radio.toggled.connect(self.on_view_changed)
        self.view_button_group.addButton(self.back_view_radio)
        view_group.addWidget(self.back_view_radio)
        
        self.left_view_radio = QRadioButton("Left (YZ)")
        self.left_view_radio.toggled.connect(self.on_view_changed)
        self.view_button_group.addButton(self.left_view_radio)
        view_group.addWidget(self.left_view_radio)
        
        self.right_view_radio = QRadioButton("Right (YZ)")
        self.right_view_radio.toggled.connect(self.on_view_changed)
        self.view_button_group.addButton(self.right_view_radio)
        view_group.addWidget(self.right_view_radio)
        
        view_group.addStretch()
        layout.addLayout(view_group)
        
        # Instructions
        instructions = QLabel("Click to add polygon points. Right-click to finish.\nPoints will be manually placed, not snapped to existing points.")
        instructions.setStyleSheet("font-weight: bold; color: #4CAF50; margin: 10px;")
        layout.addWidget(instructions)
        
        # Matplotlib figure - make it more portrait for tree visualization
        self.figure = Figure(figsize=(6, 10))  # width=6, height=10 inches
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Connect matplotlib events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.finish_button = QPushButton("Finish Polygon")
        self.finish_button.clicked.connect(self.finish_polygon)
        self.finish_button.setEnabled(False)
        button_layout.addWidget(self.finish_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_polygon)
        button_layout.addWidget(self.clear_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Initial plot
        self.plot_points()
        
    def on_view_changed(self):
        """Handle view mode change."""
        if self.top_view_radio.isChecked():
            self.view_mode = 'top'
        elif self.bottom_view_radio.isChecked():
            self.view_mode = 'bottom'
        elif self.front_view_radio.isChecked():
            self.view_mode = 'front'
        elif self.back_view_radio.isChecked():
            self.view_mode = 'back'
        elif self.left_view_radio.isChecked():
            self.view_mode = 'left'
        elif self.right_view_radio.isChecked():
            self.view_mode = 'right'
        
        # Clear polygon when switching views
        self.clear_polygon()
        self.plot_points()
        
    def plot_points(self):
        """Plot the 2D points."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get 2D coordinates based on view mode
        if self.view_mode == 'top':
            points_2d = self.points_3d[:, :2]  # X, Y coordinates
            xlabel = 'X Coordinate'
            ylabel = 'Y Coordinate'
            title = 'Top View (XY) - Click to draw polygon'
        elif self.view_mode == 'bottom':
            # Bottom view: flip Y coordinates to simulate viewing from below
            points_2d = self.points_3d[:, :2].copy()  # X, Y coordinates
            points_2d[:, 1] = -points_2d[:, 1]  # Flip Y axis
            xlabel = 'X Coordinate'
            ylabel = 'Y Coordinate (Flipped)'
            title = 'Bottom View (XY) - Click to draw polygon'
        elif self.view_mode == 'front':
            points_2d = self.points_3d[:, [0, 2]]  # X, Z coordinates
            xlabel = 'X Coordinate'
            ylabel = 'Z Coordinate (Height)'
            title = 'Front View (XZ) - Click to draw polygon'
        elif self.view_mode == 'back':
            # Back view: flip X coordinates to simulate viewing from behind
            points_2d = self.points_3d[:, [0, 2]].copy()  # X, Z coordinates
            points_2d[:, 0] = -points_2d[:, 0]  # Flip X axis
            xlabel = 'X Coordinate (Flipped)'
            ylabel = 'Z Coordinate (Height)'
            title = 'Back View (XZ) - Click to draw polygon'
        elif self.view_mode == 'left':
            points_2d = self.points_3d[:, [1, 2]]  # Y, Z coordinates
            xlabel = 'Y Coordinate'
            ylabel = 'Z Coordinate (Height)'
            title = 'Left View (YZ) - Click to draw polygon'
        elif self.view_mode == 'right':
            # Right view: flip Y coordinates to simulate viewing from right
            points_2d = self.points_3d[:, [1, 2]].copy()  # Y, Z coordinates
            points_2d[:, 0] = -points_2d[:, 0]  # Flip Y axis
            xlabel = 'Y Coordinate (Flipped)'
            ylabel = 'Z Coordinate (Height)'
            title = 'Right View (YZ) - Click to draw polygon'
        else:
            # Default to top view
            points_2d = self.points_3d[:, :2]
            xlabel = 'X Coordinate'
            ylabel = 'Y Coordinate'
            title = 'Top View (XY) - Click to draw polygon'
        
        # Plot points colored by ITC values using the same 9-color scheme as 3D view
        # Create color indices for 9 cycling colors
        color_indices = self.itc_values % 9
        scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                           c=color_indices, s=3, alpha=0.8, cmap='tab10', vmin=0, vmax=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio to 'equal' for accurate spatial representation
        # This ensures points maintain their true spatial relationships without distortion
        ax.set_aspect('equal', adjustable='datalim')
        
        # Add colorbar with ITC value labels
        cbar = self.figure.colorbar(scatter, ax=ax, shrink=0.8, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        cbar.set_label('ITC Color Group')
        cbar.set_ticklabels(['0-8', '1-9', '2-10', '3-11', '4-12', '5-13', '6-14', '7-15', '8-16'])
        
        # Plot current polygon
        if len(self.polygon_points) > 0:
            poly_x = [p[0] for p in self.polygon_points]
            poly_y = [p[1] for p in self.polygon_points]
            
            # Plot points
            ax.scatter(poly_x, poly_y, c='red', s=50, marker='x', linewidth=2)
            
            # Plot lines
            if len(self.polygon_points) > 1:
                for i in range(len(self.polygon_points) - 1):
                    ax.plot([poly_x[i], poly_x[i+1]], [poly_y[i], poly_y[i+1]], 
                           'r-', linewidth=2)
        
        self.canvas.draw()
        
    def on_canvas_click(self, event):
        """Handle mouse clicks on the canvas."""
        if event.button == 1 and event.xdata is not None and event.ydata is not None:  # Left click
            # Add point manually (not snapped to existing points)
            point = [event.xdata, event.ydata]
            self.polygon_points.append(point)
            self.plot_points()
            
            if len(self.polygon_points) >= 3:
                self.finish_button.setEnabled(True)
                
        elif event.button == 3:  # Right click
            if len(self.polygon_points) >= 3:
                self.finish_polygon()
    
    def finish_polygon(self):
        """Finish the polygon and emit signal."""
        if len(self.polygon_points) >= 3:
            self._polygon_finished = True
            polygon_array = np.array(self.polygon_points)
            self.polygon_completed.emit(polygon_array)
            self.accept()
    
    def clear_polygon(self):
        """Clear the current polygon."""
        self.polygon_points = []
        self.finish_button.setEnabled(False)
        self.plot_points()


