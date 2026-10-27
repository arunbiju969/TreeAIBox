"""Stylesheet, ribbon UI, panels, console. (TreeVisualizerGUI mixin)."""

import os
import sys
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit, QLabel,
    QProgressBar, QSplitter, QFrame, QComboBox, QCheckBox,
    QGroupBox, QScrollArea, QTextEdit, QButtonGroup, QRadioButton,
    QSpinBox, QDoubleSpinBox, QSlider, QStackedWidget, QToolButton, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QIcon
from forestry_suite.widgets import ModernButton


class UiMixin:
    """Stylesheet, ribbon UI, panels, console."""

    def apply_stylesheet(self):
        """Apply modern dark theme stylesheet."""
        icons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "icons")
        os.makedirs(icons_dir, exist_ok=True)
        up_icon_path = os.path.join(icons_dir, "spin_up.svg").replace("\\", "/")
        down_icon_path = os.path.join(icons_dir, "spin_down.svg").replace("\\", "/")

        stylesheet = f"""
            QMainWindow, QDialog {{
                background-color: #2b2b2b;
            }}
            QWidget {{
                color: #e0e0e0;
                font-family: "Segoe UI", Arial, sans-serif;
            }}
            QGroupBox {{
                border: 1px solid #3d3d3d;
                border-radius: 0px;
                margin-top: 18px;
                padding-top: 14px;
                background-color: #242424;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 0px;
                padding: 0px 6px;
                color: #4CAF50;
                font-weight: bold;
                font-size: 11px;
            }}
            QComboBox {{
                background-color: #383838;
                border: 1px solid #555555;
                border-radius: 0px;
                padding: 2px 22px 2px 6px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 600;
                min-height: 22px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #555555;
                background-color: #444444;
            }}
            QComboBox::drop-down:hover {{
                background-color: #555555;
            }}
            QComboBox::down-arrow {{
                image: url("{down_icon_path}");
                width: 8px;
                height: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #383838;
                border: 1px solid #555555;
                color: #ffffff;
                selection-background-color: #4CAF50;
            }}
            QSpinBox, QDoubleSpinBox, QLineEdit {{
                background-color: #383838;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 0px;
                padding: 2px 4px;
                font-size: 11px;
                font-weight: 600;
                min-height: 22px;
            }}
            QSpinBox, QDoubleSpinBox {{
                padding-right: 18px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                height: 12px;
                background-color: #444444;
                border-left: 1px solid #555555;
                border-bottom: 1px solid #383838;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                height: 12px;
                background-color: #444444;
                border-left: 1px solid #555555;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: #555555;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: url("{up_icon_path}");
                width: 8px;
                height: 5px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url("{down_icon_path}");
                width: 8px;
                height: 5px;
            }}
            QCheckBox {{
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:unchecked {{
                border: 2px solid #555555;
                background-color: #333333;
            }}
            QCheckBox::indicator:checked {{
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
            }}
            QLabel {{
                color: #e0e0e0;
            }}
            QProgressBar {{
                border: 1px solid #555555;
                border-radius: 0px;
                text-align: center;
                background-color: #333333;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: #4CAF50;
                border-radius: 0px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: #404040;
                border-radius: 0px;
            }}
            QSlider::sub-page:horizontal {{
                background: #4CAF50;
                border-radius: 0px;
            }}
            QSlider::handle:horizontal {{
                background: #e0e0e0;
                border: 2px solid #4CAF50;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 0px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #ffffff;
                border-color: #66BB6A;
            }}
        """
        qapp = QApplication.instance()
        if qapp:
            qapp.setStyleSheet(stylesheet)
        self.setStyleSheet(stylesheet)


    def init_ui(self):
        """Initialize the user interface with top category navigation bar and collapsible tree/layer panel."""
        import time
        ui_start = time.perf_counter()
        print(f"[UI] {time.strftime('%H:%M:%S')} - init_ui starting...")
        
        self.setWindowTitle("Forestry AI Suite - Multi-Model Tree Visualizer & Analytics")
        self.setGeometry(100, 80, 1520, 960)
        
        # Set window icon if available
        icon_path = None
        for candidate in [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "treeaibox_logo.ico"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "treeaibox_logo.ico"),
        ]:
            if os.path.exists(candidate):
                icon_path = candidate
                break
        if icon_path and os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Central widget and top-level vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 10)
        main_layout.setSpacing(6)

        # 1. Initialize all interactive controls and inputs
        self._init_controls()

        # 2. Top Header Bar (Branding, Multi-Model Chips, Active File, Quick Actions)
        header_bar = self.create_header_bar()
        main_layout.addWidget(header_bar)

        # 3. Top Category Navigation Bar / Ribbon
        ribbon_bar = self.create_top_ribbon()
        main_layout.addWidget(ribbon_bar)

        # 4. Center Area: Splitter with Left Collapsible Panel and 3D Plotter Viewport
        self.center_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.center_splitter.setHandleWidth(6)
        self.center_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #383838;
                border-radius: 0px;
            }
            QSplitter::handle:hover {
                background-color: #4CAF50;
            }
        """)

        # Left panel: Compact & Collapsible Trees & Layers Panel
        self.left_panel = self.create_left_panel()
        self.sidebar = self.left_panel  # backwards-compatibility alias
        self.center_splitter.addWidget(self.left_panel)

        # Right panel: 3D Visualization Canvas
        plotter_frame = QFrame()
        plotter_frame.setObjectName("plotterFrame")
        plotter_frame.setStyleSheet("""
            #plotterFrame {
                background-color: #242424;
                border: 1px solid #444444;
                border-radius: 0px;
            }
        """)
        plotter_layout = QVBoxLayout(plotter_frame)
        plotter_layout.setContentsMargins(5, 5, 5, 5)
        plotter_layout.setSpacing(4)

        # Quick view control buttons directly above plotter
        view_buttons_layout = QHBoxLayout()
        view_buttons_layout.setSpacing(4)
        
        self.view_buttons = {}
        view_configs = [
            ('Front', 'view_front'),
            ('Back', 'view_back'), 
            ('Top', 'view_top'),
            ('Bottom', 'view_bottom'),
            ('Left', 'view_left'),
            ('Right', 'view_right')
        ]
        
        for label, view_type in view_configs:
            button = ModernButton(label)
            button.setFixedSize(54, 26)
            button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 4px; }")
            button.clicked.connect(lambda checked, vt=view_type: self.set_camera_view(vt))
            view_buttons_layout.addWidget(button)
            self.view_buttons[view_type] = button

        view_buttons_layout.addWidget(self.fit_to_screen_button)
        view_buttons_layout.addWidget(self.clear_button)
        view_buttons_layout.addStretch()

        plotter_layout.addLayout(view_buttons_layout)

        # 3D QtInteractor Plotter
        plotter_start = time.perf_counter()
        print(f"[UI] {time.strftime('%H:%M:%S')} - Creating QtInteractor (3D plotter)...")
        self.plotter = QtInteractor()
        print(f"[UI] {time.strftime('%H:%M:%S')} - QtInteractor created ({time.perf_counter() - plotter_start:.3f}s)")
        self.plotter.set_background('#222222')
        plotter_layout.addWidget(self.plotter.interactor)

        # Enable point picking
        self.plotter.enable_point_picking(callback=self.on_point_picked, show_message=False)
        self.plotter.add_text("Click on points to see ITC values", position='lower_left', 
                            font_size=10, color='#FFFFFF', name='itc_instruction_text')

        # Progress bar
        plotter_layout.addWidget(self.progress_bar)

        self.center_splitter.addWidget(plotter_frame)
        self.center_splitter.setStretchFactor(0, 0)
        self.center_splitter.setStretchFactor(1, 1)
        self.center_splitter.setSizes([320, 1200])

        main_layout.addWidget(self.center_splitter, 1)

        # 5. Bottom Console Output
        console_widget = self.create_console_widget()
        main_layout.addWidget(console_widget)

        # Initialize empty plot
        self.clear_plot()
        print(f"[UI] {time.strftime('%H:%M:%S')} - init_ui complete ({time.perf_counter() - ui_start:.3f}s total)")


    def _init_controls(self):
        """Instantiate and configure all interactive controls and inputs."""
        # File & Layers
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #90caf9; font-weight: 500; font-size: 11px;")

        self.load_button = ModernButton("Add LAS File(s)")
        self.load_button.setToolTip("Add single or multiple LAS/LAZ files to layer manager")
        self.load_button.clicked.connect(self.load_las_file)
        self.load_button.setFixedHeight(26)

        self.save_button = ModernButton("Save LAS")
        self.save_button.setToolTip("Save modified point cloud with updated ITC/classifications to LAS")
        self.save_button.clicked.connect(self.save_las_file)
        self.save_button.setEnabled(False)
        self.save_button.setFixedHeight(26)

        btn_small_style = """
            QPushButton {
                background-color: #383838;
                border: 1px solid #4d4d4d;
                color: #e0e0e0;
                font-size: 10px;
                font-weight: 500;
                padding: 2px 4px;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #484848;
                color: #ffffff;
                border-color: #5d5d5d;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
            }
            QPushButton:disabled {
                background-color: #262626;
                color: #666666;
                border-color: #333333;
            }
        """

        self.add_layer_btn = ModernButton("Add Layer")
        self.add_layer_btn.setToolTip("Add LAS/LAZ file(s) to layer manager")
        self.add_layer_btn.setFixedHeight(26)
        self.add_layer_btn.setStyleSheet(btn_small_style)
        self.add_layer_btn.clicked.connect(self.load_las_file)

        self.remove_las_btn = ModernButton("Remove")
        self.remove_las_btn.setToolTip("Remove selected LAS file from layer manager")
        self.remove_las_btn.setFixedHeight(26)
        self.remove_las_btn.setStyleSheet(btn_small_style)
        self.remove_las_btn.clicked.connect(self.remove_selected_las_layer)

        self.toggle_las_batch_btn = ModernButton("Check All")
        self.toggle_las_batch_btn.setToolTip("Toggle checking all LAS files for batch processing")
        self.toggle_las_batch_btn.setFixedHeight(26)
        self.toggle_las_batch_btn.setStyleSheet(btn_small_style)
        self.toggle_las_batch_btn.clicked.connect(self.toggle_las_batch_checks)

        self.compare_las_btn = ModernButton("Compare Side-by-Side")
        self.compare_las_btn.setToolTip("Open two LAS files side-by-side for simultaneous 3D comparison")
        self.compare_las_btn.setFixedHeight(26)
        self.compare_las_btn.setStyleSheet("""
            QPushButton {
                background-color: #0288D1;
                border: none;
                color: white;
                font-weight: bold;
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #039BE5;
            }
        """)
        self.compare_las_btn.clicked.connect(self.open_las_comparison)

        self.las_layer_list = QListWidget()
        self.las_layer_list.setMaximumHeight(140)
        self.las_layer_list.setToolTip("Loaded LAS layers. Check items to include in multi-LAS batch processing.\nClick an item to switch active viewer.")
        self.las_layer_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 0px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #1976D2;
                color: white;
            }
        """)
        self.las_layer_list.itemClicked.connect(self._on_las_layer_clicked)
        self.las_layer_list.itemChanged.connect(self._on_las_layer_check_changed)

        self.las_layer_group = QGroupBox("LAS Layer Manager")

        # GDB Layer Controls
        self.import_gdb_button = ModernButton("Import Points from GDB")
        self.import_gdb_button.clicked.connect(self.import_points_from_gdb)
        self.import_gdb_button.setFixedHeight(26)
        self.import_gdb_button.setToolTip("Import point features from a .gdb geodatabase and view them in 3D.")
        self.import_gdb_button.setStyleSheet("""
            QPushButton {
                background-color: #1976D2; border: none; color: white;
                padding: 4px 12px; font-size: 11px; border-radius: 0px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1565C0; }
        """)

        self.gdb_layer_group = QGroupBox("GDB Layer Manager")
        self.gdb_layer_group.setVisible(False)
        gdb_layer_vbox = QVBoxLayout(self.gdb_layer_group)
        gdb_layer_vbox.setSpacing(6)
        gdb_layer_vbox.setContentsMargins(8, 14, 8, 8)

        self.gdb_pick_mode_button = QPushButton("Pick GDB Points: OFF")
        self.gdb_pick_mode_button.setCheckable(True)
        self.gdb_pick_mode_button.setToolTip(
            "Toggle GDB pick mode. When ON, clicking a GDB sphere shows its attributes "
            "from the geodatabase. When OFF, normal tree point picking is used."
        )
        self.gdb_pick_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #363636; border: 1px solid #555555; color: #e0e0e0;
                padding: 4px 10px; font-size: 10px; border-radius: 0px;
            }
            QPushButton:checked {
                background-color: #1976D2; border: 1px solid #1976D2; color: white;
            }
        """)
        self.gdb_pick_mode_button.clicked.connect(self._on_gdb_pick_mode_toggled)
        gdb_layer_vbox.addWidget(self.gdb_pick_mode_button)

        self.gdb_layers_scroll = QScrollArea()
        self.gdb_layers_scroll.setWidgetResizable(True)
        self.gdb_layers_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gdb_layers_scroll.setMinimumHeight(150)
        self.gdb_layers_scroll.setMaximumHeight(260)
        self.gdb_layers_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.gdb_layers_container = QWidget()
        self.gdb_layers_container_layout = QVBoxLayout(self.gdb_layers_container)
        self.gdb_layers_container_layout.setSpacing(4)
        self.gdb_layers_container_layout.setContentsMargins(0, 0, 0, 0)
        self.gdb_layers_container_layout.addStretch()
        self.gdb_layers_scroll.setWidget(self.gdb_layers_container)
        gdb_layer_vbox.addWidget(self.gdb_layers_scroll)

        # Volume Folder
        self.volume_folder_label = QLabel("Auto-assigned")
        self.volume_folder_label.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 11px;")
        self.volume_folder_label.setWordWrap(True)
        self.volume_folder_checkbox = QCheckBox("Check volume folder on visualize")
        self.volume_folder_checkbox.setChecked(False)
        self.volume_folder_checkbox.setToolTip("When enabled, tree IDs with saved volume data in the selected folder will be colored red")
        self.volume_folder_checkbox.stateChanged.connect(self.on_volume_folder_checkbox_changed)

        # Tree selection controls
        self.tree_search_input = QLineEdit()
        self.tree_search_input.setPlaceholderText(" Filter Tree IDs...")
        self.tree_search_input.setFixedHeight(26)
        self.tree_search_input.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                border: 1px solid #4d4d4d;
                border-radius: 0px;
                color: #e0e0e0;
                padding: 2px 6px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        self.tree_search_input.textChanged.connect(self._filter_tree_list)

        self.tree_list = QListWidget()
        self.tree_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.tree_list.setMinimumHeight(180)
        self.tree_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 0px;
                color: #ffffff;
                selection-background-color: #2E7D32;
            }
            QListWidget::item {
                padding: 4px 6px;
                border-bottom: 1px solid #333333;
            }
            QListWidget::item:selected {
                background-color: #2E7D32;
                color: #ffffff;
                font-weight: bold;
            }
        """)

        self.select_all_button = ModernButton("Select All")
        self.select_all_button.clicked.connect(self.select_all_trees)
        self.select_all_button.setFixedHeight(26)
        self.select_all_button.setStyleSheet(btn_small_style)

        self.clear_selection_button = ModernButton("Clear All")
        self.clear_selection_button.clicked.connect(self.clear_tree_selection)
        self.clear_selection_button.setFixedHeight(26)
        self.clear_selection_button.setStyleSheet(btn_small_style)

        self.prev_tree_button = ModernButton("< Prev Tree")
        self.prev_tree_button.clicked.connect(self.prev_tree)
        self.prev_tree_button.setEnabled(False)
        self.prev_tree_button.setFixedHeight(26)
        self.prev_tree_button.setToolTip("Select and visualize the previous tree in the list")
        self.prev_tree_button.setStyleSheet("""
            QPushButton {
                background-color: #37474F;
                color: #eceff1;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #455A64;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #455A64;
                color: #ffffff;
            }
            QPushButton:disabled {
                background-color: #263238;
                color: #546E7A;
                border-color: #37474F;
            }
        """)

        self.next_tree_button = ModernButton("Next Tree >")
        self.next_tree_button.clicked.connect(self.next_tree)
        self.next_tree_button.setEnabled(False)
        self.next_tree_button.setFixedHeight(26)
        self.next_tree_button.setToolTip("Advance to and immediately visualize the next tree in the list")
        self.next_tree_button.setStyleSheet("""
            QPushButton {
                background-color: #1B5E20;
                color: #E8F5E9;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #2E7D32;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #2E7D32;
                color: #ffffff;
            }
            QPushButton:disabled {
                background-color: #1e2e1e;
                color: #4a6a4a;
                border-color: #2d3d2d;
            }
        """)

        self.visualize_button = ModernButton(" Visualize Selected Tree(s)")
        self.visualize_button.clicked.connect(self.visualize_tree)
        self.visualize_button.setEnabled(False)
        self.visualize_button.setFixedHeight(34)
        self.visualize_button.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                border: 1px solid #388E3C;
                color: white;
                font-size: 11px;
                font-weight: bold;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #777777;
                border: 1px solid #555555;
            }
        """)

        # View / Display inputs
        self.color_combo = QComboBox()
        self.color_combo.addItem("Default (green)")
        self.color_combo.setFixedHeight(26)
        self.color_combo.currentTextChanged.connect(self.update_tree_colors)

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(2)
        self.point_size_spin.setToolTip("Point rendering size in pixels (1-20)")
        self.point_size_spin.setFixedHeight(26)
        self.point_size_spin.setFixedWidth(80)
        self.point_size_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.point_size_spin.valueChanged.connect(self._on_point_style_changed)

        self.point_shape_combo = QComboBox()
        self.point_shape_combo.addItem("Square", "square")
        self.point_shape_combo.addItem("Circle / Sphere", "sphere")
        self.point_shape_combo.setToolTip("Point rendering shape:\nSquare = 2D billboard quads (fast, sharp)\nCircle / Sphere = 3D shaded spheres (smooth, rounded)")
        self.point_shape_combo.setFixedHeight(26)
        self.point_shape_combo.setMinimumWidth(115)
        self.point_shape_combo.currentIndexChanged.connect(self._on_point_style_changed)

        self.filter_checkbox = QCheckBox("Filter tree points")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.setToolTip("Keep only tree points (field/value auto-detected per data output) in visualization and analysis")
        self.filter_checkbox.stateChanged.connect(self.on_filter_changed)

        self.tree_filter_field_combo = QComboBox()
        self.tree_filter_field_combo.addItem("treefilter", "treefilter")
        self.tree_filter_field_combo.addItem("stemcls", "stemcls")
        self.tree_filter_field_combo.addItem("semantic_pred", "semantic_pred")
        self.tree_filter_field_combo.addItem("No filter", "__NONE__")
        self.tree_filter_field_combo.setToolTip("Scalar field marking tree points (auto-selected on load).\nTreeAIBox: treefilter (tree=2) · TreeLearn: stemcls (stem=2) · semantic segmentation: semantic_pred (0=ground, 1=trunk, 2=foliage)")
        self.tree_filter_field_combo.setFixedHeight(26)
        self.tree_filter_field_combo.setCurrentIndex(self.tree_filter_field_combo.findData("__NONE__"))
        self.tree_filter_field_combo.currentIndexChanged.connect(self._on_tree_filter_field_changed)

        self.tree_filter_value_spin = QSpinBox()
        self.tree_filter_value_spin.setRange(0, 255)
        self.tree_filter_value_spin.setValue(2)
        self.tree_filter_value_spin.setToolTip("Field value that marks tree points (default: 2)")
        self.tree_filter_value_spin.setFixedHeight(26)
        self.tree_filter_value_spin.setFixedWidth(80)
        self.tree_filter_value_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tree_filter_value_spin.valueChanged.connect(self._on_tree_filter_value_changed)

        self.crown_polygon_button = ModernButton("Crown Polygon")
        self.crown_polygon_button.clicked.connect(self.create_tree_crown_polygon)
        self.crown_polygon_button.setEnabled(False)
        self.crown_polygon_button.setFixedHeight(26)
        self.crown_polygon_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.neighbor_radius_input = QLineEdit("20.0")
        self.neighbor_radius_input.setFixedWidth(65)
        self.neighbor_radius_input.setFixedHeight(26)
        self.neighbor_radius_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.neighbor_radius_input.setToolTip("Radius in meters to search for neighboring trees")
        self.view_neighbors_button = ModernButton("Neighbour")
        self.view_neighbors_button.clicked.connect(self.view_neighbors)
        self.view_neighbors_button.setEnabled(False)
        self.view_neighbors_button.setFixedHeight(26)
        self.view_neighbors_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.fit_to_screen_button = ModernButton("Fit View")
        self.fit_to_screen_button.clicked.connect(self.fit_to_screen)
        self.fit_to_screen_button.setEnabled(False)
        self.fit_to_screen_button.setFixedSize(58, 26)
        self.fit_to_screen_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 4px; }")

        self.clear_button = ModernButton("Clear View")
        self.clear_button.clicked.connect(self.clear_plot)
        self.clear_button.setFixedSize(65, 26)
        self.clear_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 4px; }")

        # Selection controls
        self.draw_polygon_button = ModernButton("Draw Polygon")
        self.draw_polygon_button.clicked.connect(self.on_draw_polygon_button_clicked)
        self.draw_polygon_button.setEnabled(False)
        self.draw_polygon_button.setFixedHeight(26)
        self.draw_polygon_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")
        self.draw_polygon_button.setToolTip("Draw a polygon to select points and assign new ITC values")
        self.polygon_selection_active = False

        self.select_all_points_button = ModernButton("Select All Points")
        self.select_all_points_button.clicked.connect(self.select_all_points)
        self.select_all_points_button.setEnabled(False)
        self.select_all_points_button.setFixedHeight(26)
        self.select_all_points_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.assign_itc_button = ModernButton("Assign ITC")
        self.assign_itc_button.clicked.connect(lambda: self.assign_itc_to_selection())
        self.assign_itc_button.setEnabled(False)
        self.assign_itc_button.setFixedHeight(26)
        self.assign_itc_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.radius_value_input = QDoubleSpinBox()
        self.radius_value_input.setMinimum(0.05)
        self.radius_value_input.setMaximum(100.0)
        self.radius_value_input.setSingleStep(0.1)
        self.radius_value_input.setDecimals(2)
        self.radius_value_input.setValue(1.0)
        self.radius_value_input.setSuffix(" m")
        self.radius_value_input.setFixedWidth(80)
        self.radius_value_input.setFixedHeight(26)
        self.radius_value_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.radius_export_button = ModernButton("Pick & Export Radius")
        self.radius_export_button.clicked.connect(self.on_radius_export_button_clicked)
        self.radius_export_button.setEnabled(False)
        self.radius_export_button.setFixedHeight(26)
        self.radius_export_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        # Trunk & Branch inputs
        self.trunk_slice_height_input = QSpinBox()
        self.trunk_slice_height_input.setMinimum(5)
        self.trunk_slice_height_input.setMaximum(50)
        self.trunk_slice_height_input.setValue(25)
        self.trunk_slice_height_input.setSuffix(" cm")
        self.trunk_slice_height_input.setFixedWidth(80)
        self.trunk_slice_height_input.setFixedHeight(26)
        self.trunk_slice_height_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_base_zone_input = QDoubleSpinBox()
        self.trunk_base_zone_input.setMinimum(0.0)
        self.trunk_base_zone_input.setMaximum(50.0)
        self.trunk_base_zone_input.setSingleStep(0.1)
        self.trunk_base_zone_input.setDecimals(2)
        self.trunk_base_zone_input.setValue(0.0)
        self.trunk_base_zone_input.setSuffix(" m")
        self.trunk_base_zone_input.setFixedWidth(80)
        self.trunk_base_zone_input.setFixedHeight(26)
        self.trunk_base_zone_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_eps_input = QDoubleSpinBox()
        self.trunk_eps_input.setMinimum(0.01)
        self.trunk_eps_input.setMaximum(5.0)
        self.trunk_eps_input.setSingleStep(0.01)
        self.trunk_eps_input.setDecimals(2)
        self.trunk_eps_input.setValue(0.18)
        self.trunk_eps_input.setSuffix(" m")
        self.trunk_eps_input.setFixedWidth(80)
        self.trunk_eps_input.setFixedHeight(26)
        self.trunk_eps_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_min_samples_input = QSpinBox()
        self.trunk_min_samples_input.setMinimum(1)
        self.trunk_min_samples_input.setMaximum(1000)
        self.trunk_min_samples_input.setValue(8)
        self.trunk_min_samples_input.setFixedWidth(80)
        self.trunk_min_samples_input.setFixedHeight(26)
        self.trunk_min_samples_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_merge_dist_input = QDoubleSpinBox()
        self.trunk_merge_dist_input.setMinimum(0.01)
        self.trunk_merge_dist_input.setMaximum(5.0)
        self.trunk_merge_dist_input.setSingleStep(0.01)
        self.trunk_merge_dist_input.setDecimals(2)
        self.trunk_merge_dist_input.setValue(0.25)
        self.trunk_merge_dist_input.setSuffix(" m")
        self.trunk_merge_dist_input.setFixedWidth(80)
        self.trunk_merge_dist_input.setFixedHeight(26)
        self.trunk_merge_dist_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_min_points_input = QSpinBox()
        self.trunk_min_points_input.setMinimum(1)
        self.trunk_min_points_input.setMaximum(100000)
        self.trunk_min_points_input.setValue(120)
        self.trunk_min_points_input.setFixedWidth(80)
        self.trunk_min_points_input.setFixedHeight(26)
        self.trunk_min_points_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_min_span_input = QDoubleSpinBox()
        self.trunk_min_span_input.setMinimum(0.0)
        self.trunk_min_span_input.setMaximum(50.0)
        self.trunk_min_span_input.setSingleStep(0.1)
        self.trunk_min_span_input.setDecimals(2)
        self.trunk_min_span_input.setValue(1.0)
        self.trunk_min_span_input.setSuffix(" m")
        self.trunk_min_span_input.setFixedWidth(80)
        self.trunk_min_span_input.setFixedHeight(26)
        self.trunk_min_span_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trunk_full_assign_input = QDoubleSpinBox()
        self.trunk_full_assign_input.setMinimum(0.01)
        self.trunk_full_assign_input.setMaximum(10.0)
        self.trunk_full_assign_input.setSingleStep(0.01)
        self.trunk_full_assign_input.setDecimals(2)
        self.trunk_full_assign_input.setValue(0.45)
        self.trunk_full_assign_input.setSuffix(" m")
        self.trunk_full_assign_input.setFixedWidth(80)
        self.trunk_full_assign_input.setFixedHeight(26)
        self.trunk_full_assign_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.detect_trunks_button = ModernButton("Detect Trunks")
        self.detect_trunks_button.clicked.connect(self.detect_trunks_in_slices)
        self.detect_trunks_button.setEnabled(False)
        self.detect_trunks_button.setFixedHeight(26)
        self.detect_trunks_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.raw_trunk_view_checkbox = QCheckBox("Raw")
        self.raw_trunk_view_checkbox.setChecked(False)
        self.raw_trunk_view_checkbox.setToolTip("Show raw base-zone trunk detections instead of filtered trunks")

        self.trunk_button = ModernButton("Trunk Visualization")
        self.trunk_button.clicked.connect(self.visualize_trunk)
        self.trunk_button.setEnabled(False)
        self.trunk_button.setFixedHeight(26)
        self.trunk_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.slice_height_label = QLabel("Slice:")
        self.slice_height_label.setStyleSheet("font-size: 10px;")
        self.slice_height_input = QSpinBox()
        self.slice_height_input.setMinimum(5)
        self.slice_height_input.setMaximum(50)
        self.slice_height_input.setValue(25)
        self.slice_height_input.setSuffix(" cm")
        self.slice_height_input.setFixedWidth(80)
        self.slice_height_input.setFixedHeight(26)
        self.slice_height_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.branch_eps_input = QDoubleSpinBox()
        self.branch_eps_input.setMinimum(0.01)
        self.branch_eps_input.setMaximum(5.0)
        self.branch_eps_input.setSingleStep(0.01)
        self.branch_eps_input.setDecimals(2)
        self.branch_eps_input.setValue(0.08)
        self.branch_eps_input.setSuffix(" m")
        self.branch_eps_input.setFixedWidth(80)
        self.branch_eps_input.setFixedHeight(26)
        self.branch_eps_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.branch_min_samples_input = QSpinBox()
        self.branch_min_samples_input.setMinimum(1)
        self.branch_min_samples_input.setMaximum(1000)
        self.branch_min_samples_input.setValue(3)
        self.branch_min_samples_input.setFixedWidth(80)
        self.branch_min_samples_input.setFixedHeight(26)
        self.branch_min_samples_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.branch_merge_dist_input = QDoubleSpinBox()
        self.branch_merge_dist_input.setMinimum(0.01)
        self.branch_merge_dist_input.setMaximum(5.0)
        self.branch_merge_dist_input.setSingleStep(0.01)
        self.branch_merge_dist_input.setDecimals(2)
        self.branch_merge_dist_input.setValue(0.15)
        self.branch_merge_dist_input.setSuffix(" m")
        self.branch_merge_dist_input.setFixedWidth(80)
        self.branch_merge_dist_input.setFixedHeight(26)
        self.branch_merge_dist_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.detect_branches_button = ModernButton("Detect Branches")
        self.detect_branches_button.clicked.connect(self.detect_branch_clusters_in_slices)
        self.detect_branches_button.setEnabled(False)
        self.detect_branches_button.setFixedHeight(26)
        self.detect_branches_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        # Volume controls
        self.dbh_height_input = QDoubleSpinBox()
        self.dbh_height_input.setRange(0.1, 10.0)
        self.dbh_height_input.setSingleStep(0.1)
        self.dbh_height_input.setDecimals(2)
        self.dbh_height_input.setValue(1.3)
        self.dbh_height_input.setSuffix(" m")
        self.dbh_height_input.setFixedWidth(80)
        self.dbh_height_input.setFixedHeight(26)
        self.dbh_height_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dbh_height_input.valueChanged.connect(self._on_dbh_height_changed)

        self.volume_calc_button = ModernButton("Calculate Volume")
        self.volume_calc_button.clicked.connect(self.calculate_volume_from_selection)
        self.volume_calc_button.setEnabled(False)
        self.volume_calc_button.setFixedHeight(26)
        self.volume_calc_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 8px; font-weight: bold; }")

        self.save_volume_button = ModernButton("Save Volume Data")
        self.save_volume_button.clicked.connect(self.save_volume_data)
        self.save_volume_button.setEnabled(False)
        self.save_volume_button.setFixedHeight(26)
        self.save_volume_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.load_volume_button = ModernButton("Load Volume Data")
        self.load_volume_button.clicked.connect(self.load_volume_data)
        self.load_volume_button.setEnabled(True)
        self.load_volume_button.setFixedHeight(26)
        self.load_volume_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.clear_volume_button = ModernButton("Clear Results")
        self.clear_volume_button.clicked.connect(self.clear_accumulated_volume)
        self.clear_volume_button.setEnabled(False)
        self.clear_volume_button.setFixedHeight(26)
        self.clear_volume_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        self.export_trunks_button = ModernButton("Export Trunks CSV")
        self.export_trunks_button.clicked.connect(self.export_trunk_metrics)
        self.export_trunks_button.setEnabled(False)
        self.export_trunks_button.setFixedHeight(26)
        self.export_trunks_button.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; }")

        # AI & Analysis
        self.analyze_button = ModernButton("AI Analyze Tree")
        self.analyze_button.clicked.connect(self.analyze_tree_with_ai)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setFixedHeight(26)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #7B1FA2;
                border: none;
                color: white;
                font-weight: bold;
                font-size: 10px;
                padding: 2px 8px;
                border-radius: 0px;
            }
            QPushButton:hover { background-color: #8E24AA; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)

        self.max_height_input = QLineEdit("8.0")
        self.max_height_input.setFixedWidth(65)
        self.max_height_input.setFixedHeight(26)
        self.max_height_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.step_input = QLineEdit("1.0")
        self.step_input.setFixedWidth(65)
        self.step_input.setFixedHeight(26)
        self.step_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.thickness_input = QLineEdit("0.1")
        self.thickness_input.setFixedWidth(65)
        self.thickness_input.setFixedHeight(26)
        self.thickness_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.min_points_input = QLineEdit("3")
        self.min_points_input.setFixedWidth(65)
        self.min_points_input.setFixedHeight(26)
        self.min_points_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.batch_analyze_button = ModernButton("Batch Analyze Trunks")
        self.batch_analyze_button.clicked.connect(self.batch_analyze_trunk_volumes)
        self.batch_analyze_button.setEnabled(False)
        self.batch_analyze_button.setFixedHeight(26)
        self.batch_analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #E65100;
                border: none;
                color: white;
                font-weight: bold;
                font-size: 10px;
                padding: 2px 8px;
                border-radius: 0px;
            }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setTextVisible(True)


    def create_header_bar(self):
        """Create the top brand and status header bar."""
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setStyleSheet("""
            #headerFrame {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 0px;
                padding: 2px 8px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(8, 4, 8, 4)
        h_layout.setSpacing(12)

        # Brand Title
        brand_label = QLabel("Forestry AI Suite")
        brand_label.setStyleSheet("color: #4CAF50; font-size: 15px; font-weight: bold; letter-spacing: 0.5px;")
        h_layout.addWidget(brand_label)

        h_layout.addStretch()

        # Active File Label
        file_container = QFrame()
        file_container.setStyleSheet("""
            background-color: #2b2b2b;
            border: 1px solid #444444;
            border-radius: 0px;
            padding: 2px 8px;
        """)
        file_layout = QHBoxLayout(file_container)
        file_layout.setContentsMargins(6, 2, 6, 2)
        file_layout.setSpacing(6)
        file_tag = QLabel("Active:")
        file_tag.setStyleSheet("color: #888888; font-size: 10px; font-weight: bold;")
        file_layout.addWidget(file_tag)
        file_layout.addWidget(self.file_label)
        h_layout.addWidget(file_container)

        # Quick action buttons
        h_layout.addWidget(self.load_button)
        h_layout.addWidget(self.save_button)

        return header_frame


    def create_top_ribbon(self):
        """Create the category-based top navigation ribbon."""
        ribbon_container = QWidget()
        v_layout = QVBoxLayout(ribbon_container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(0)

        # Top Category Tab Bar Row
        tab_bar_frame = QFrame()
        tab_bar_frame.setStyleSheet("background-color: #242424; border-bottom: 2px solid #388E3C;")
        tab_layout = QHBoxLayout(tab_bar_frame)
        tab_layout.setContentsMargins(4, 0, 4, 0)
        tab_layout.setSpacing(2)

        self.ribbon_button_group = QButtonGroup(self)
        self.ribbon_button_group.setExclusive(True)

        categories = [
            (" View & Display", 0),
            (" File & Layers", 1),
            (" Selection & Editing", 2),
            (" Trunk & Branch", 3),
            ("[Metrics] Volume & Metrics", 4),
            (" AI & Batch Analysis", 5)
        ]

        tab_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                color: #bbbbbb;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
            }
            QPushButton:hover {
                background-color: #2e2e2e;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #2d452d;
                border-bottom: 3px solid #4CAF50;
                color: #81c784;
            }
        """

        self.ribbon_tabs = []
        for title, index in categories:
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.setStyleSheet(tab_style)
            if index == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, idx=index: self._on_ribbon_tab_clicked(idx))
            self.ribbon_button_group.addButton(btn, index)
            tab_layout.addWidget(btn)
            self.ribbon_tabs.append(btn)

        tab_layout.addStretch()

        # Ribbon Collapse/Expand Toggle
        self.ribbon_toggle_btn = QPushButton("▲ Minimize")
        self.ribbon_toggle_btn.setToolTip("Toggle Ribbon toolbar visibility")
        self.ribbon_toggle_btn.setFixedHeight(24)
        self.ribbon_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                color: #aaaaaa;
                font-size: 10px;
                padding: 2px 8px;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                color: white;
            }
        """)
        self.ribbon_toggle_btn.clicked.connect(self.toggle_ribbon)
        tab_layout.addWidget(self.ribbon_toggle_btn)

        v_layout.addWidget(tab_bar_frame)

        # Stacked Widget with each Category's horizontal toolbar
        self.ribbon_stack = QStackedWidget()
        self.ribbon_stack.setFixedHeight(132)
        self.ribbon_stack.setStyleSheet("""
            QStackedWidget {
                background-color: #2c2c2c;
                border: 1px solid #3d3d3d;
                border-top: none;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
        """)

        # Add category pages
        self.ribbon_stack.addWidget(self.create_ribbon_view_page())
        self.ribbon_stack.addWidget(self.create_ribbon_file_page())
        self.ribbon_stack.addWidget(self.create_ribbon_selection_page())
        self.ribbon_stack.addWidget(self.create_ribbon_trunk_page())
        self.ribbon_stack.addWidget(self.create_ribbon_volume_page())
        self.ribbon_stack.addWidget(self.create_ribbon_ai_page())

        v_layout.addWidget(self.ribbon_stack)

        return ribbon_container


    def _create_ribbon_card(self, title):
        """Helper to create a neat titled card frame for ribbon panels."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #242424;
                border: 1px solid #3a3a3a;
                border-radius: 0px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(6, 4, 6, 4)
        card_layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color: #4CAF50; font-size: 9px; font-weight: bold; text-transform: uppercase;")
        card_layout.addWidget(title_lbl)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        card_layout.addWidget(content_widget, 1)

        return card, content_layout


    def create_ribbon_file_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Card 1: LAS Cloud Files
        card1, c1_layout = self._create_ribbon_card("LAS Layer Actions")
        c1_layout.addWidget(self.compare_las_btn)

        # Card 2: Geodatabase (GDB)
        card2, c2_layout = self._create_ribbon_card("Geodatabase (GDB)")
        c2_layout.addWidget(self.import_gdb_button)

        # Card 3: Volume Directory (Horizontal flow)
        card3, c3_layout = self._create_ribbon_card("Volume Data Directory")
        lbl_dir = QLabel("Folder:")
        lbl_dir.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c3_layout.addWidget(lbl_dir)
        c3_layout.addWidget(self.volume_folder_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #444444; margin: 2px 6px;")
        c3_layout.addWidget(sep)

        c3_layout.addWidget(self.volume_folder_checkbox)

        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addWidget(card3)
        layout.addStretch()
        return page


    def create_ribbon_view_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Card 1: Point Style & Color (Horizontal flow)
        card1, c1_layout = self._create_ribbon_card("Point Color & Style")
        lbl_color = QLabel("Color by:")
        lbl_color.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.color_combo.setMinimumWidth(150)
        self.color_combo.setMaximumWidth(180)
        c1_layout.addWidget(lbl_color)
        c1_layout.addWidget(self.color_combo)

        # Subtle vertical separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #444444; margin: 2px 4px;")
        c1_layout.addWidget(sep1)

        lbl_size = QLabel("Size:")
        lbl_size.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c1_layout.addWidget(lbl_size)
        c1_layout.addWidget(self.point_size_spin)

        c1_layout.addSpacing(6)

        lbl_shape = QLabel("Shape:")
        lbl_shape.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c1_layout.addWidget(lbl_shape)
        c1_layout.addWidget(self.point_shape_combo)

        # Card 2: Point Filtering & Terrain (Horizontal flow)
        card2, c2_layout = self._create_ribbon_card("Point Filtering & Terrain")
        self.filter_checkbox.setStyleSheet("color: #cccccc; font-size: 11px;")
        c2_layout.addWidget(self.filter_checkbox)

        c2_layout.addSpacing(6)

        lbl_field = QLabel("Field:")
        lbl_field.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.tree_filter_field_combo.setMinimumWidth(110)
        self.tree_filter_field_combo.setMaximumWidth(135)
        c2_layout.addWidget(lbl_field)
        c2_layout.addWidget(self.tree_filter_field_combo)

        c2_layout.addSpacing(6)

        lbl_val = QLabel("Value:")
        lbl_val.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c2_layout.addWidget(lbl_val)
        c2_layout.addWidget(self.tree_filter_value_spin)

        # Subtle vertical separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #444444; margin: 2px 4px;")
        c2_layout.addWidget(sep2)

        self._ribbon_ground_cb = QCheckBox("Show Terrain Context")
        self._ribbon_ground_cb.setChecked(False)
        self._ribbon_ground_cb.setToolTip("Display terrain/ground points (ASPRS Class 2 or ITC 0 surface) beneath the selected tree")
        self._ribbon_ground_cb.setStyleSheet("color: #00E676; font-size: 11px;")
        self._ribbon_ground_cb.stateChanged.connect(self._on_show_ground_toggled)
        c2_layout.addWidget(self._ribbon_ground_cb)

        # Card 3: Spatial Crown & Context (Horizontal flow)
        card3, c3_layout = self._create_ribbon_card("Crown & Neighbors")
        lbl_rad = QLabel("Radius:")
        lbl_rad.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.neighbor_radius_input.setFixedWidth(65)
        c3_layout.addWidget(lbl_rad)
        c3_layout.addWidget(self.neighbor_radius_input)
        c3_layout.addWidget(self.view_neighbors_button)

        # Subtle vertical separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet("color: #444444; margin: 2px 4px;")
        c3_layout.addWidget(sep3)

        c3_layout.addWidget(self.crown_polygon_button)

        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addWidget(card3)
        layout.addStretch()
        return page


    def create_ribbon_selection_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Card 1: Point Selection & ITC (Horizontal flow)
        card1, c1_layout = self._create_ribbon_card("Polygon & Point Picking")
        c1_layout.addWidget(self.draw_polygon_button)
        c1_layout.addWidget(self.select_all_points_button)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #444444; margin: 2px 4px;")
        c1_layout.addWidget(sep1)

        c1_layout.addWidget(self.assign_itc_button)

        # Card 2: Radius Selection & Export (Horizontal flow)
        card2, c2_layout = self._create_ribbon_card("Radius Selection & Export")
        lbl_rad = QLabel("Radius:")
        lbl_rad.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c2_layout.addWidget(lbl_rad)
        c2_layout.addWidget(self.radius_value_input)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #444444; margin: 2px 4px;")
        c2_layout.addWidget(sep2)

        c2_layout.addWidget(self.radius_export_button)

        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addStretch()
        return page


    def create_ribbon_trunk_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        def _make_param_lbl(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return lbl

        # Card 1: Trunk Separation Parameters & Actions
        card1, c1_layout = self._create_ribbon_card("Trunk Separation (Base Zone Slicing)")
        grid1 = QGridLayout()
        grid1.setSpacing(4)
        grid1.addWidget(_make_param_lbl("Slice:"), 0, 0)
        grid1.addWidget(self.trunk_slice_height_input, 0, 1)
        grid1.addWidget(_make_param_lbl("Base:"), 0, 2)
        grid1.addWidget(self.trunk_base_zone_input, 0, 3)
        grid1.addWidget(_make_param_lbl("eps:"), 0, 4)
        grid1.addWidget(self.trunk_eps_input, 0, 5)
        grid1.addWidget(_make_param_lbl("minS:"), 0, 6)
        grid1.addWidget(self.trunk_min_samples_input, 0, 7)

        grid1.addWidget(_make_param_lbl("merge:"), 1, 0)
        grid1.addWidget(self.trunk_merge_dist_input, 1, 1)
        grid1.addWidget(_make_param_lbl("minPts:"), 1, 2)
        grid1.addWidget(self.trunk_min_points_input, 1, 3)
        grid1.addWidget(_make_param_lbl("span:"), 1, 4)
        grid1.addWidget(self.trunk_min_span_input, 1, 5)
        grid1.addWidget(_make_param_lbl("extend:"), 1, 6)
        grid1.addWidget(self.trunk_full_assign_input, 1, 7)
        c1_layout.addLayout(grid1)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #444444; margin: 2px 6px;")
        c1_layout.addWidget(sep1)

        # Action Buttons for Trunk
        v_acts = QVBoxLayout()
        v_acts.setSpacing(4)
        h_row = QHBoxLayout()
        h_row.setSpacing(6)
        h_row.addWidget(self.detect_trunks_button)
        h_row.addWidget(self.raw_trunk_view_checkbox)
        v_acts.addLayout(h_row)
        v_acts.addWidget(self.trunk_button)
        c1_layout.addLayout(v_acts)

        # Card 2: Branch Detection
        card2, c2_layout = self._create_ribbon_card("Branch Detection")
        grid2 = QGridLayout()
        grid2.setSpacing(4)
        grid2.addWidget(_make_param_lbl("Slice:"), 0, 0)
        grid2.addWidget(self.slice_height_input, 0, 1)
        grid2.addWidget(_make_param_lbl("eps:"), 0, 2)
        grid2.addWidget(self.branch_eps_input, 0, 3)
        grid2.addWidget(_make_param_lbl("minS:"), 1, 0)
        grid2.addWidget(self.branch_min_samples_input, 1, 1)
        grid2.addWidget(_make_param_lbl("merge:"), 1, 2)
        grid2.addWidget(self.branch_merge_dist_input, 1, 3)
        c2_layout.addLayout(grid2)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #444444; margin: 2px 6px;")
        c2_layout.addWidget(sep2)

        c2_layout.addWidget(self.detect_branches_button)

        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addStretch()
        return page


    def create_ribbon_volume_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Card 1: Volume Calculation & DBH (Horizontal flow)
        card1, c1_layout = self._create_ribbon_card("Volume Analysis & DBH")
        lbl_dbh = QLabel("DBH Height:")
        lbl_dbh.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c1_layout.addWidget(lbl_dbh)
        c1_layout.addWidget(self.dbh_height_input)
        c1_layout.addWidget(self.volume_calc_button)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #444444; margin: 2px 4px;")
        c1_layout.addWidget(sep1)

        c1_layout.addWidget(self.export_trunks_button)

        # Card 2: Volume Data Storage (Horizontal flow)
        card2, c2_layout = self._create_ribbon_card("Volume Data Storage")
        c2_layout.addWidget(self.save_volume_button)
        c2_layout.addWidget(self.load_volume_button)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #444444; margin: 2px 4px;")
        c2_layout.addWidget(sep2)

        c2_layout.addWidget(self.clear_volume_button)

        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addStretch()
        return page


    def create_ribbon_ai_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Card 1: AI Model Segmentation
        card1, c1_layout = self._create_ribbon_card("AI Segmentation Suite")
        c1_layout.addWidget(self.analyze_button)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #444444; margin: 2px 4px;")
        c1_layout.addWidget(sep1)

        lbl = QLabel("Multi-Model Suite\n(TreeAIBox · TreeLearn · ForestFormer)")
        lbl.setStyleSheet("color: #888888; font-size: 9px;")
        c1_layout.addWidget(lbl)

        # Card 2: Split Detection (Horizontal flow)
        card2, c2_layout = self._create_ribbon_card("Split Detection Parameters")
        lbl_maxh = QLabel("MaxH:")
        lbl_maxh.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c2_layout.addWidget(lbl_maxh)
        c2_layout.addWidget(self.max_height_input)

        lbl_step = QLabel("Step:")
        lbl_step.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c2_layout.addWidget(lbl_step)
        c2_layout.addWidget(self.step_input)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #444444; margin: 2px 4px;")
        c2_layout.addWidget(sep2)

        lbl_thick = QLabel("Thick:")
        lbl_thick.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c2_layout.addWidget(lbl_thick)
        c2_layout.addWidget(self.thickness_input)

        lbl_minpts = QLabel("MinPts:")
        lbl_minpts.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        c2_layout.addWidget(lbl_minpts)
        c2_layout.addWidget(self.min_points_input)

        # Card 3: Batch Analysis
        card3, c3_layout = self._create_ribbon_card("Batch Processing")
        c3_layout.addWidget(self.batch_analyze_button)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet("color: #444444; margin: 2px 4px;")
        c3_layout.addWidget(sep3)

        hint = QLabel("Iterates all trees in\nchecked layers")
        hint.setStyleSheet("color: #888888; font-size: 9px;")
        c3_layout.addWidget(hint)

        layout.addWidget(card1)
        layout.addWidget(card2)
        layout.addWidget(card3)
        layout.addStretch()
        return page


    def create_left_panel(self):
        """Create the left panel with full Layers view and slim bookmark tab."""
        self.left_panel_widget = QWidget()
        self.left_panel_widget.setObjectName("leftPanelWidget")
        self.left_panel_widget.setStyleSheet("""
            #leftPanelWidget {
                background-color: #262626;
                border: 1px solid #383838;
                border-radius: 0px;
            }
        """)
        self.left_panel_widget.setMinimumWidth(280)
        self.left_panel_widget.setMaximumWidth(360)

        panel_layout = QHBoxLayout(self.left_panel_widget)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # -------------------------------------------------------------
        # 1. Expanded Container (visible when panel is open)
        # -------------------------------------------------------------
        self.expanded_container = QWidget()
        expanded_layout = QVBoxLayout(self.expanded_container)
        expanded_layout.setContentsMargins(8, 8, 8, 8)
        expanded_layout.setSpacing(8)

        # Header: Clean title "Layers" + Collapse Button
        header_layout = QHBoxLayout()
        header_title = QLabel("Layers")
        header_title.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 13px; letter-spacing: 0.5px;")
        header_layout.addWidget(header_title)
        header_layout.addStretch()

        self.panel_toggle_btn = QPushButton("Collapse")
        self.panel_toggle_btn.setToolTip("Collapse Layers panel into bookmark tab")
        self.panel_toggle_btn.setFixedSize(65, 22)
        self.panel_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                border: 1px solid #4a4a4a;
                color: #cccccc;
                font-size: 10px;
                font-weight: 600;
                border-radius: 0px;
                padding: 1px 4px;
            }
            QPushButton:hover {
                background-color: #444444;
                color: #ffffff;
                border-color: #555555;
            }
        """)
        self.panel_toggle_btn.clicked.connect(self.toggle_left_panel)
        header_layout.addWidget(self.panel_toggle_btn)
        expanded_layout.addLayout(header_layout)

        # Scrollable area for cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)

        # --- Card 1: LAS Layers (Placed on top of ITC box) ---
        las_card = QFrame()
        las_card.setStyleSheet("""
            QFrame {
                background-color: #202020;
                border: 1px solid #363636;
                border-radius: 0px;
            }
        """)
        las_vbox = QVBoxLayout(las_card)
        las_vbox.setContentsMargins(8, 8, 8, 8)
        las_vbox.setSpacing(6)

        las_header = QLabel("LAS Layers")
        las_header.setStyleSheet("""
            color: #4CAF50;
            font-size: 11px;
            font-weight: bold;
            padding-bottom: 3px;
            border-bottom: 1px solid #333333;
        """)
        las_vbox.addWidget(las_header)

        las_btn_row = QHBoxLayout()
        las_btn_row.setSpacing(4)
        las_btn_row.addWidget(self.add_layer_btn)
        las_btn_row.addWidget(self.remove_las_btn)
        las_btn_row.addWidget(self.toggle_las_batch_btn)
        las_vbox.addLayout(las_btn_row)

        las_vbox.addWidget(self.las_layer_list)
        cards_layout.addWidget(las_card)

        # --- Card 2: ITC Trees ---
        itc_card = QFrame()
        itc_card.setStyleSheet("""
            QFrame {
                background-color: #202020;
                border: 1px solid #363636;
                border-radius: 0px;
            }
        """)
        itc_vbox = QVBoxLayout(itc_card)
        itc_vbox.setContentsMargins(8, 8, 8, 8)
        itc_vbox.setSpacing(6)

        itc_header = QLabel("ITC Trees")
        itc_header.setStyleSheet("""
            color: #4CAF50;
            font-size: 11px;
            font-weight: bold;
            padding-bottom: 3px;
            border-bottom: 1px solid #333333;
        """)
        itc_vbox.addWidget(itc_header)

        itc_vbox.addWidget(self.tree_search_input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.addWidget(self.select_all_button)
        btn_row.addWidget(self.clear_selection_button)
        itc_vbox.addLayout(btn_row)

        itc_vbox.addWidget(self.tree_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        nav_row.addWidget(self.prev_tree_button)
        nav_row.addWidget(self.next_tree_button)
        itc_vbox.addLayout(nav_row)

        itc_vbox.addWidget(self.visualize_button)
        cards_layout.addWidget(itc_card)

        # --- Card 3: GDB Overlay Layers ---
        cards_layout.addWidget(self.gdb_layer_group)
        cards_layout.addStretch()

        scroll_area.setWidget(cards_container)
        expanded_layout.addWidget(scroll_area, 1)

        panel_layout.addWidget(self.expanded_container)

        # -------------------------------------------------------------
        # 2. Bookmark Tab Container (visible when collapsed)
        # -------------------------------------------------------------
        self.bookmark_container = QWidget()
        self.bookmark_container.setObjectName("bookmarkContainer")
        self.bookmark_container.setFixedWidth(28)
        self.bookmark_container.setStyleSheet("""
            #bookmarkContainer {
                background-color: #222222;
                border: 1px solid #3a3a3a;
                border-left: 3px solid #4CAF50;
                border-top-right-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            #bookmarkContainer:hover {
                background-color: #2a2a2a;
                border-color: #4CAF50;
            }
        """)
        bookmark_layout = QVBoxLayout(self.bookmark_container)
        bookmark_layout.setContentsMargins(2, 8, 2, 8)
        bookmark_layout.setSpacing(10)
        bookmark_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Expand arrow button at top of bookmark
        expand_arrow = QPushButton(">")
        expand_arrow.setFixedSize(22, 22)
        expand_arrow.setToolTip("Click to expand Layers panel")
        expand_arrow.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                border: none;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        expand_arrow.clicked.connect(self.toggle_left_panel)
        bookmark_layout.addWidget(expand_arrow)

        # Vertical text label "L A Y E R S"
        vertical_text = QLabel("L\nA\nY\nE\nR\nS")
        vertical_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vertical_text.setStyleSheet("""
            color: #81c784;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 2px;
            border: none;
            background: transparent;
        """)
        bookmark_layout.addWidget(vertical_text)

        bookmark_layout.addStretch()

        # Connect entire bookmark container click to toggle
        self.bookmark_container.mousePressEvent = lambda event: self.toggle_left_panel()
        self.bookmark_container.setCursor(Qt.CursorShape.PointingHandCursor)
        self.bookmark_container.setToolTip("Click to expand Layers panel")

        self.bookmark_container.setVisible(False)
        panel_layout.addWidget(self.bookmark_container)

        return self.left_panel_widget


    def create_console_widget(self):
        """Create the bottom console log panel."""
        console_frame = QFrame()
        console_frame.setObjectName("consoleFrame")
        console_frame.setStyleSheet("""
            #consoleFrame {
                background-color: #1a1a1a;
                border: 1px solid #3d3d3d;
                border-radius: 0px;
            }
        """)
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(6, 4, 6, 4)
        console_layout.setSpacing(4)

        header_row = QHBoxLayout()
        console_label = QLabel("Console Output:")
        console_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
        header_row.addWidget(console_label)
        header_row.addStretch()

        self.console_toggle_btn = QPushButton("Hide")
        self.console_toggle_btn.setFixedSize(45, 18)
        self.console_toggle_btn.setStyleSheet("background: #2b2b2b; color: #aaaaaa; border: 1px solid #444; border-radius: 0px; font-size: 9px;")
        self.console_toggle_btn.clicked.connect(self.toggle_console)
        header_row.addWidget(self.console_toggle_btn)
        console_layout.addLayout(header_row)

        self.console = QTextEdit()
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #222222;
                color: #d0d0d0;
                border: 1px solid #444444;
                border-radius: 0px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(100)
        self.console.setMinimumHeight(60)
        console_layout.addWidget(self.console)

        self._console_frame = console_frame
        return console_frame


    def _on_ribbon_tab_clicked(self, index):
        """Switch active ribbon category page."""
        self.ribbon_stack.setCurrentIndex(index)
        if getattr(self, '_ribbon_collapsed', False):
            self.toggle_ribbon()


    def toggle_ribbon(self):
        """Toggle ribbon toolbar visibility."""
        is_collapsed = getattr(self, '_ribbon_collapsed', False)
        if not is_collapsed:
            self.ribbon_stack.setVisible(False)
            self.ribbon_toggle_btn.setText("Expand Toolbar")
            self._ribbon_collapsed = True
        else:
            self.ribbon_stack.setVisible(True)
            self.ribbon_toggle_btn.setText("Minimize")
            self._ribbon_collapsed = False


    def toggle_left_panel(self):
        """Toggle left panel between full Layers view and slim bookmark tab."""
        is_collapsed = getattr(self, '_panel_collapsed', False)
        total_w = self.center_splitter.width() if hasattr(self, 'center_splitter') else 1400
        if total_w < 100:
            total_w = 1400

        if not is_collapsed:
            # Collapse to bookmark tab
            self.expanded_container.setVisible(False)
            self.bookmark_container.setVisible(True)
            self.left_panel_widget.setMinimumWidth(28)
            self.left_panel_widget.setMaximumWidth(28)
            self.center_splitter.setSizes([28, total_w - 28])
            self._panel_collapsed = True
        else:
            # Expand to full Layers panel
            self.bookmark_container.setVisible(False)
            self.expanded_container.setVisible(True)
            self.left_panel_widget.setMinimumWidth(280)
            self.left_panel_widget.setMaximumWidth(380)
            self.center_splitter.setSizes([310, total_w - 310])
            self._panel_collapsed = False


    def toggle_console(self):
        """Toggle console output visibility."""
        is_visible = self.console.isVisible()
        self.console.setVisible(not is_visible)
        self.console_toggle_btn.setText("Show" if not is_visible else "Hide")


    def _filter_tree_list(self, text):
        """Filter items in tree_list by search query."""
        query = text.strip().lower()
        for i in range(self.tree_list.count()):
            item = self.tree_list.item(i)
            if not query:
                item.setHidden(False)
            else:
                item.setHidden(query not in item.text().lower())


