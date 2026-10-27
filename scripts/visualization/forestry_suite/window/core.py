"""Core window state, logging helpers, camera, profiling. (TreeVisualizerGUI mixin)."""

import sys
import os
import time
import logging
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QDialogButtonBox, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt
try:
    from scripts.pointcloud_processing.las_utils import (
        detect_tree_filter_field,
        find_tree_id_dimension,
    )
except ImportError:
    from modules.las_utils import (
        detect_tree_filter_field,
        find_tree_id_dimension,
    )


class CoreMixin:
    """Core window state, logging helpers, camera, profiling."""

    def __init__(self):
        import time
        init_start = time.perf_counter()
        print(f"[INIT] {time.strftime('%H:%M:%S')} - TreeVisualizerGUI.__init__ starting...")
        
        super().__init__()
        print(f"[INIT] {time.strftime('%H:%M:%S')} - super().__init__ done ({time.perf_counter() - init_start:.3f}s)")
        
        self.las_data = None
        self.current_las_path = None
        self.las_layers = []  # List of layer dicts: {'path', 'name', 'num_points', 'num_trees', 'has_stemcls', 'checked'}
        self._updating_las_layer_list = False
        self.unique_itc_values = []
        self.current_tree_points = None
        self.current_tree_id = None
        self.current_mask = None
        self.current_color_values = None
        self.current_color_field = None
        self.all_masks = None
        self.current_itc_values = None
        self.plotter = None
        # GDB overlay layers: name -> {coords, color, point_size, visible, render_spheres}
        self.gdb_layers = {}
        # When True, the point picker resolves GDB overlay points instead of tree points
        self.gdb_pick_mode = False
        # Polygon selection variables
        self.polygon_selection_mode = False
        self.polygon_points = []
        self.selected_polygon = None
        self.selected_point_indices = None
        self.radius_selection_active = False
        self.radius_selection_center = None
        
        # Volume calculation accumulation variables
        self.accumulated_volume_sections = []  # List of (points, circles_data) tuples
        self.trees_with_volume_data = set()  # Track tree IDs that have volume calculation data
        self.volume_folder_path = None  # Path to folder containing volume data files
        self.itc_changes = []  # Track all ITC assignments in session
        self.last_analyzed_points = None  # Store points from last AI analysis for noise conversion
        self.tree_centroids_cache = {}  # Cache for tree centroids to speed up neighbor calculations
        self.centroids_kdtree = None  # KDTree for fast spatial queries
        self.centroids_kdtree_tree_ids = None  # Tree IDs in the same row order as centroids_kdtree
        
        # Trunk detection variables (Phase 1)
        self.trunk_assignment = None  # Array: -1 = noise, >=0 = trunk_id
        self.trunk_models = {}  # Dict: trunk_id -> {'trajectory': [centroids], 'points': count, 'z_span': (min, max)}
        self.trunk_detection_params = {
            'slice_height_cm': 25,
            'base_zone_height_m': 0.0,
            'eps_m': 0.18,
            'min_samples': 8,
            'merge_dist_m': 0.25,
            'min_trunk_points': 120,
            'min_vertical_span_m': 1.0,
            'full_height_assign_dist_m': 0.45
        }
        # Detection state/versioning to prevent redundant recalculation
        self._trunk_detection_version = 0
        self._branch_detection_version = 0
        # Signature of last volume calculation: (tree_id, trunk_version, branch_version)
        self._last_volume_calc_signature = None
        # DBH reference height (customizable by user, default = 1.3m)
        self.dbh_reference_height = 1.3
        # Universal tree-point filter: which scalar field marks "tree points" and with
        # which value. Auto-detected per data output on LAS load:
        #   TreeAIBox -> 'treefilter' (tree = 2), TreeLearn -> 'stemcls' (stem = 2),
        #   ForestFormer -> None (no point classification available).
        self.tree_filter_field = None
        self.tree_filter_value = 2
        # Last used detection parameters for change detection
        self._last_trunk_detection_params = None
        self._last_branch_detection_params = None
        # Monotonic counter to keep volume overlay actor names unique per successful calculation run
        self._volume_calc_run_id = 0
        
        # Branch detection variables
        self.branch_assignment = None  # Array: -1 = noise, >=0 = branch_id
        self.branch_meta = {}  # Dict: global_branch_id -> {trunk_id, local_id, point_count, z_span}
        print(f"[INIT] {time.strftime('%H:%M:%S')} - Instance variables set ({time.perf_counter() - init_start:.3f}s)")
        

        
        ui_start = time.perf_counter()
        print(f"[INIT] {time.strftime('%H:%M:%S')} - Starting init_ui...")
        self.init_ui()
        print(f"[INIT] {time.strftime('%H:%M:%S')} - init_ui done ({time.perf_counter() - ui_start:.3f}s)")
        
        style_start = time.perf_counter()
        self.apply_stylesheet()
        print(f"[INIT] {time.strftime('%H:%M:%S')} - apply_stylesheet done ({time.perf_counter() - style_start:.3f}s)")
        print(f"[INIT] {time.strftime('%H:%M:%S')} - __init__ complete ({time.perf_counter() - init_start:.3f}s total)")


    def get_tree_point_mask(self):
        """Boolean mask over all points of the active LAS marking tree points.

        Returns None when the filter is disabled, no field is selected, or the
        selected field is missing from the active LAS (e.g. ForestFormer output).
        """
        if self.las_data is None:
            return None
        if not getattr(self, 'filter_checkbox', None) or not self.filter_checkbox.isChecked():
            return None
        field = getattr(self, 'tree_filter_field', None)
        if not field:
            return None
        if field not in list(self.las_data.point_format.dimension_names):
            return None
        val = int(getattr(self, 'tree_filter_value', 2))
        vals = np.asarray(self.las_data[field])
        if field == 'semantic_pred' and val == 2:
            # For semantic_pred: 0=ground, 1=trunk, 2=foliage.
            # Default tree filter value 2 includes all tree points (trunk=1 + foliage=2).
            return (vals == 1) | (vals == 2)
        return vals == val


    def detect_tree_point_filter(self):
        """Auto-select the tree-point filter field from the active LAS dimensions.

        Priority: 'treefilter' (TreeAIBox, tree = 2) > 'semantic_pred'
        (semantic segmentation: 0 = ground, 1 = trunk, 2 = foliage) > 'stemcls' (TreeLearn,
        stem = 2, stem points only). Outputs with none of these -> filter off.
        """
        field = None
        value = 2
        if self.las_data is not None:
            dims = list(self.las_data.point_format.dimension_names)
            field, value = detect_tree_filter_field(dims)
        self.tree_filter_field = field
        self.tree_filter_value = value
        if getattr(self, 'tree_filter_field_combo', None) is not None:
            # Show the field actually used for filtering (or 'No filter')
            self.tree_filter_field_combo.blockSignals(True)
            idx = self.tree_filter_field_combo.findData(field) if field else -1
            self.tree_filter_field_combo.setCurrentIndex(
                idx if idx >= 0 else self.tree_filter_field_combo.findData("__NONE__"))
            self.tree_filter_field_combo.blockSignals(False)
        if getattr(self, 'tree_filter_value_spin', None) is not None:
            self.tree_filter_value_spin.blockSignals(True)
            self.tree_filter_value_spin.setValue(value)
            self.tree_filter_value_spin.blockSignals(False)
        if field:
            self.log_to_console(f"[Filter] Tree-point filter auto-detected: {field} == {self.tree_filter_value}")
        else:
            self.log_to_console("[Filter] No tree-point classification field found - filter disabled")


    def _on_tree_filter_field_changed(self, index):
        """Handle tree-point filter field selection (treefilter/stemcls/semantic_pred/None)."""
        data = self.tree_filter_field_combo.itemData(index)
        if data == "__NONE__":
            self.tree_filter_field = None
        else:
            self.tree_filter_field = data
            # Default tree value is 2 (semantic_pred: 0=ground, 1=trunk, 2=foliage)
            self.tree_filter_value = 2
            if getattr(self, 'tree_filter_value_spin', None) is not None:
                self.tree_filter_value_spin.blockSignals(True)
                self.tree_filter_value_spin.setValue(2)
                self.tree_filter_value_spin.blockSignals(False)
        self.on_filter_changed()


    def _on_tree_filter_value_changed(self, value):
        """Handle tree-point filter value change."""
        self.tree_filter_value = int(value)
        self.on_filter_changed()


    def set_camera_view(self, view_type):
        """Set the camera to a standard orthographic view."""
        if not self.plotter:
            return
            
        # Get the bounds of the current scene to determine appropriate camera distance and focal point
        bounds = self.plotter.bounds
        if bounds is not None:
            # Calculate the center of the bounding box
            center_x = (bounds[0] + bounds[1]) / 2
            center_y = (bounds[2] + bounds[3]) / 2
            center_z = (bounds[4] + bounds[5]) / 2
            focal_point = (center_x, center_y, center_z)
            
            # Calculate the diagonal of the bounding box to determine camera distance
            x_range = bounds[1] - bounds[0]
            y_range = bounds[3] - bounds[2] 
            z_range = bounds[5] - bounds[4]
            diagonal = np.sqrt(x_range**2 + y_range**2 + z_range**2)
            camera_distance = diagonal * 2.0  # Place camera at 2x the diagonal distance
        else:
            focal_point = (0, 0, 0)
            camera_distance = 100.0  # Default fallback distance
        
        # Reset camera to default position first
        self.plotter.reset_camera()
        
        if view_type == 'view_front':
            # Front view: looking along positive Y axis
            self.plotter.camera.position = (focal_point[0], focal_point[1] - camera_distance, focal_point[2])
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = (0, 0, 1)
        elif view_type == 'view_back':
            # Back view: looking along negative Y axis
            self.plotter.camera.position = (focal_point[0], focal_point[1] + camera_distance, focal_point[2])
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = (0, 0, 1)
        elif view_type == 'view_top':
            # Top view: looking down negative Z axis
            self.plotter.camera.position = (focal_point[0], focal_point[1], focal_point[2] + camera_distance)
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = (0, 1, 0)
        elif view_type == 'view_bottom':
            # Bottom view: looking up positive Z axis
            self.plotter.camera.position = (focal_point[0], focal_point[1], focal_point[2] - camera_distance)
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = (0, -1, 0)
        elif view_type == 'view_left':
            # Left view: looking along positive X axis
            self.plotter.camera.position = (focal_point[0] - camera_distance, focal_point[1], focal_point[2])
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = (0, 0, 1)
        elif view_type == 'view_right':
            # Right view: looking along negative X axis
            self.plotter.camera.position = (focal_point[0] + camera_distance, focal_point[1], focal_point[2])
            self.plotter.camera.focal_point = focal_point
            self.plotter.camera.up = (0, 0, 1)
        
        # Reset camera to ensure proper zoom for the new view
        self.plotter.reset_camera()
        self.plotter.update()


    def log_to_console(self, message):
        """Add a message to the console output."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.console.append(formatted_msg)
        # Also print to terminal for visibility
        try:
            print(formatted_msg)
        except Exception:
            try:
                print(formatted_msg.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass
        # Auto-scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


    def log_many_to_console(self, messages):
        """Append many messages to the console in ONE widget update.

        Much faster than calling log_to_console() per message for bulk output
        (avoids one Qt repaint + auto-scroll per line).
        """
        if not messages:
            return
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        block = "\n".join(f"[{timestamp}] {m}" for m in messages)
        self.console.append(block)
        try:
            print(block)
        except Exception:
            try:
                print(block.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


    def _set_progress(self, value, text=None):
        """Update the determinate progress bar and keep the UI responsive."""
        try:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(value))
            if text:
                self.progress_bar.setFormat(f"{text}  %p%")
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
        except Exception:
            pass

    def _set_indeterminate_progress(self, text=None):
        """Show indeterminate progress bar and keep UI responsive."""
        try:
            self.progress_bar.setRange(0, 0)
            if text:
                self.progress_bar.setFormat(text)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
        except Exception:
            pass

    def _hide_progress(self):
        """Hide progress bar and restore normal state."""
        try:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            QApplication.processEvents()
        except Exception:
            pass


    def _start_volume_load_profile(self):
        """Start a timed Load Data profile log on disk."""
        import time
        from datetime import datetime

        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis_log')
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(log_dir, f'volume_load_profile_{stamp}.log')
        self._volume_load_profile_path = path
        self._volume_load_profile_t0 = time.perf_counter()
        self._volume_load_profile_last = self._volume_load_profile_t0
        header = (
            f"Volume Load Data profile started {datetime.now().isoformat(timespec='seconds')}\n"
            f"Log file: {path}\n"
        )
        try:
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(header)
                handle.flush()
        except Exception as exc:
            self._volume_load_profile_path = None
            self.log_to_console(f"[WARN] Could not create load-data profile log: {exc}")
            return None
        self.log_to_console(f"[Log] Load Data profile log: {path}")
        return path


    def _profile_volume_load(self, step_name, extra=''):
        """Record elapsed time for one Load Data step to the profile log and console."""
        import time
        from datetime import datetime

        path = getattr(self, '_volume_load_profile_path', None)
        if not path:
            return
        now = time.perf_counter()
        dt = now - getattr(self, '_volume_load_profile_last', now)
        total = now - getattr(self, '_volume_load_profile_t0', now)
        self._volume_load_profile_last = now
        extra_text = f"  | {extra}" if extra else ""
        line = (
            f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}  "
            f"+{dt:8.3f}s  total={total:8.3f}s  {step_name}{extra_text}\n"
        )
        try:
            with open(path, 'a', encoding='utf-8') as handle:
                handle.write(line)
                handle.flush()
        except Exception:
            pass
        self.log_to_console(f"[Time] {step_name}: {dt:.3f}s (total {total:.3f}s){extra_text}")


    def _finish_volume_load_profile(self, status='done'):
        """Close the Load Data profile log with a final status line."""
        path = getattr(self, '_volume_load_profile_path', None)
        if not path:
            return
        self._profile_volume_load(f'LOAD_DATA_{status.upper()}')
        self._volume_load_profile_path = None


    def _show_text_dialog(self, title, text):
        """Show a simple read-only text dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(420, 480)
        layout = QVBoxLayout(dlg)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        layout.addWidget(text_edit)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()


