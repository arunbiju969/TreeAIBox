"""Side-by-side LAS comparison dialog (dual viewports, picking, isolation)."""

import os
import sys
import numpy as np
import laspy
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QSlider, QFrame,
    QSplitter, QProgressBar, QMessageBox, QFileDialog,
    QButtonGroup, QRadioButton, QSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class LASComparisonDialog(QDialog):
    """Side-by-side dual viewport 3D comparison window for two LAS/LAZ point clouds."""

    def __init__(self, parent=None, initial_file_a=None, initial_file_b=None, loaded_layers=None):
        super().__init__(parent)
        self.setWindowTitle("LAS Side-by-Side 3D Comparison - Forestry AI Suite")
        self.resize(1400, 900)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinMaxButtonsHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self.loaded_layers = loaded_layers or []
        self.file_a_path = initial_file_a
        self.file_b_path = initial_file_b

        self.data_a = None
        self.data_b = None
        self._syncing = False

        self.init_ui()
        self.setWindowState(Qt.WindowState.WindowMaximized)

        if self.file_a_path and os.path.isfile(self.file_a_path):
            self.load_file_a(self.file_a_path)
        if self.file_b_path and os.path.isfile(self.file_b_path):
            self.load_file_b(self.file_b_path)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── Top Toolbar ──
        tb_frame = QFrame()
        tb_frame.setStyleSheet("background-color: #2b2b2b; border-radius: 0px; padding: 4px;")
        tb_layout = QHBoxLayout(tb_frame)
        tb_layout.setContentsMargins(6, 4, 6, 4)
        tb_layout.setSpacing(10)

        self.sync_cameras_check = QCheckBox("Link / Sync Cameras")
        self.sync_cameras_check.setChecked(True)
        self.sync_cameras_check.setToolTip("When checked, orbiting, panning, or zooming in either viewport synchronizes the other view.")
        self.sync_cameras_check.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
        tb_layout.addWidget(self.sync_cameras_check)

        self.align_origin_check = QCheckBox("Align Centers to (0,0)")
        self.align_origin_check.setChecked(True)
        self.align_origin_check.setToolTip("Center both point clouds to (0,0,0) locally so synced camera rotation orbits around each cloud's center.")
        self.align_origin_check.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self.align_origin_check.toggled.connect(self._on_align_toggled)
        tb_layout.addWidget(self.align_origin_check)

        tb_layout.addSpacing(6)

        self.unique_tree_colors_check = QCheckBox("Unique Tree Colors")
        self.unique_tree_colors_check.setChecked(True)
        self.unique_tree_colors_check.setToolTip("Generate distinct non-repeating colors for every tree across both views so identical trees share matching colors.")
        self.unique_tree_colors_check.setStyleSheet("color: #00E676; font-weight: bold; font-size: 11px;")
        self.unique_tree_colors_check.toggled.connect(self._on_unique_colors_toggled)
        tb_layout.addWidget(self.unique_tree_colors_check)

        tb_layout.addSpacing(10)

        lbl_views = QLabel("Views:")
        lbl_views.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        tb_layout.addWidget(lbl_views)

        btn_top = QPushButton("Top (XY)")
        btn_top.setFixedHeight(26)
        btn_top.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        btn_top.clicked.connect(lambda: self._set_preset_view('xy'))
        tb_layout.addWidget(btn_top)

        btn_front = QPushButton("Front (XZ)")
        btn_front.setFixedHeight(26)
        btn_front.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        btn_front.clicked.connect(lambda: self._set_preset_view('xz'))
        tb_layout.addWidget(btn_front)

        btn_side = QPushButton("Side (YZ)")
        btn_side.setFixedHeight(26)
        btn_side.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        btn_side.clicked.connect(lambda: self._set_preset_view('yz'))
        tb_layout.addWidget(btn_side)

        btn_iso = QPushButton("Isometric")
        btn_iso.setFixedHeight(26)
        btn_iso.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        btn_iso.clicked.connect(lambda: self._set_preset_view('iso'))
        tb_layout.addWidget(btn_iso)

        btn_reset = QPushButton("Reset Both")
        btn_reset.setFixedHeight(26)
        btn_reset.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        btn_reset.clicked.connect(self.reset_both_cameras)
        tb_layout.addWidget(btn_reset)

        tb_layout.addSpacing(10)

        # Isolate Picked Tree mode toggle
        self.isolate_mode_btn = QPushButton("Isolate Picked Tree: OFF")
        self.isolate_mode_btn.setCheckable(True)
        self.isolate_mode_btn.setFixedHeight(26)
        self.isolate_mode_btn.setToolTip("When ON, clicking any point on a tree in either view immediately isolates that tree on BOTH views for synchronized inspection.")
        self.isolate_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 0px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 10px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:checked {
                background-color: #00E676;
                color: #000000;
                border: 1px solid #00C853;
            }
        """)
        self.isolate_mode_btn.toggled.connect(self._on_isolate_mode_toggled)
        tb_layout.addWidget(self.isolate_mode_btn)

        self.btn_show_all_trees = QPushButton("Show All Trees")
        self.btn_show_all_trees.setFixedHeight(26)
        self.btn_show_all_trees.setToolTip("Reset tree filters across both views and show the complete forest")
        self.btn_show_all_trees.setStyleSheet("""
            QPushButton {
                background-color: #263238;
                color: #80CBC4;
                border: 1px solid #37474F;
                border-radius: 0px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 10px;
            }
            QPushButton:hover {
                background-color: #37474F;
                color: #ffffff;
            }
        """)
        self.btn_show_all_trees.clicked.connect(self.show_all_trees)
        tb_layout.addWidget(self.btn_show_all_trees)

        tb_layout.addStretch()

        btn_swap = QPushButton("↔ Swap A/B")
        btn_swap.setFixedHeight(26)
        btn_swap.setStyleSheet("font-size: 11px; padding: 2px 8px; background-color: #37474F; color: white;")
        btn_swap.setToolTip("Swap Left (File A) and Right (File B) views")
        btn_swap.clicked.connect(self.swap_views)
        tb_layout.addWidget(btn_swap)

        btn_diff = QPushButton("[Metrics] Metrics Comparison")
        btn_diff.setFixedHeight(26)
        btn_diff.setStyleSheet("background-color: #0288D1; color: white; font-weight: bold; font-size: 11px; padding: 2px 10px; border-radius: 0px;")
        btn_diff.setToolTip("View detailed tabular difference between File A and File B")
        btn_diff.clicked.connect(self.show_metrics_diff)
        tb_layout.addWidget(btn_diff)

        main_layout.addWidget(tb_frame)

        # ── Splitter with Left and Right Viewports ──
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = self._create_viewport_panel('A')
        right_widget = self._create_viewport_panel('B')

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([680, 680])

        main_layout.addWidget(self.splitter, 1)

        # ── Bottom Summary Banner ──
        self.summary_label = QLabel("Select two LAS files above to compare.")
        self.summary_label.setStyleSheet("color: #4CAF50; font-size: 11px; background-color: #1e1e1e; padding: 5px 8px; border-radius: 0px; border: 1px solid #333333;")
        main_layout.addWidget(self.summary_label)

        # ── Picked Points Comparison Banner ──
        self.picked_compare_label = QLabel("")
        self.picked_compare_label.setStyleSheet("color: #00E5FF; font-size: 11px; background-color: #212121; padding: 5px 8px; border-radius: 0px; border: 1px solid #00838F;")
        self.picked_compare_label.setTextFormat(Qt.TextFormat.RichText)
        self.picked_compare_label.setVisible(False)
        main_layout.addWidget(self.picked_compare_label)

        # Setup camera sync observers
        self.p1.interactor.AddObserver('InteractionEvent', self._on_p1_interact)
        self.p2.interactor.AddObserver('InteractionEvent', self._on_p2_interact)

    def _create_viewport_panel(self, side):
        panel = QFrame()
        panel.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333333; border-radius: 0px;")
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # Control Row
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(6)

        lbl = QLabel(f"<b>View {side}:</b>")
        lbl.setStyleSheet("color: #4CAF50; font-size: 11px;" if side == 'A' else "color: #00BCD4; font-size: 11px;")
        ctrl_layout.addWidget(lbl)

        combo_file = QComboBox()
        combo_file.setMinimumWidth(180)
        combo_file.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 11px;")
        ctrl_layout.addWidget(combo_file, 1)

        lbl_tree = QLabel("Tree:")
        lbl_tree.setStyleSheet("color: #cccccc; font-size: 11px;")
        ctrl_layout.addWidget(lbl_tree)

        combo_tree = QComboBox()
        combo_tree.addItem("All Trees")
        combo_tree.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 11px;")
        ctrl_layout.addWidget(combo_tree)

        lbl_color = QLabel("Color:")
        lbl_color.setStyleSheet("color: #cccccc; font-size: 11px;")
        ctrl_layout.addWidget(lbl_color)

        combo_color = QComboBox()
        combo_color.addItems(["ITC / Tree ID", "RGB (Natural)", "Elevation (Z)", "Classification", "Intensity", "Default Green"])
        combo_color.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 11px;")
        ctrl_layout.addWidget(combo_color)

        lbl_pt = QLabel("Size:")
        lbl_pt.setStyleSheet("color: #cccccc; font-size: 11px;")
        ctrl_layout.addWidget(lbl_pt)

        spin_pt = QSpinBox()
        spin_pt.setRange(1, 10)
        spin_pt.setValue(2)
        spin_pt.setFixedWidth(45)
        spin_pt.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 11px;")
        ctrl_layout.addWidget(spin_pt)

        vbox.addLayout(ctrl_layout)

        # Plotter
        plotter = QtInteractor()
        plotter.set_background('#2e2e2e')
        vbox.addWidget(plotter.interactor, 1)

        # Metadata info bar
        info_label = QLabel("No file loaded")
        info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 2px;")
        vbox.addWidget(info_label)

        # Selected point inspection card with clear button
        pick_box = QHBoxLayout()
        pick_box.setSpacing(4)

        pick_label = QLabel(" Click any point in the 3D view to inspect its ITC, coordinates, and parameters")
        pick_label.setStyleSheet("""
            background-color: #242424;
            color: #dcdcdc;
            border: 1px solid #3c3c3c;
            border-radius: 0px;
            font-size: 11px;
            padding: 4px 8px;
        """)
        pick_label.setTextFormat(Qt.TextFormat.RichText)
        pick_label.setWordWrap(True)
        pick_box.addWidget(pick_label, 1)

        btn_isolate_tree = QPushButton("Isolate Tree")
        btn_isolate_tree.setFixedHeight(26)
        btn_isolate_tree.setToolTip("Isolate this picked tree across both View A and View B")
        btn_isolate_tree.setStyleSheet("""
            QPushButton {
                background-color: #1b5e20;
                color: #e8f5e9;
                border: 1px solid #2e7d32;
                border-radius: 0px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 8px;
            }
            QPushButton:hover {
                background-color: #2e7d32;
                color: #ffffff;
            }
        """)
        btn_isolate_tree.clicked.connect(lambda checked, s=side: self._isolate_picked_tree_button_clicked(s))
        pick_box.addWidget(btn_isolate_tree)

        btn_clear_pick = QPushButton("Clear")
        btn_clear_pick.setFixedSize(54, 26)
        btn_clear_pick.setToolTip("Clear picked point highlight and selection")
        btn_clear_pick.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #bbbbbb;
                border: 1px solid #444444;
                border-radius: 0px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
                color: #ffffff;
            }
        """)
        btn_clear_pick.clicked.connect(lambda checked, s=side: self._clear_pick(s))
        pick_box.addWidget(btn_clear_pick)

        vbox.addLayout(pick_box)

        if side == 'A':
            self.p1 = plotter
            self.file_a_combo = combo_file
            self.tree_a_combo = combo_tree
            self.color_a_combo = combo_color
            self.pt_a_spin = spin_pt
            self.info_a_label = info_label
            self.pick_a_label = pick_label

            self.file_a_combo.currentIndexChanged.connect(self._on_file_a_selected)
            self.tree_a_combo.currentIndexChanged.connect(lambda: self._update_view('A'))
            self.color_a_combo.currentIndexChanged.connect(lambda: self._update_view('A'))
            self.pt_a_spin.valueChanged.connect(lambda: self._update_view('A'))

            self.p1.enable_point_picking(callback=lambda pt: self._on_point_picked('A', pt), show_message=False)
        else:
            self.p2 = plotter
            self.file_b_combo = combo_file
            self.tree_b_combo = combo_tree
            self.color_b_combo = combo_color
            self.pt_b_spin = spin_pt
            self.info_b_label = info_label
            self.pick_b_label = pick_label

            self.file_b_combo.currentIndexChanged.connect(self._on_file_b_selected)
            self.tree_b_combo.currentIndexChanged.connect(lambda: self._update_view('B'))
            self.color_b_combo.currentIndexChanged.connect(lambda: self._update_view('B'))
            self.pt_b_spin.valueChanged.connect(lambda: self._update_view('B'))

            self.p2.enable_point_picking(callback=lambda pt: self._on_point_picked('B', pt), show_message=False)

        self._populate_file_combo(combo_file, self.file_a_path if side == 'A' else self.file_b_path)
        return panel

    def _format_display_label(self, path):
        fname = os.path.basename(path)
        p_norm = os.path.normpath(path).lower()
        if 'treelearn' in p_norm:
            tag = "TreeLearn"
        elif 'forestformer' in p_norm or 'forest_former' in p_norm or 'forest-former' in p_norm:
            tag = "ForestFormer"
        elif 'segmented_results' in p_norm or 'treeaibox' in p_norm:
            tag = "TreeAIBox"
        else:
            tag = os.path.basename(os.path.dirname(path))
        return f"[LAS] {fname} [{tag}]" if tag else f"[LAS] {fname}"

    def _populate_file_combo(self, combo, current_path):
        combo.blockSignals(True)
        combo.clear()
        paths_seen = set()

        if current_path and os.path.isfile(current_path):
            combo.addItem(self._format_display_label(current_path), current_path)
            paths_seen.add(os.path.normpath(current_path))

        for layer in self.loaded_layers:
            lp = layer.get('path')
            if lp and os.path.isfile(lp) and os.path.normpath(lp) not in paths_seen:
                combo.addItem(self._format_display_label(lp), lp)
                paths_seen.add(os.path.normpath(lp))

        combo.addItem("... Browse Other LAS File...", "__BROWSE__")
        combo.blockSignals(False)

    def _find_tree_dim(self, dim_names):
        dim_list = list(dim_names)
        candidates = [
            'itc', 'treeID', 'TreeID', 'tree_id', 'treeId', 'TreeId',
            'InstanceID', 'instance_id', 'InstanceId', 'instanceID',
            'ClusterID', 'cluster_id', 'ClusterId', 'cluster',
            'treeoff', 'pred_itc', 'pred_treeitc', 'itc_id'
        ]
        for c in candidates:
            if c in dim_list:
                return c
        lower_map = {d.lower(): d for d in dim_list}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    def _read_las_cache(self, path):
        las = laspy.read(path)
        tree_field = self._find_tree_dim(las.point_format.dimension_names)
        coords = np.column_stack([np.array(las.x), np.array(las.y), np.array(las.z)])
        centroid = np.mean(coords, axis=0)
        ground_z = np.min(coords[:, 2])
        center_anchor = np.array([centroid[0], centroid[1], ground_z])

        trees = np.array([])
        if tree_field:
            tvals = np.array(las[tree_field])
            trees = np.unique(tvals[tvals > 0]).astype(int)

        p_norm = os.path.normpath(path).lower()
        if 'treelearn' in p_norm:
            tag = "TreeLearn"
        elif 'forestformer' in p_norm or 'forest_former' in p_norm or 'forest-former' in p_norm:
            tag = "ForestFormer"
        elif 'segmented_results' in p_norm or 'treeaibox' in p_norm:
            tag = "TreeAIBox"
        else:
            tag = os.path.basename(os.path.dirname(path))

        return {
            'path': path,
            'name': f"{os.path.basename(path)} [{tag}]",
            'las': las,
            'coords': coords,
            'center_anchor': center_anchor,
            'tree_field': tree_field,
            'trees': trees,
            'num_points': len(coords),
            'dx': coords[:, 0].max() - coords[:, 0].min(),
            'dy': coords[:, 1].max() - coords[:, 1].min(),
            'dz': coords[:, 2].max() - coords[:, 2].min(),
            'kdtree': None,
        }

    def _get_kdtree(self, data):
        """Lazily build and cache KDTree for fast spatial nearest-neighbor lookups."""
        if data is None or 'coords' not in data or len(data['coords']) == 0:
            return None
        if data.get('kdtree') is None:
            from scipy.spatial import cKDTree
            data['kdtree'] = cKDTree(data['coords'])
        return data['kdtree']

    def _add_picked_marker(self, side, x, y, z):
        """Add screen-space spherical highlight marker at real-world coordinates (x, y, z)."""
        data = self.data_a if side == 'A' else self.data_b
        plotter = self.p1 if side == 'A' else self.p2
        pt_spin = self.pt_a_spin if side == 'A' else self.pt_b_spin
        if data is None or plotter is None:
            return
        try:
            align_origin = self.align_origin_check.isChecked()
            center_anchor = data['center_anchor']
            render_pt = np.array([[x, y, z]]) - center_anchor if align_origin else np.array([[x, y, z]])
            marker_cloud = pv.PolyData(render_pt)
            marker_color = '#FF1744' if side == 'A' else '#00E5FF'
            m_size = max(pt_spin.value() + 6, 9)
            plotter.add_points(
                marker_cloud,
                color=marker_color,
                point_size=m_size,
                render_points_as_spheres=True,
                name='picked_point_marker',
                reset_camera=False
            )
            plotter.render()
        except Exception:
            pass

    def load_file_a(self, path):
        try:
            self.data_a = self._read_las_cache(path)
            self.file_a_path = path
            self._update_tree_combo(self.tree_a_combo, self.data_a['trees'])
            self._update_color_combo(self.color_a_combo, self.data_a)
            self._update_view('A', reset_cam=True)
            self._update_summary()
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load View A file:\n{e}")

    def load_file_b(self, path):
        try:
            self.data_b = self._read_las_cache(path)
            self.file_b_path = path
            self._update_tree_combo(self.tree_b_combo, self.data_b['trees'])
            self._update_color_combo(self.color_b_combo, self.data_b)
            self._update_view('B', reset_cam=True)
            self._update_summary()
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load View B file:\n{e}")

    def _update_color_combo(self, combo, data):
        """Populate the color combobox with all attributes and scalar dimensions from the LAS file."""
        if not data or not data.get('las'):
            return
        las = data['las']
        dim_names = list(las.point_format.dimension_names)

        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()

        options = []
        if data.get('tree_field'):
            options.append("ITC / Tree ID")
        options.append("Elevation (Z)")
        if 'red' in dim_names and 'green' in dim_names and 'blue' in dim_names:
            options.append("RGB (Natural)")
        options.append("Default Green")

        skip_dims = {'x', 'y', 'z', 'red', 'green', 'blue'}
        extra_fields = [d for d in dim_names if d.lower() not in skip_dims]
        for f in sorted(extra_fields, key=lambda s: s.lower()):
            if f not in options:
                options.append(f)

        combo.addItems(options)

        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _update_tree_combo(self, combo, trees):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All Trees (Full Forest)")
        for t in sorted(trees):
            combo.addItem(f"Tree {t}", t)
        combo.blockSignals(False)

    def _on_file_a_selected(self, index):
        val = self.file_a_combo.itemData(index)
        if val == "__BROWSE__":
            p, _ = QFileDialog.getOpenFileName(self, "Select First LAS/LAZ File (View A)", "", "LAS files (*.las *.laz)")
            if p:
                self._populate_file_combo(self.file_a_combo, p)
                self.load_file_a(p)
        elif val and os.path.isfile(val):
            self.load_file_a(val)

    def _on_file_b_selected(self, index):
        val = self.file_b_combo.itemData(index)
        if val == "__BROWSE__":
            p, _ = QFileDialog.getOpenFileName(self, "Select Second LAS/LAZ File (View B)", "", "LAS files (*.las *.laz)")
            if p:
                self._populate_file_combo(self.file_b_combo, p)
                self.load_file_b(p)
        elif val and os.path.isfile(val):
            self.load_file_b(val)

    def _on_align_toggled(self):
        self._update_view('A', reset_cam=True)
        self._update_view('B', reset_cam=True)

    def _on_unique_colors_toggled(self, checked=False):
        self._update_view('A', reset_cam=False)
        self._update_view('B', reset_cam=False)

    def _get_unique_tree_colors(self, tvals):
        """Map tree ID values to distinct, non-repeating RGB colors using golden-ratio hue distribution."""
        max_id = int(np.max(tvals)) if len(tvals) > 0 else 0
        if max_id < 0:
            max_id = 0

        # Build or expand cached color lookup table if needed
        if not hasattr(self, '_unique_color_lut') or len(self._unique_color_lut) <= max_id:
            import colorsys
            lut_size = max(max_id + 500, 2048)
            lut = np.zeros((lut_size, 3), dtype=np.uint8)
            # Tree ID <= 0: unassigned / ground / noise -> neutral dark gray
            lut[0] = [80, 80, 80]
            for i in range(1, lut_size):
                # Golden-ratio hue dispersal ensures maximum perceptual difference between consecutive IDs
                h = (i * 0.618033988749895) % 1.0
                s = 0.75 + (i % 3) * 0.08
                v = 0.85 + (i % 2) * 0.12
                r, g, b = colorsys.hsv_to_rgb(h, min(s, 1.0), min(v, 1.0))
                lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
            self._unique_color_lut = lut

        safe_indices = np.clip(tvals, 0, len(self._unique_color_lut) - 1).astype(np.int64)
        return self._unique_color_lut[safe_indices]

    def _update_view(self, side, reset_cam=False):
        data = self.data_a if side == 'A' else self.data_b
        plotter = self.p1 if side == 'A' else self.p2
        tree_combo = self.tree_a_combo if side == 'A' else self.tree_b_combo
        color_combo = self.color_a_combo if side == 'A' else self.color_b_combo
        pt_spin = self.pt_a_spin if side == 'A' else self.pt_b_spin
        info_lbl = self.info_a_label if side == 'A' else self.info_b_label

        if data is None:
            return

        las = data['las']
        coords = data['coords']
        center_anchor = data['center_anchor']
        align_origin = self.align_origin_check.isChecked()

        # Filter by tree if selected
        tree_idx = tree_combo.currentIndex()
        tree_id = tree_combo.itemData(tree_idx)

        mask = None
        if tree_id is not None and data['tree_field']:
            tvals = np.array(las[data['tree_field']])
            mask = (tvals == tree_id)
            filtered_coords = coords[mask]
            data['rendered_indices'] = np.where(mask)[0]
        else:
            filtered_coords = coords
            data['rendered_indices'] = np.arange(len(coords))

        if len(filtered_coords) == 0:
            return

        # Coordinates offset
        if align_origin:
            pts_to_render = filtered_coords - center_anchor
        else:
            pts_to_render = filtered_coords

        data['rendered_pts'] = pts_to_render

        cloud = pv.PolyData(pts_to_render)
        pt_size = pt_spin.value()
        color_mode = color_combo.currentText()

        # Coloring logic - clear old actors and markers
        plotter.clear()
        if color_mode == "ITC / Tree ID" and data['tree_field']:
            tvals = np.array(las[data['tree_field']])
            if mask is not None:
                tvals = tvals[mask]
            if getattr(self, 'unique_tree_colors_check', None) and self.unique_tree_colors_check.isChecked():
                rgb_colors = self._get_unique_tree_colors(tvals)
                cloud['rgb'] = rgb_colors
                plotter.add_points(cloud, scalars='rgb', rgb=True, point_size=pt_size, name='cloud', reset_camera=reset_cam)
            else:
                c_indices = (tvals - 1) % 10
                c_indices[tvals <= 0] = -1
                plotter.add_points(cloud, scalars=c_indices, cmap='tab10', clim=[-1, 9], point_size=pt_size, name='cloud', reset_camera=reset_cam)
        elif color_mode == "RGB (Natural)" and 'red' in las.point_format.dimension_names:
            r = np.array(las.red, dtype=np.float32)
            g = np.array(las.green, dtype=np.float32)
            b = np.array(las.blue, dtype=np.float32)
            if mask is not None:
                r, g, b = r[mask], g[mask], b[mask]
            m = max(r.max(), g.max(), b.max(), 1.0)
            if m > 255:
                r /= 65535.0; g /= 65535.0; b /= 65535.0
            elif m > 1.0:
                r /= 255.0; g /= 255.0; b /= 255.0
            cloud['rgb'] = np.column_stack([r, g, b])
            plotter.add_points(cloud, scalars='rgb', rgb=True, point_size=pt_size, name='cloud', reset_camera=reset_cam)
        elif color_mode == "Elevation (Z)":
            plotter.add_points(cloud, scalars=pts_to_render[:, 2], cmap='viridis', point_size=pt_size, name='cloud', reset_camera=reset_cam)
        elif color_mode in las.point_format.dimension_names:
            scalars = np.array(las[color_mode], dtype=np.float32)
            if mask is not None:
                scalars = scalars[mask]
            if color_mode.lower() == 'semantic_pred':
                # Semantic segmentation: 0=Ground (Red), 1=Trunk (Blue), 2=Foliage (Green)
                sem_vals = np.asarray(scalars, dtype=np.int32)
                rgb = np.full((len(sem_vals), 3), 158, dtype=np.uint8)
                rgb[sem_vals == 0] = [229, 57, 53]   # Red: Ground
                rgb[sem_vals == 1] = [30, 136, 229]  # Blue: Trunk
                rgb[sem_vals == 2] = [67, 160, 71]   # Green: Foliage
                cloud['rgb'] = rgb
                plotter.add_points(cloud, scalars='rgb', rgb=True, point_size=pt_size, name='cloud', reset_camera=reset_cam)
            else:
                categorical_dims = {'classification', 'return_number', 'number_of_returns', 'stemcls', 'treefilter', 'scanner_channel', 'point_source_id', 'user_data'}
                if color_mode.lower() in categorical_dims:
                    plotter.add_points(cloud, scalars=scalars, cmap='tab10', point_size=pt_size, name='cloud', reset_camera=reset_cam)
                elif color_mode.lower() == 'intensity':
                    plotter.add_points(cloud, scalars=scalars, cmap='gray', point_size=pt_size, name='cloud', reset_camera=reset_cam)
                else:
                    plotter.add_points(cloud, scalars=scalars, cmap='cividis', point_size=pt_size, name='cloud', reset_camera=reset_cam)
        else:
            def_color = '#4CAF50' if side == 'A' else '#00BCD4'
            plotter.add_points(cloud, color=def_color, point_size=pt_size, name='cloud', reset_camera=reset_cam)

        dx = filtered_coords[:, 0].max() - filtered_coords[:, 0].min()
        dy = filtered_coords[:, 1].max() - filtered_coords[:, 1].min()
        dz = filtered_coords[:, 2].max() - filtered_coords[:, 2].min()
        n_pts = len(filtered_coords)
        n_trees = len(data['trees'])
        tree_txt = f"Tree {tree_id}" if tree_id is not None else f"{n_trees} Trees"
        info_lbl.setText(f"{data['name']} | {n_pts:,} pts | {tree_txt} | Size: {dx:.1f}m × {dy:.1f}m × {dz:.1f}m")

        plotter.render()

    def _on_point_picked(self, side, picked_info):
        """Handle point picking in viewport: resolves point index, finds spatially corresponding point on other view, displays ITC and active parameters."""
        data = self.data_a if side == 'A' else self.data_b
        other_data = self.data_b if side == 'A' else self.data_a
        if data is None:
            return
        plotter = self.p1 if side == 'A' else self.p2
        color_combo = self.color_a_combo if side == 'A' else self.color_b_combo
        pick_label = self.pick_a_label if side == 'A' else self.pick_b_label
        other_pick_label = self.pick_b_label if side == 'A' else self.pick_a_label

        las = data['las']
        rendered_pts = data.get('rendered_pts')
        rendered_indices = data.get('rendered_indices')
        if rendered_pts is None or len(rendered_pts) == 0:
            return

        # Extract coordinates from pick payload
        picked_xyz = None
        if hasattr(picked_info, 'picked_point') and picked_info.picked_point is not None:
            picked_xyz = np.asarray(picked_info.picked_point, dtype=float)
        elif hasattr(picked_info, 'points') and picked_info.points is not None and len(picked_info.points) > 0:
            picked_xyz = np.asarray(picked_info.points[0], dtype=float)
        elif isinstance(picked_info, (list, tuple, np.ndarray)) and len(picked_info) >= 3:
            picked_xyz = np.asarray(picked_info[:3], dtype=float)

        if picked_xyz is None or len(picked_xyz) < 3:
            return

        # Find closest point in current rendered view by squared distance
        dists_sq = np.sum((rendered_pts - picked_xyz[:3]) ** 2, axis=1)
        local_idx = int(np.argmin(dists_sq))
        global_idx = int(rendered_indices[local_idx]) if rendered_indices is not None else int(local_idx)

        # Real world scaled coordinates of picked point
        rx = float(data['coords'][global_idx, 0])
        ry = float(data['coords'][global_idx, 1])
        rz = float(data['coords'][global_idx, 2])
        rel_z = float(rendered_pts[local_idx, 2])

        # Place highlight marker on picked point
        self._add_picked_marker(side, rx, ry, rz)

        # Tree ID / ITC value of picked point
        tree_field = data.get('tree_field')
        if tree_field and tree_field in las.point_format.dimension_names:
            tree_val = int(las[tree_field][global_idx])
            if tree_val > 0:
                tree_html = f"<span style='color: #FFEB3B; font-weight: bold;'>Tree {tree_val}</span>"
            else:
                tree_html = "<span style='color: #888888;'>0 (Ground / Unassigned)</span>"
        else:
            tree_val = None
            tree_html = "<span style='color: #888888;'>N/A</span>"

        # Active visual color mode parameter value
        color_mode = color_combo.currentText()
        if color_mode == "ITC / Tree ID":
            vis_param_html = f"ITC: <b>{tree_val if tree_val is not None else 'N/A'}</b>"
        elif color_mode == "Elevation (Z)":
            vis_param_html = f"Elevation: <b>{rz:.2f} m</b>"
        elif color_mode == "RGB (Natural)" and 'red' in las.point_format.dimension_names:
            r = int(las.red[global_idx])
            g = int(las.green[global_idx])
            b = int(las.blue[global_idx])
            vis_param_html = f"RGB: <b>({r}, {g}, {b})</b>"
        elif color_mode in las.point_format.dimension_names:
            c_val = las[color_mode][global_idx]
            if color_mode == "Classification":
                asprs_map = {0: "Created", 1: "Unassigned", 2: "Ground", 3: "Low Veg", 4: "Med Veg", 5: "High Veg", 6: "Building", 7: "Noise", 8: "Key-point", 9: "Water", 12: "Overlap"}
                c_name = asprs_map.get(int(c_val), "")
                vis_param_html = f"Classification: <b>{int(c_val)} ({c_name})</b>" if c_name else f"Classification: <b>{int(c_val)}</b>"
            elif isinstance(c_val, (float, np.floating)):
                vis_param_html = f"{color_mode}: <b>{c_val:.2f}</b>"
            else:
                vis_param_html = f"{color_mode}: <b>{c_val}</b>"
        else:
            vis_param_html = "Default Green"

        chips = []
        if 'classification' in las.point_format.dimension_names and color_mode != "Classification":
            c = int(las.classification[global_idx])
            asprs_map = {0: "Created", 1: "Unassigned", 2: "Ground", 3: "Low Veg", 4: "Med Veg", 5: "High Veg", 6: "Building", 7: "Noise", 8: "Key-point", 9: "Water", 12: "Overlap"}
            c_name = asprs_map.get(c, "")
            chips.append(f"Class: <b>{c}{f' ({c_name})' if c_name else ''}</b>")
        if 'intensity' in las.point_format.dimension_names and color_mode != "Intensity":
            inte = int(las.intensity[global_idx])
            if inte > 0:
                chips.append(f"Intensity: <b>{inte}</b>")
        if 'return_number' in las.point_format.dimension_names:
            rn = int(las.return_number[global_idx])
            nr = int(las.number_of_returns[global_idx]) if 'number_of_returns' in las.point_format.dimension_names else 0
            if nr > 0:
                chips.append(f"Return: <b>{rn}/{nr}</b>")

        chips_html = (" &nbsp;|&nbsp; " + " &nbsp;|&nbsp; ".join(chips)) if chips else ""

        html = (
            f"<b> Selected Point [View {side}]:</b> &nbsp; "
            f"<b>ITC / Tree ID:</b> {tree_html} &nbsp;|&nbsp; "
            f"<b>Active Param [{color_mode}]:</b> <span style='color: #4CAF50;'>{vis_param_html}</span> &nbsp;|&nbsp; "
            f"<b>Height:</b> {rz:.2f}m (Rel: {rel_z:+.2f}m) &nbsp;|&nbsp; "
            f"<b>XYZ:</b> ({rx:.2f}, {ry:.2f}, {rz:.2f})"
            f"{chips_html}"
        )
        pick_label.setText(html)

        # ── Find Spatially Corresponding Point on the Other View ──
        other_tree_val = None
        other_dist = None
        ox, oy, oz = rx, ry, rz
        other_side = 'B' if side == 'A' else 'A'

        if other_data is not None:
            kdtree = self._get_kdtree(other_data)
            if kdtree is not None:
                other_dist, other_idx = kdtree.query([rx, ry, rz], k=1)
                other_idx = int(other_idx)
                ox = float(other_data['coords'][other_idx, 0])
                oy = float(other_data['coords'][other_idx, 1])
                oz = float(other_data['coords'][other_idx, 2])

                other_las = other_data['las']
                other_tree_field = other_data.get('tree_field')
                if other_tree_field and other_tree_field in other_las.point_format.dimension_names:
                    other_tree_val = int(other_las[other_tree_field][other_idx])
                    if other_tree_val > 0:
                        other_tree_html = f"<span style='color: #00E5FF; font-weight: bold;'>Tree {other_tree_val}</span>"
                    else:
                        other_tree_html = "<span style='color: #888888;'>0 (Ground / Unassigned)</span>"
                else:
                    other_tree_html = "<span style='color: #888888;'>N/A</span>"

                # Place matching marker on other viewport
                self._add_picked_marker(other_side, ox, oy, oz)

                other_html = (
                    f"<b> Spatially Matched Point [View {other_side}]:</b> &nbsp; "
                    f"<b>ITC / Tree ID:</b> {other_tree_html} &nbsp;|&nbsp; "
                    f"<b>Distance to Pick:</b> <span style='color: #00E676;'>{other_dist:.3f} m</span> &nbsp;|&nbsp; "
                    f"<b>XYZ:</b> ({ox:.2f}, {oy:.2f}, {oz:.2f})"
                )
                other_pick_label.setText(other_html)

        # Record picked values for both sides
        if side == 'A':
            self.last_picked_a = (tree_val, rz, vis_param_html, (rx, ry, rz))
            if other_data is not None:
                self.last_picked_b = (other_tree_val, oz, f"Spatial Match (Δ={other_dist:.2f}m)" if other_dist is not None else "Matched", (ox, oy, oz))
        else:
            self.last_picked_b = (tree_val, rz, vis_param_html, (rx, ry, rz))
            if other_data is not None:
                self.last_picked_a = (other_tree_val, oz, f"Spatial Match (Δ={other_dist:.2f}m)" if other_dist is not None else "Matched", (ox, oy, oz))

        self._update_picked_comparison_banner()

        # If Isolate Mode is ON, isolate both corresponding trees
        if getattr(self, 'isolate_mode_btn', None) and self.isolate_mode_btn.isChecked():
            tree_a_target = tree_val if side == 'A' else other_tree_val
            tree_b_target = other_tree_val if side == 'A' else tree_val
            self._isolate_trees(tree_a_target, tree_b_target)

    def _update_picked_comparison_banner(self):
        """Update bottom banner comparing picked point in View A vs View B."""
        pa = getattr(self, 'last_picked_a', None)
        pb = getattr(self, 'last_picked_b', None)
        if not pa and not pb:
            return

        parts = ["<b> Picked Comparison:</b>"]
        if pa and self.data_a:
            t_str = f"Tree {pa[0]}" if pa[0] is not None and pa[0] > 0 else (f"ITC {pa[0]}" if pa[0] is not None else "0 (Ground)")
            parts.append(f"<b>View A:</b> <span style='color: #FFEB3B;'>{t_str}</span> (Z={pa[1]:.2f}m)")
        if pb and self.data_b:
            t_str = f"Tree {pb[0]}" if pb[0] is not None and pb[0] > 0 else (f"ITC {pb[0]}" if pb[0] is not None else "0 (Ground)")
            parts.append(f"<b>View B:</b> <span style='color: #00E5FF;'>{t_str}</span> (Z={pb[1]:.2f}m)")

        if pa and pb and pa[0] is not None and pb[0] is not None:
            if pa[0] == pb[0]:
                dz = pb[1] - pa[1]
                parts.append(f"<span style='color: #4CAF50; font-weight: bold;'>[Identical ID] Tree {pa[0]} (ΔZ: {dz:+.2f}m)</span>")
            else:
                parts.append(f"<span style='color: #00E676; font-weight: bold;'>[Spatially Matched] View A: Tree {pa[0]} &lt;-&gt; View B: Tree {pb[0]}</span>")

        self.picked_compare_label.setText(" &nbsp;•&nbsp; ".join(parts))
        self.picked_compare_label.setVisible(True)

    def _clear_pick(self, side=None):
        """Clear picked point highlight and text for a given viewport (or both)."""
        sides = ['A', 'B'] if side is None else [side]
        for s in sides:
            plotter = self.p1 if s == 'A' else self.p2
            pick_label = self.pick_a_label if s == 'A' else self.pick_b_label
            try:
                plotter.remove_actor('picked_point_marker')
                plotter.render()
            except Exception:
                pass
            pick_label.setText(" Click any point in the 3D view to inspect its ITC, coordinates, and parameters")
            if s == 'A':
                self.last_picked_a = None
            else:
                self.last_picked_b = None
        self._update_picked_comparison_banner()

    def _on_isolate_mode_toggled(self, checked):
        if checked:
            self.isolate_mode_btn.setText("Isolate Picked Tree: ON")
            pa = getattr(self, 'last_picked_a', None)
            pb = getattr(self, 'last_picked_b', None)
            tree_a = pa[0] if pa and pa[0] is not None and pa[0] > 0 else None
            tree_b = pb[0] if pb and pb[0] is not None and pb[0] > 0 else None
            if tree_a or tree_b:
                self._isolate_trees(tree_a, tree_b)
        else:
            self.isolate_mode_btn.setText("Isolate Picked Tree: OFF")

    def _isolate_picked_tree_button_clicked(self, side):
        pa = getattr(self, 'last_picked_a', None)
        pb = getattr(self, 'last_picked_b', None)
        tree_a = pa[0] if pa and pa[0] is not None and pa[0] > 0 else None
        tree_b = pb[0] if pb and pb[0] is not None and pb[0] > 0 else None
        if tree_a or tree_b:
            self._isolate_trees(tree_a, tree_b)
        else:
            lbl = self.pick_a_label if side == 'A' else self.pick_b_label
            lbl.setText("<span style='color: #FF9800; font-weight: bold;'>No tree at picked location. Click on a tree point in the 3D viewport first.</span>")

    def _isolate_trees(self, tree_a, tree_b):
        """Isolate tree_a on View A and tree_b on View B (spatially matched trees from different models)."""
        found_a = False
        found_b = False

        if tree_a is not None and tree_a > 0:
            for i in range(self.tree_a_combo.count()):
                if self.tree_a_combo.itemData(i) == tree_a:
                    if self.tree_a_combo.currentIndex() != i:
                        self.tree_a_combo.setCurrentIndex(i)
                    else:
                        self._update_view('A')
                    found_a = True
                    break
            if not found_a:
                self._update_view('A')
        else:
            if self.tree_a_combo.currentIndex() != 0:
                self.tree_a_combo.setCurrentIndex(0)
            else:
                self._update_view('A')

        if tree_b is not None and tree_b > 0:
            for i in range(self.tree_b_combo.count()):
                if self.tree_b_combo.itemData(i) == tree_b:
                    if self.tree_b_combo.currentIndex() != i:
                        self.tree_b_combo.setCurrentIndex(i)
                    else:
                        self._update_view('B')
                    found_b = True
                    break
            if not found_b:
                self._update_view('B')
        else:
            if self.tree_b_combo.currentIndex() != 0:
                self.tree_b_combo.setCurrentIndex(0)
            else:
                self._update_view('B')

        # Reset camera on both views so the isolated tree is zoomed and centered
        self.reset_both_cameras()

        # Re-apply markers if points were stored
        pa = getattr(self, 'last_picked_a', None)
        pb = getattr(self, 'last_picked_b', None)
        if pa and len(pa) >= 4 and pa[3] is not None:
            self._add_picked_marker('A', *pa[3])
        if pb and len(pb) >= 4 and pb[3] is not None:
            self._add_picked_marker('B', *pb[3])

        # Update picked comparison banner with isolation notice
        status_parts = []
        if found_a and found_b:
            if tree_a == tree_b:
                status_parts.append(f"<b>[ISOLATED] Tree {tree_a}</b> (Identical ID in both models)")
            else:
                status_parts.append(f"<b>[ISOLATED SPATIAL PAIR] View A: Tree {tree_a} &lt;-&gt; View B: Tree {tree_b}</b>")
            status_parts.append("<span style='color: #00E676; font-weight: bold;'>Both views isolated to corresponding physical tree</span>")
        elif found_a:
            status_parts.append(f"<b>[ISOLATED] View A: Tree {tree_a}</b> | <span style='color: #FFEB3B;'>View B: Ground / Unassigned</span>")
        elif found_b:
            status_parts.append(f"<b>[ISOLATED] View B: Tree {tree_b}</b> | <span style='color: #00E5FF;'>View A: Ground / Unassigned</span>")
        else:
            status_parts.append("<span style='color: #FF9800; font-weight: bold;'>Picked location has no tree segmentation in either model</span>")

        status_parts.append("<span style='color: #aaaaaa;'>— Click 'Show All Trees' to exit isolation</span>")

        if hasattr(self, 'picked_compare_label'):
            self.picked_compare_label.setText(" &nbsp;|&nbsp; ".join(status_parts))
            self.picked_compare_label.setVisible(True)


    def show_all_trees(self):
        """Reset tree filters across both views to display all trees."""
        if self.tree_a_combo.currentIndex() != 0:
            self.tree_a_combo.setCurrentIndex(0)
        else:
            self._update_view('A')

        if self.tree_b_combo.currentIndex() != 0:
            self.tree_b_combo.setCurrentIndex(0)
        else:
            self._update_view('B')

        self.reset_both_cameras()
        if hasattr(self, 'picked_compare_label'):
            self.picked_compare_label.setText("<b>All Trees Visible:</b> Reset to full forest view across View A and View B.")
            self.picked_compare_label.setVisible(True)
            self.picked_compare_label.setVisible(True)

    def _on_p1_interact(self, *args):
        if self._syncing or not self.sync_cameras_check.isChecked():
            return
        self._syncing = True
        try:
            self.p2.camera.position = self.p1.camera.position
            self.p2.camera.focal_point = self.p1.camera.focal_point
            self.p2.camera.up = self.p1.camera.up
            self.p2.camera.view_angle = self.p1.camera.view_angle
            self.p2.camera.clipping_range = self.p1.camera.clipping_range
            self.p2.render()
        finally:
            self._syncing = False

    def _on_p2_interact(self, *args):
        if self._syncing or not self.sync_cameras_check.isChecked():
            return
        self._syncing = True
        try:
            self.p1.camera.position = self.p2.camera.position
            self.p1.camera.focal_point = self.p2.camera.focal_point
            self.p1.camera.up = self.p2.camera.up
            self.p1.camera.view_angle = self.p2.camera.view_angle
            self.p1.camera.clipping_range = self.p2.camera.clipping_range
            self.p1.render()
        finally:
            self._syncing = False

    def _set_preset_view(self, mode):
        for p in [self.p1, self.p2]:
            if mode == 'xy':
                p.view_xy()
            elif mode == 'xz':
                p.view_xz()
            elif mode == 'yz':
                p.view_yz()
            elif mode == 'iso':
                p.view_isometric()
        self.p1.render()
        self.p2.render()

    def reset_both_cameras(self):
        self.p1.reset_camera()
        self.p2.reset_camera()
        self.p1.render()
        self.p2.render()

    def swap_views(self):
        self.file_a_path, self.file_b_path = self.file_b_path, self.file_a_path
        self.data_a, self.data_b = self.data_b, self.data_a
        self._populate_file_combo(self.file_a_combo, self.file_a_path)
        self._populate_file_combo(self.file_b_combo, self.file_b_path)
        if self.data_a:
            self._update_tree_combo(self.tree_a_combo, self.data_a['trees'])
        if self.data_b:
            self._update_tree_combo(self.tree_b_combo, self.data_b['trees'])
        self._update_view('A', reset_cam=False)
        self._update_view('B', reset_cam=False)
        self._update_summary()

    def _update_summary(self):
        if not self.data_a or not self.data_b:
            return
        diff_pts = self.data_b['num_points'] - self.data_a['num_points']
        pct_pts = (diff_pts / max(self.data_a['num_points'], 1)) * 100
        diff_trees = len(self.data_b['trees']) - len(self.data_a['trees'])
        diff_h = self.data_b['dz'] - self.data_a['dz']

        self.summary_label.setText(
            f"Comparison: {self.data_a['name']} vs {self.data_b['name']}  |  "
            f"Points: {self.data_a['num_points']:,} vs {self.data_b['num_points']:,} (Diff: {diff_pts:+,} / {pct_pts:+.1f}%)  |  "
            f"Trees: {len(self.data_a['trees'])} vs {len(self.data_b['trees'])} (Diff: {diff_trees:+})  |  "
            f"Height Extent: {self.data_a['dz']:.2f}m vs {self.data_b['dz']:.2f}m (Diff: {diff_h:+.2f}m)"
        )

    def show_metrics_diff(self):
        if not self.data_a or not self.data_b:
            QMessageBox.information(self, "Metrics", "Please load both File A and File B to compare metrics.")
            return

        diff_pts = self.data_b['num_points'] - self.data_a['num_points']
        pct_pts = (diff_pts / max(self.data_a['num_points'], 1)) * 100
        diff_trees = len(self.data_b['trees']) - len(self.data_a['trees'])

        text = (
            f"=== POINT CLOUD COMPARISON REPORT ===\n\n"
            f"File A: {self.data_a['name']} ({self.data_a['path']})\n"
            f"File B: {self.data_b['name']} ({self.data_b['path']})\n\n"
            f"----------------------------------------------------------------------\n"
            f"{'Metric':<22} | {'File A':<18} | {'File B':<18} | {'Difference':<15}\n"
            f"----------------------------------------------------------------------\n"
            f"{'Point Count':<22} | {self.data_a['num_points']:<18,} | {self.data_b['num_points']:<18,} | {diff_pts:+,} ({pct_pts:+.1f}%)\n"
            f"{'Segmented Trees':<22} | {len(self.data_a['trees']):<18} | {len(self.data_b['trees']):<18} | {diff_trees:+}\n"
            f"{'Extent X (m)':<22} | {self.data_a['dx']:<18.2f} | {self.data_b['dx']:<18.2f} | {self.data_b['dx'] - self.data_a['dx']:+.2f}\n"
            f"{'Extent Y (m)':<22} | {self.data_a['dy']:<18.2f} | {self.data_b['dy']:<18.2f} | {self.data_b['dy'] - self.data_a['dy']:+.2f}\n"
            f"{'Height Z (m)':<22} | {self.data_a['dz']:<18.2f} | {self.data_b['dz']:<18.2f} | {self.data_b['dz'] - self.data_a['dz']:+.2f}\n"
            f"----------------------------------------------------------------------\n"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("Side-by-Side Comparison Metrics")
        msg.setText(text)
        msg.setFont(self.font())
        msg.exec()

    def closeEvent(self, event):
        try:
            self.p1.close()
            self.p2.close()
        except Exception:
            pass
        super().closeEvent(event)

