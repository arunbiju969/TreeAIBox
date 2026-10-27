"""Dialog for importing point features from an ESRI File Geodatabase."""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFileDialog,
    QMessageBox, QComboBox, QCheckBox, QGroupBox, QSpinBox,
    QDoubleSpinBox, QDialogButtonBox, QPushButton, QColorDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class GdbImportDialog(QDialog):
    """Dialog for importing points from an ESRI File Geodatabase (.gdb) to view in 3D space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gdb_path = None
        self.selected_layer = None
        self.z_field = None
        self.z_constant = 0.0
        self.color = "#FF3333"
        self.point_size = 8
        self.render_spheres = True
        self._layer_schema = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Import Points from Geodatabase")
        self.setModal(True)
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Import & View Point Features from Geodatabase (.gdb)")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #4CAF50; margin-bottom: 4px;")
        title.setWordWrap(True)
        layout.addWidget(title)

        gdb_group = QGroupBox("Geodatabase")
        gdb_layout = QHBoxLayout(gdb_group)
        self.gdb_path_label = QLabel("No geodatabase selected")
        self.gdb_path_label.setStyleSheet("color: #bbbbbb;")
        self.gdb_path_label.setWordWrap(True)
        gdb_layout.addWidget(self.gdb_path_label, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_gdb)
        gdb_layout.addWidget(browse_btn)
        layout.addWidget(gdb_group)

        layer_group = QGroupBox("Feature Layer")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_combo = QComboBox()
        self.layer_combo.setEnabled(False)
        self.layer_combo.addItem("<Select a geodatabase first>")
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        self.layer_info_label = QLabel("")
        self.layer_info_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        layer_layout.addWidget(self.layer_info_label)
        layout.addWidget(layer_group)

        z_group = QGroupBox("Z Elevation")
        z_layout = QGridLayout(z_group)
        z_layout.setSpacing(8)
        z_layout.addWidget(QLabel("Z elevation field:"), 0, 0)
        self.z_field_combo = QComboBox()
        self.z_field_combo.setEnabled(False)
        self.z_field_combo.setToolTip(
            "Attribute column containing Z elevation (metres).\n"
            "<Use geometry Z / constant> reads Z from feature geometry if present."
        )
        self.z_field_combo.currentTextChanged.connect(self._on_z_field_changed)
        z_layout.addWidget(self.z_field_combo, 0, 1)
        self.z_const_label = QLabel("Constant Z value (m):")
        self.z_const_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        z_layout.addWidget(self.z_const_label, 1, 0)
        self.z_const_spin = QDoubleSpinBox()
        self.z_const_spin.setRange(-9999.0, 99999.0)
        self.z_const_spin.setDecimals(3)
        self.z_const_spin.setValue(0.0)
        self.z_const_spin.setFixedWidth(110)
        self.z_const_spin.setToolTip("Fallback height if features lack 3D Z coordinates.")
        z_layout.addWidget(self.z_const_spin, 1, 1)
        layout.addWidget(z_group)

        style_group = QGroupBox("3D Display Settings")
        style_layout = QGridLayout(style_group)
        style_layout.setSpacing(8)
        style_layout.addWidget(QLabel("Marker Color:"), 0, 0)
        self.color_combo = QComboBox()
        self.colors_map = {
            "Red": "#FF3333", "Yellow": "#FFDD33", "Cyan": "#33DDFF",
            "Magenta": "#FF33FF", "Lime Green": "#33FF57",
            "Bright Orange": "#FF8C00", "White": "#FFFFFF", "Blue": "#3366FF"
        }
        for name in self.colors_map:
            self.color_combo.addItem(name)
        style_layout.addWidget(self.color_combo, 0, 1)
        style_layout.addWidget(QLabel("Point Size / Radius:"), 1, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 50)
        self.size_spin.setValue(8)
        style_layout.addWidget(self.size_spin, 1, 1)
        self.spheres_cb = QCheckBox("Render as 3D Spheres")
        self.spheres_cb.setChecked(True)
        style_layout.addWidget(self.spheres_cb, 2, 0, 1, 2)
        layout.addWidget(style_group)

        note = QLabel("\u2139\ufe0f  Requires: pyogrio and geopandas  (pip install pyogrio geopandas)")
        note.setStyleSheet("color: #888888; font-size: 10px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Display in 3D")
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(self.ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _browse_gdb(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select ESRI File Geodatabase (.gdb folder)", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if not path:
            return
        if not path.lower().endswith(".gdb"):
            QMessageBox.warning(self, "Not a Geodatabase",
                                f"Please select a folder with the .gdb extension.\n\nSelected: {path}")
            return
        self.gdb_path = path
        self.gdb_path_label.setText(path)
        self.gdb_path_label.setStyleSheet("color: #e0e0e0;")
        self._populate_layers()

    def _populate_layers(self):
        has_pyogrio = False
        has_fiona = False
        try:
            import pyogrio; has_pyogrio = True
        except ImportError:
            pass
        try:
            import fiona; has_fiona = True
        except ImportError:
            pass
        if not has_pyogrio and not has_fiona:
            QMessageBox.critical(self, "Missing Dependency",
                "pyogrio or fiona is required.\n\npip install pyogrio geopandas")
            return
        point_layers = []
        if has_pyogrio:
            try:
                import pyogrio
                for lyr, gtype in pyogrio.list_layers(self.gdb_path):
                    if gtype and "point" in str(gtype).lower():
                        point_layers.append(lyr)
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Could not list layers:\n{exc}"); return
        else:
            try:
                import fiona
                for lyr in fiona.listlayers(self.gdb_path):
                    try:
                        with fiona.open(self.gdb_path, layer=lyr) as src:
                            if "point" in src.schema.get("geometry","").lower():
                                point_layers.append(lyr)
                    except Exception:
                        pass
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Could not list layers:\n{exc}"); return
        self.layer_combo.clear()
        if not point_layers:
            self.layer_combo.addItem("<No Point layers found>")
            self.layer_combo.setEnabled(False)
            self.layer_info_label.setText("No Point layers found in this GDB.")
            return
        self.layer_combo.addItem("<Select layer>")
        for lyr in point_layers:
            self.layer_combo.addItem(lyr)
        self.layer_combo.setEnabled(True)

    def _on_layer_changed(self, layer_name):
        if not self.gdb_path or not layer_name or layer_name.startswith("<"):
            self.ok_btn.setEnabled(False); return
        try:
            import pyogrio
            info = pyogrio.read_info(self.gdb_path, layer=layer_name)
            gtype = info.get("geometry_type","?")
            n = info.get("features", 0)
            fields = info.get("fields", [])
            dtypes = info.get("dtypes", [])
            self._layer_schema = dict(zip(fields, [str(d) for d in dtypes]))
            self.layer_info_label.setText(f"Geometry: {gtype}  |  Features: {n:,}  |  Fields: {len(self._layer_schema)}")
        except ImportError:
            try:
                import fiona
                with fiona.open(self.gdb_path, layer=layer_name) as src:
                    schema = src.schema
                    self._layer_schema = schema.get("properties",{})
                    self.layer_info_label.setText(
                        f"Geometry: {schema.get('geometry','?')}  |  Features: {len(src):,}  |  Fields: {len(self._layer_schema)}")
            except Exception as exc:
                self.layer_info_label.setText(f"Error: {exc}")
                self._layer_schema = {}; self.ok_btn.setEnabled(False); return
        except Exception as exc:
            self.layer_info_label.setText(f"Error: {exc}")
            self._layer_schema = {}; self.ok_btn.setEnabled(False); return
        self.z_field_combo.clear()
        self.z_field_combo.addItem("<Use geometry Z / constant>")
        for f, t in self._layer_schema.items():
            if any(k in str(t).lower() for k in ("int","float","num")):
                self.z_field_combo.addItem(f)
        self.z_field_combo.setEnabled(True)
        self.ok_btn.setEnabled(True)

    def _on_z_field_changed(self, field_name):
        use_const = field_name.startswith("<")
        self.z_const_label.setStyleSheet(
            "color: #e0e0e0; font-size: 10px;" if use_const else "color: #555555; font-size: 10px;"
        )
        self.z_const_spin.setEnabled(use_const)

    def _on_ok(self):
        if not self.gdb_path:
            QMessageBox.warning(self, "No GDB", "Please select a geodatabase first."); return
        layer_name = self.layer_combo.currentText()
        if not layer_name or layer_name.startswith("<"):
            QMessageBox.warning(self, "No Layer", "Please select a feature layer."); return
        self.selected_layer = layer_name
        z_txt = self.z_field_combo.currentText()
        self.z_field = None if z_txt.startswith("<") else z_txt
        self.z_constant = self.z_const_spin.value()
        self.accept()

    def get_import_params(self):
        color_name = self.color_combo.currentText()
        return {
            "gdb_path": self.gdb_path,
            "layer": self.selected_layer,
            "z_field": self.z_field,
            "z_constant": self.z_constant,
            "color": self.colors_map.get(color_name, "#FF3333"),
            "color_name": color_name,
            "point_size": self.size_spin.value(),
            "render_spheres": self.spheres_cb.isChecked(),
        }



