"""GDB overlay layers, picking, and layer rows. (TreeVisualizerGUI mixin)."""

import os
import sys
import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QDialog, QMessageBox, QFileDialog, QWidget, QHBoxLayout,
    QVBoxLayout, QLabel, QCheckBox, QPushButton, QSlider, QColorDialog,
    QFrame, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from forestry_suite.dialogs.gdb_import import GdbImportDialog


class GdbMixin:
    """GDB overlay layers, picking, and layer rows."""

    def import_points_from_gdb(self):
        """Open GDB import dialog and render features in 3D. No LAS file required."""
        dlg = GdbImportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        params = dlg.get_import_params()
        gdb_path       = params["gdb_path"]
        layer_name     = params["layer"]
        z_field        = params["z_field"]
        z_constant     = params["z_constant"]
        color          = params["color"]
        color_name     = params["color_name"]
        point_size     = params["point_size"]
        render_spheres = params["render_spheres"]

        try:
            import geopandas as gpd
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                "geopandas is required.\n\npip install pyogrio geopandas"); return

        try:
            import pyogrio  # noqa
        except ImportError:
            try:
                import fiona  # noqa
            except ImportError:
                QMessageBox.critical(self, "Missing Dependency",
                    "pyogrio or fiona is required.\n\npip install pyogrio geopandas"); return

        if not self.plotter:
            QMessageBox.warning(self, "Warning", "3D viewer not initialized."); return

        self.log_to_console(f"\U0001f4c2 Reading GDB layer: {layer_name} from {os.path.basename(gdb_path)}")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        QApplication.processEvents()

        try:
            gdf = gpd.read_file(gdb_path, layer=layer_name)
        except Exception as exc:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Read Error", f"Failed to read layer:\n{exc}"); return

        if gdf.empty:
            self.progress_bar.setVisible(False)
            QMessageBox.warning(self, "Empty Layer", f"Layer '{layer_name}' has no features."); return

        n_features = len(gdf)
        self.log_to_console(f"\U0001f4ca GDB layer read: {n_features:,} features")

        try:
            xs = gdf.geometry.x.to_numpy(dtype=float)
            ys = gdf.geometry.y.to_numpy(dtype=float)
            if z_field:
                zs = gdf[z_field].to_numpy(dtype=float)
            else:
                try:
                    zs = gdf.geometry.z.to_numpy(dtype=float)
                    if not np.all(np.isfinite(zs)):
                        raise ValueError("Non-finite Z")
                except Exception:
                    zs = np.full(n_features, z_constant, dtype=float)
                    self.log_to_console(f"\u2139\ufe0f  No geometry Z – using constant {z_constant:.3f} m")
        except Exception as exc:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Coordinate Error", f"Could not extract XYZ:\n{exc}"); return

        try:
            coords = np.column_stack([xs, ys, zs])
            actor_name = f"gdb_layer_{layer_name}"
            idx = 2
            while actor_name in self.gdb_layers:
                actor_name = f"gdb_layer_{layer_name}_{idx}"
                idx += 1

            self.gdb_layers[actor_name] = {
                "coords": coords,
                "color": color,
                "color_name": color_name,
                "point_size": point_size,
                "render_spheres": render_spheres,
                "visible": True,
                "layer_name": layer_name,
                "n_features": n_features,
                "gdf": gdf,
                "label_field": None,
                "label_visible": False,
            }

            self.plotter.add_points(
                pv.PolyData(coords),
                color=color,
                point_size=point_size,
                render_points_as_spheres=render_spheres,
                name=actor_name
            )
            self._set_actor_pickable(actor_name, self.gdb_pick_mode)
            self.plotter.reset_camera()
            self.plotter.update()

            self._add_gdb_layer_row(actor_name)

            self.log_to_console(
                f"\u2705 Displayed {n_features:,} GDB points from \'{layer_name}\' ({color_name} markers)."
            )
            self.progress_bar.setVisible(False)
            QMessageBox.information(
                self, "Layer Rendered",
                f"Displayed {n_features:,} GDB points from \'{layer_name}\' in 3D viewer!\n\n"
                f"\u2022 Color: {color_name}\n"
                f"\u2022 Marker Size: {point_size}\n"
                f"\u2022 3D Spheres: {'Yes' if render_spheres else 'No'}\n\n"
                "Manage visibility and size in the \'GDB Layer Manager\' panel."
            )
        except Exception as exc:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Rendering Error", f"Could not render features:\n{exc}")
        finally:
            self.progress_bar.setVisible(False)


    def _restore_gdb_layers(self):
        """Re-render all visible GDB overlay layers after plotter.clear()."""
        if not self.plotter:
            return
        for actor_name, layer in self.gdb_layers.items():
            if not layer.get("visible", True):
                continue
            try:
                self.plotter.add_points(
                    pv.PolyData(layer["coords"]),
                    color=layer["color"],
                    point_size=layer["point_size"],
                    render_points_as_spheres=layer["render_spheres"],
                    name=actor_name,
                    reset_camera=False
                )
                self._set_actor_pickable(actor_name, self.gdb_pick_mode)
                # Restore labels if they were enabled
                if layer.get("label_visible") and layer.get("label_field"):
                    self._refresh_gdb_labels(actor_name)
            except Exception:
                pass
        # Re-arm point picking after scene rebuild
        self._rearm_point_picking()


    def _rearm_point_picking(self):
        """Re-enable point picking after the plotter has been cleared and repopulated.
        
        Must be called after every plotter.clear() + _restore_gdb_layers() to ensure
        accurate point picking across scene rebuilds (trunk detection, branch detection,
        color changes, etc.).
        """
        if self.plotter is None:
            return
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        try:
            self.plotter.enable_point_picking(callback=self.on_point_picked, show_message=False)
        except Exception:
            pass
        self._set_volume_overlay_actors_unpickable()


    def _set_actor_pickable(self, name, pickable):
        """Toggle picking on a named actor.

        Overlay layers (GDB imports, bounding box, DBH marker, etc.) are made
        non-pickable so they do not steal picks from the tree point cloud and
        corrupt point-picking accuracy.
        """
        if not self.plotter:
            return
        try:
            actors = self.plotter.actors
            actor = actors.get(name) if hasattr(actors, "get") else actors[name]
        except Exception:
            actor = None
        if actor is None:
            return
        candidates = [actor]
        try:
            prop = getattr(actor, "prop", None)
        except Exception:
            prop = None
        if prop is not None:
            candidates.append(prop)
        for obj in candidates:
            try:
                setter = getattr(obj, "SetPickable", None)
            except Exception:
                setter = None
            if callable(setter):
                try:
                    setter(bool(pickable))
                    return
                except Exception:
                    continue
            try:
                if hasattr(obj, "pickable"):
                    obj.pickable = bool(pickable)
                    return
            except Exception:
                continue


    def _set_volume_overlay_actors_unpickable(self):
        """Keep volume/DBH circle overlays out of the point picker.

        Loaded and calculated volume circles are dense PolyData (dozens of
        vertices per ring). If those actors stay pickable, vtkPointPicker
        snaps to the circle vertices instead of the tree point cloud.

        This iterates the actor dict ONCE and sets pickability directly on
        each actor (no per-actor re-lookup), so it stays fast even with
        thousands of actors.
        """
        if not self.plotter:
            return
        try:
            actors = self.plotter.actors
            items = list(actors.items()) if hasattr(actors, "items") else list(actors)
        except Exception:
            return

        overlay_tokens = (
            'dbh_',
            'volume_section_',
            'trunk_dbh_',
            'section_info_label',
            'volume_info_text',
            'circle',
            'trajectory',
        )

        def _set_pickable(obj, pickable):
            try:
                setter = getattr(obj, "SetPickable", None)
                if callable(setter):
                    setter(bool(pickable))
                    return
            except Exception:
                pass
            try:
                if hasattr(obj, "pickable"):
                    obj.pickable = bool(pickable)
            except Exception:
                pass

        for item in items:
            if isinstance(item, tuple):
                name, actor = item
            else:
                name, actor = item, None
            if name == 'tree_point_cloud':
                continue
            lname = str(name).lower()
            if not any(token in lname for token in overlay_tokens):
                continue
            if actor is not None:
                _set_pickable(actor, False)
                try:
                    prop = getattr(actor, "prop", None)
                except Exception:
                    prop = None
                if prop is not None:
                    _set_pickable(prop, False)


    def _on_gdb_pick_mode_toggled(self, checked):
        """Toggle between GDB point picking and tree point picking."""
        self.gdb_pick_mode = bool(checked)
        if hasattr(self, 'gdb_pick_mode_button'):
            self.gdb_pick_mode_button.setText("Pick GDB Points: ON" if checked else "Pick GDB Points: OFF")
        self._apply_pick_mode()
        if checked:
            self.log_to_console("GDB pick mode ON: click a GDB sphere to view its attributes.")
        else:
            self.log_to_console("GDB pick mode OFF: tree point picking restored.")


    def _apply_pick_mode(self):
        """Set actor pickability based on the current pick mode.

        GDB pick mode ON  -> GDB layers pickable, tree point cloud not pickable.
        GDB pick mode OFF -> tree point cloud pickable, GDB layers not pickable.
        """
        if not self.plotter:
            return
        gdb_mode = getattr(self, 'gdb_pick_mode', False)
        for actor_name in list(self.gdb_layers.keys()):
            self._set_actor_pickable(actor_name, gdb_mode)
        self._set_actor_pickable('tree_point_cloud', not gdb_mode)
        self._set_volume_overlay_actors_unpickable()


    def _extract_picked_position(self, picked_info):
        """Extract the 3D picked position from a PyVista pick callback payload."""
        try:
            if hasattr(picked_info, 'picked_point'):
                return np.asarray(picked_info.picked_point, dtype=float)
            if isinstance(picked_info, dict) and 'picked_point' in picked_info:
                return np.asarray(picked_info['picked_point'], dtype=float)
            if hasattr(picked_info, 'points') and picked_info.points is not None:
                return np.asarray(picked_info.points[0], dtype=float)
            if isinstance(picked_info, np.ndarray):
                return picked_info.ravel()
        except Exception:
            return None
        return None


    def _handle_gdb_pick(self, picked_info):
        """Resolve a pick against GDB overlay layers and show feature attributes."""
        picked_point = self._extract_picked_position(picked_info)
        if picked_point is None or len(picked_point) < 3:
            self.log_to_console("Warning: Could not resolve GDB pick position")
            return

        best = None  # (distance, actor_name, feature_index)
        for actor_name, layer in self.gdb_layers.items():
            if not layer.get("visible", True):
                continue
            coords = layer.get("coords")
            if coords is None or len(coords) == 0:
                continue
            d2 = np.sum((np.asarray(coords)[:, :3] - picked_point[:3]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            dist = float(np.sqrt(d2[idx]))
            if best is None or dist < best[0]:
                best = (dist, actor_name, idx)

        if best is None:
            self.log_to_console("No visible GDB points available to pick.")
            return

        dist, actor_name, idx = best
        self._show_gdb_feature_details(actor_name, idx)


    def _show_gdb_feature_details(self, actor_name, feature_index):
        """Log the geodatabase attributes for a picked GDB feature to the UI console."""
        layer = self.gdb_layers.get(actor_name, {})
        layer_label = layer.get("layer_name", actor_name)
        coords = layer.get("coords")
        gdf = layer.get("gdf")

        self.log_to_console("=" * 50)
        self.log_to_console(f" GDB Point Picked — Layer: {layer_label}  |  Feature #{feature_index}")
        self.log_to_console("-" * 50)

        if coords is not None and 0 <= feature_index < len(coords):
            x, y, z = np.asarray(coords)[feature_index][:3]
            self.log_to_console(f"[Point] X: {x:.3f}  Y: {y:.3f}  Z: {z:.3f}")
        self.log_to_console("-" * 50)

        if gdf is not None and 0 <= feature_index < len(gdf):
            try:
                geom_name = gdf.geometry.name
            except Exception:
                geom_name = 'geometry'
            row = gdf.iloc[feature_index]
            for col in gdf.columns:
                if col == geom_name:
                    continue
                self.log_to_console(f"   {col}: {row[col]}")
        else:
            self.log_to_console("(no attribute table stored for this layer)")

        self.log_to_console("=" * 50)


    def _add_gdb_layer_row(self, actor_name):
        """Add a control row to the GDB Layer Manager sidebar panel."""
        layer = self.gdb_layers[actor_name]

        # Use a QFrame card matching the viewer theme
        row = QFrame()
        row.setObjectName(f"gdb_row_{actor_name}")
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(10, 10, 10, 10)
        row_layout.setSpacing(10)
        row.setStyleSheet("""
            QFrame {
                background-color: #2e2e2e;
                border: 1px solid #444444;
                border-radius: 0px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)

        # Top line: checkbox + color dot + layer name + point count + remove button
        top = QHBoxLayout()
        top.setSpacing(6)

        vis_cb = QCheckBox()
        vis_cb.setChecked(True)
        vis_cb.setToolTip("Toggle layer visibility in 3D view")

        color_dot = QLabel("●")
        color_dot.setStyleSheet(f"color: {layer['color']}; font-size: 14px; font-weight: bold;")
        color_dot.setObjectName(f"color_dot_{actor_name}")  # store actor_name for later lookup

        # Color picker button
        color_btn = QPushButton("Color")
        color_btn.setFixedSize(24, 20)
        color_btn.setToolTip("Change point color for this layer")
        color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {layer['color']};
                border: 1px solid #555555;
                border-radius: 0px;
                font-size: 10px;
                padding: 0px;
            }}
            QPushButton:hover {{
                border-color: #888888;
            }}
        """)
        color_btn.clicked.connect(lambda _, an=actor_name, cd=color_dot, cb=color_btn:
            self._on_gdb_color_pick(an, cd, cb))

        name_lbl = QLabel(layer["layer_name"])
        name_lbl.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold;")
        name_lbl.setWordWrap(False)

        feat_lbl = QLabel(f"{layer['n_features']:,} pts")
        feat_lbl.setStyleSheet("color: #aaaaaa; font-size: 9px; background-color: #383838; padding: 2px 5px; border-radius: 0px;")

        remove_btn = QPushButton("Clear")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setToolTip("Remove layer from 3D view")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d2424;
                color: #ff6666;
                border: 1px solid #663333;
                border-radius: 0px;
                font-size: 10px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #5c2c2c;
                color: #ff9999;
                border-color: #884444;
            }
        """)
        remove_btn.clicked.connect(lambda _, an=actor_name, r=row: self._remove_gdb_layer(an, r))

        top.addWidget(vis_cb)
        top.addWidget(color_dot)
        top.addWidget(color_btn)
        top.addWidget(name_lbl, 1)
        top.addWidget(feat_lbl)
        top.addWidget(remove_btn)
        row_layout.addLayout(top)

        # Size slider line
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        size_lbl = QLabel("Size:")
        size_lbl.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        size_lbl.setFixedWidth(28)
        size_slider = QSlider(Qt.Orientation.Horizontal)
        size_slider.setMinimum(1)
        size_slider.setMaximum(50)
        size_slider.setValue(layer["point_size"])
        size_slider.setToolTip("Adjust marker point / sphere size")

        size_val_lbl = QLabel(str(layer["point_size"]))
        size_val_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px; min-width: 18px;")

        def on_size_change(val, an=actor_name, vl=size_val_lbl):
            vl.setText(str(val))
            self.gdb_layers[an]["point_size"] = val
            if self.gdb_layers[an]["visible"] and self.plotter:
                try:
                    self.plotter.add_points(
                        pv.PolyData(self.gdb_layers[an]["coords"]),
                        color=self.gdb_layers[an]["color"],
                        point_size=val,
                        render_points_as_spheres=self.gdb_layers[an]["render_spheres"],
                        name=an,
                        reset_camera=False
                    )
                    self._set_actor_pickable(an, self.gdb_pick_mode)
                    self.plotter.update()
                except Exception:
                    pass

        size_slider.valueChanged.connect(on_size_change)
        size_row.addWidget(size_lbl)
        size_row.addWidget(size_slider, 1)
        size_row.addWidget(size_val_lbl)
        row_layout.addLayout(size_row)

        # ── Label field row ─────────────────────────────────────────
        label_row = QHBoxLayout()
        label_row.setSpacing(6)

        label_field_lbl = QLabel("Label:")
        label_field_lbl.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        label_field_lbl.setFixedWidth(35)

        # Build the combo with GDF column names (excluding geometry column)
        gdf = layer.get("gdf")
        label_combo = QComboBox()
        label_combo.setStyleSheet("QComboBox { font-size: 10px; padding: 2px; }")
        label_combo.addItem("None", None)
        if gdf is not None:
            try:
                geom_name = gdf.geometry.name
            except Exception:
                geom_name = "geometry"
            for col in gdf.columns:
                if col == geom_name:
                    continue
                label_combo.addItem(str(col), str(col))

        # Restore previous selection if any
        saved_field = layer.get("label_field")
        if saved_field:
            idx = label_combo.findData(saved_field)
            if idx >= 0:
                label_combo.setCurrentIndex(idx)

        # Label visibility toggle
        label_vis_cb = QCheckBox("Show")
        label_vis_cb.setChecked(layer.get("label_visible", False))
        label_vis_cb.setToolTip("Toggle point label visibility for this layer")
        label_vis_cb.setStyleSheet("QCheckBox { color: #bbbbbb; font-size: 10px; }")

        def on_label_field_change(idx, an=actor_name, combo=label_combo):
            field = combo.itemData(idx)
            layer = self.gdb_layers.get(an)
            if layer is None:
                return
            layer["label_field"] = field
            if field and layer.get("label_visible", False):
                self._refresh_gdb_labels(an)
            elif not field:
                # Remove labels when "None" selected
                if self.plotter:
                    try:
                        self.plotter.remove_actor(f"{an}_labels")
                        self.plotter.update()
                    except Exception:
                        pass

        def on_label_vis_change(state, an=actor_name):
            layer = self.gdb_layers.get(an)
            if layer is None:
                return
            layer["label_visible"] = (state == 2)
            if state == 2 and layer.get("label_field"):
                self._refresh_gdb_labels(an)
            else:
                if self.plotter:
                    try:
                        self.plotter.remove_actor(f"{an}_labels")
                        self.plotter.update()
                    except Exception:
                        pass

        label_combo.currentIndexChanged.connect(on_label_field_change)
        label_vis_cb.stateChanged.connect(on_label_vis_change)

        label_row.addWidget(label_field_lbl)
        label_row.addWidget(label_combo, 1)
        label_row.addWidget(label_vis_cb)
        row_layout.addLayout(label_row)

        def on_vis_change(checked, an=actor_name):
            self.gdb_layers[an]["visible"] = checked
            if not self.plotter:
                return
            if checked:
                try:
                    li = self.gdb_layers[an]
                    self.plotter.add_points(
                        pv.PolyData(li["coords"]),
                        color=li["color"],
                        point_size=li["point_size"],
                        render_points_as_spheres=li["render_spheres"],
                        name=an,
                        reset_camera=False
                    )
                    self._set_actor_pickable(an, self.gdb_pick_mode)
                    # Restore labels if they were enabled
                    if li.get("label_visible") and li.get("label_field"):
                        self._refresh_gdb_labels(an)
                    self.plotter.update()
                except Exception:
                    pass
            else:
                try:
                    self.plotter.remove_actor(an)
                    # Also remove labels when hiding
                    try:
                        self.plotter.remove_actor(f"{an}_labels")
                    except Exception:
                        pass
                    self.plotter.update()
                except Exception:
                    pass

        vis_cb.stateChanged.connect(lambda state, an=actor_name: on_vis_change(state == 2, an))

        count = self.gdb_layers_container_layout.count()
        self.gdb_layers_container_layout.insertWidget(count - 1, row)
        self.gdb_layer_group.setVisible(True)


    def _remove_gdb_layer(self, actor_name, row_widget):
        """Remove a GDB layer from the 3D view and the sidebar."""
        if self.plotter:
            try:
                self.plotter.remove_actor(actor_name)
                # Also remove any associated label actor
                label_actor = f"{actor_name}_labels"
                try:
                    self.plotter.remove_actor(label_actor)
                except Exception:
                    pass
                self.plotter.update()
            except Exception:
                pass
        self.gdb_layers.pop(actor_name, None)
        row_widget.setParent(None)
        row_widget.deleteLater()
        if not self.gdb_layers:
            self.gdb_layer_group.setVisible(False)


    def _on_gdb_color_pick(self, actor_name, color_dot_label=None, color_button=None):
        """Open a color dialog and update the GDB layer's point color."""
        from PyQt6.QtWidgets import QColorDialog
        layer = self.gdb_layers.get(actor_name)
        if layer is None:
            return
        current = QColor(layer["color"])
        new_color = QColorDialog.getColor(current, self, "Pick Point Color")
        if not new_color.isValid():
            return  # user cancelled

        hex_color = new_color.name()
        self._refresh_gdb_layer_color(actor_name, hex_color, color_dot_label)
        # Update the color button background too
        if color_button is not None:
            color_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_color};
                    border: 1px solid #555555;
                    border-radius: 0px;
                    font-size: 10px;
                    padding: 0px;
                }}
                QPushButton:hover {{
                    border-color: #888888;
                }}
            """)


    def _refresh_gdb_layer_color(self, actor_name, new_color, color_dot_label=None):
        """Update a GDB layer's point color in both the data store and the 3D viewer."""
        layer = self.gdb_layers.get(actor_name)
        if layer is None:
            return
        layer["color"] = new_color
        layer["color_name"] = new_color  # store hex as name for simplicity

        if color_dot_label is not None:
            color_dot_label.setStyleSheet(f"color: {new_color}; font-size: 14px; font-weight: bold;")

        if self.plotter and layer.get("visible", True):
            try:
                self.plotter.add_points(
                    pv.PolyData(layer["coords"]),
                    color=new_color,
                    point_size=layer["point_size"],
                    render_points_as_spheres=layer["render_spheres"],
                    name=actor_name,
                    reset_camera=False,
                )
                self._set_actor_pickable(actor_name, self.gdb_pick_mode)
                # Re-apply labels with updated actor if they were visible
                if layer.get("label_visible") and layer.get("label_field"):
                    self._refresh_gdb_labels(actor_name)
                self.plotter.update()
            except Exception as e:
                self.log_to_console(f"[WARN] Failed to update GDB layer color: {e}")


    def _refresh_gdb_labels(self, actor_name):
        """Add or update point labels for a GDB layer based on the selected label field."""
        layer = self.gdb_layers.get(actor_name)
        if layer is None:
            return
        label_field = layer.get("label_field")
        gdf = layer.get("gdf")
        coords = layer.get("coords")
        label_actor = f"{actor_name}_labels"

        # Remove existing labels
        if self.plotter:
            try:
                self.plotter.remove_actor(label_actor)
            except Exception:
                pass

        if not label_field or gdf is None or coords is None:
            return
        if not layer.get("label_visible", False):
            return

        # Build label texts from the selected field
        try:
            if label_field not in gdf.columns:
                self.log_to_console(f"[WARN] Label field '{label_field}' not found in GDB layer attributes")
                return
            values = gdf[label_field].fillna("").astype(str).to_numpy()
            labels = [str(v) for v in values]

            if self.plotter:
                self.plotter.add_point_labels(
                    pv.PolyData(coords),
                    labels,
                    font_size=9,
                    text_color="white",
                    point_size=2,
                    always_visible=True,
                    name=label_actor,
                )
                self.plotter.update()
            self.log_to_console(f"[Tag]  {len(labels)} labels applied to '{layer.get('layer_name')}' (field: {label_field})")
        except Exception as e:
            self.log_to_console(f"[WARN] Failed to add GDB labels: {e}")



