"""LAS file loading and layer management. (TreeVisualizerGUI mixin)."""

import os
import sys
import numpy as np
import laspy
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QRadioButton, QComboBox, QDialogButtonBox, QWidget,
    QCheckBox, QPushButton, QListWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from forestry_suite.dialogs.las_comparison import LASComparisonDialog
try:
    from scripts.pointcloud_processing.las_utils import find_tree_id_dimension
except ImportError:
    from modules.las_utils import find_tree_id_dimension


class LasLayersMixin:
    """LAS file loading and layer management."""

    def load_las_file(self):
        """Open file dialog to add single or multiple LAS/LAZ files to the Layer Manager."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select LAS File(s)",
            "",
            "LAS files (*.las *.laz)"
        )
        if not file_paths:
            return

        added_paths = []
        for path in file_paths:
            rec = self._add_las_to_layers(path)
            if rec:
                added_paths.append(path)

        if added_paths:
            # Activate the last added layer
            self.activate_las_layer(added_paths[-1])


    def _find_tree_id_field(self, dimension_names):
        """Detect tree ID or crown scalar field from point dimensions."""
        return find_tree_id_dimension(dimension_names)


    def _prompt_tree_id_field(self, las_data):
        """Prompt user to select a scalar field as Tree ID / ITC, or initialize an empty/single-tree field."""
        available_fields = [
            f for f in las_data.point_format.dimension_names
            if f.lower() not in ['x', 'y', 'z']
        ]

        dialog = QDialog(self)
        dialog.setWindowTitle("Tree ID Field Selection")
        layout = QVBoxLayout(dialog)

        desc = QLabel(
            "No 'itc' or standard tree ID scalar field (e.g. 'treeID', 'tree_id') was found in this file.\n\n"
            "Please choose how you would like to handle tree identification:"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        radio_field = QRadioButton("Use existing scalar field as Tree ID:")
        layout.addWidget(radio_field)

        combo_field = QComboBox()
        preferred = ['classification', 'user_data', 'point_source_id']
        sorted_fields = sorted(available_fields, key=lambda x: (0 if x.lower() in preferred else 1, x))
        for f in sorted_fields:
            combo_field.addItem(f)
        layout.addWidget(combo_field)

        radio_empty = QRadioButton("Initialize blank ITC field (all points unassigned / 0) for manual segmentation")
        layout.addWidget(radio_empty)

        radio_single = QRadioButton("Treat all points as a single tree (Tree ID = 1)")
        layout.addWidget(radio_single)

        if available_fields:
            radio_field.setChecked(True)
        else:
            radio_empty.setChecked(True)
            radio_field.setEnabled(False)
            combo_field.setEnabled(False)

        radio_field.toggled.connect(combo_field.setEnabled)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            if radio_field.isChecked():
                return ('field', combo_field.currentText())
            elif radio_empty.isChecked():
                return ('empty', None)
            elif radio_single.isChecked():
                return ('single', None)
        return (None, None)


    def _add_las_to_layers(self, file_path):
        """Register a LAS file in self.las_layers if not already present."""
        norm_path = os.path.normpath(file_path)
        for layer in self.las_layers:
            if os.path.normpath(layer['path']) == norm_path:
                return layer  # Already present

        try:
            import laspy
            data = laspy.read(file_path)
            num_points = len(data.points)
            has_stemcls = 'stemcls' in data.point_format.dimension_names
            num_trees = 0
            tree_field = self._find_tree_id_field(data.point_format.dimension_names)
            if tree_field:
                itc_vals = np.array(data[tree_field])
                num_trees = len(np.unique(itc_vals[itc_vals > 0]))

            layer_rec = {
                'path': norm_path,
                'name': os.path.basename(file_path),
                'num_points': num_points,
                'num_trees': num_trees,
                'has_stemcls': has_stemcls,
                'checked': True
            }
            self.las_layers.append(layer_rec)
            self._update_las_layer_list_widget()
            return layer_rec
        except Exception as e:
            self.log_to_console(f"[WARN] Could not inspect metadata for {os.path.basename(file_path)}: {e}")
            return None


    def _update_las_layer_list_widget(self):
        """Re-populate the LAS layer list widget."""
        self._updating_las_layer_list = True
        self.las_layer_list.clear()

        # Count name occurrences to disambiguate identical filenames
        name_counts = {}
        for l in self.las_layers:
            name_counts[l['name']] = name_counts.get(l['name'], 0) + 1

        for layer in self.las_layers:
            name = layer['name']
            pts = layer['num_points']
            trees = layer['num_trees']
            p_norm = os.path.normpath(layer['path']).lower()
            if 'treelearn' in p_norm:
                tag = "TreeLearn"
            elif 'forestformer' in p_norm or 'forest_former' in p_norm or 'forest-former' in p_norm:
                tag = "ForestFormer"
            elif 'segmented_results' in p_norm or 'treeaibox' in p_norm:
                tag = "TreeAIBox"
            else:
                tag = os.path.basename(os.path.dirname(layer['path']))

            if name_counts.get(name, 0) > 1 or tag in ['TreeLearn', 'ForestFormer', 'TreeAIBox']:
                display_text = f"{name} [{tag}] ({trees} trees, {pts:,} pts)"
            else:
                display_text = f"{name} ({trees} trees, {pts:,} pts)"

            item = QListWidgetItem(display_text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if layer.get('checked', True) else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, layer['path'])

            # Bold font if this layer is currently active
            if self.current_las_path and os.path.normpath(self.current_las_path) == os.path.normpath(layer['path']):
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            self.las_layer_list.addItem(item)

        self._updating_las_layer_list = False


    def _on_las_layer_clicked(self, item):
        """Handle user clicking a layer item to switch active viewer."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and (not self.current_las_path or os.path.normpath(path) != os.path.normpath(self.current_las_path)):
            self.activate_las_layer(path)


    def _on_las_layer_check_changed(self, item):
        """Handle user toggling a layer checkbox."""
        if getattr(self, '_updating_las_layer_list', False):
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        is_checked = (item.checkState() == Qt.CheckState.Checked)
        for layer in self.las_layers:
            if os.path.normpath(layer['path']) == os.path.normpath(path):
                layer['checked'] = is_checked
                break


    def remove_selected_las_layer(self):
        """Remove the selected layer from self.las_layers."""
        current_item = self.las_layer_list.currentItem()
        if not current_item:
            return
        path = current_item.data(Qt.ItemDataRole.UserRole)
        self.las_layers = [l for l in self.las_layers if os.path.normpath(l['path']) != os.path.normpath(path)]
        self._update_las_layer_list_widget()
        if self.current_las_path and os.path.normpath(self.current_las_path) == os.path.normpath(path):
            if self.las_layers:
                self.activate_las_layer(self.las_layers[0]['path'])
            else:
                self.las_data = None
                self.current_las_path = None
                self.file_label.setText("No file loaded")
                self.tree_list.clear()


    def toggle_las_batch_checks(self):
        """Toggle check state of all LAS layers."""
        if not self.las_layers:
            return
        all_checked = all(l.get('checked', True) for l in self.las_layers)
        new_state = not all_checked
        for l in self.las_layers:
            l['checked'] = new_state
        self._update_las_layer_list_widget()
        if hasattr(self, 'toggle_las_batch_btn'):
            self.toggle_las_batch_btn.setText("Uncheck All" if new_state else "Check All")


    def activate_las_layer(self, file_path, silent=False):
        """Switch active LAS file, load data into memory, and update UI."""
        norm_path = os.path.normpath(file_path)
        # Ensure file is in layers list
        found = False
        for l in self.las_layers:
            if os.path.normpath(l['path']) == norm_path:
                found = True
                break
        if not found:
            self._add_las_to_layers(file_path)

        self.load_las_data(file_path, silent=silent)
        self._update_las_layer_list_widget()


    def open_las_comparison(self):
        """Open the Side-by-Side LAS Comparison window."""
        checked_paths = [
            layer['path'] for layer in self.las_layers
            if layer.get('checked', True) and os.path.isfile(layer['path'])
        ]
        all_paths = [
            layer['path'] for layer in self.las_layers
            if os.path.isfile(layer['path'])
        ]

        file_a = None
        file_b = None

        if self.current_las_path and os.path.isfile(self.current_las_path):
            file_a = self.current_las_path
        elif checked_paths:
            file_a = checked_paths[0]
        elif all_paths:
            file_a = all_paths[0]

        for p in checked_paths + all_paths:
            if os.path.normpath(p) != os.path.normpath(file_a if file_a else ""):
                file_b = p
                break

        if not file_a and not all_paths:
            file_a, _ = QFileDialog.getOpenFileName(
                self, "Select First LAS File (Left View)", "", "LAS files (*.las *.laz)"
            )
            if not file_a:
                return

        if not file_b and file_a:
            file_b, _ = QFileDialog.getOpenFileName(
                self, "Select Second LAS File (Right View) to Compare",
                os.path.dirname(file_a), "LAS files (*.las *.laz)"
            )

        if hasattr(self, 'comparison_window') and self.comparison_window is not None:
            try:
                self.comparison_window.close()
            except Exception:
                pass

        self.comparison_window = LASComparisonDialog(
            parent=self,
            initial_file_a=file_a,
            initial_file_b=file_b,
            loaded_layers=self.las_layers
        )
        self.comparison_window.showMaximized()
        self.comparison_window.raise_()
        self.comparison_window.activateWindow()
        self.log_to_console(" Opened Side-by-Side LAS Comparison Window")


    def load_las_data(self, file_path, silent=False):
        """Load LAS file and extract ITC values."""
        import time
        load_start = time.perf_counter()
        print(f"[LOAD] {time.strftime('%H:%M:%S')} - Starting to load: {os.path.basename(file_path)}")
        self.log_to_console(f"[File] Loading file: {os.path.basename(file_path)}")
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            QApplication.processEvents()

            # Load LAS file
            read_start = time.perf_counter()
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Reading LAS file...")
            self.las_data = laspy.read(file_path)
            read_time = time.perf_counter() - read_start
            num_points = len(self.las_data.points)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - LAS file read: {num_points:,} points, {file_size_mb:.1f} MB ({read_time:.2f}s)")
            self.log_to_console(f"[Metrics] File read: {num_points:,} points, {file_size_mb:.1f} MB in {read_time:.2f}s")
            
            self.file_label.setText(f"Loaded: {os.path.basename(file_path)}")
            # Remember the current LAS file path for automatic saves and volume data loading
            try:
                self.current_las_path = os.path.normpath(file_path)
                base_dir = os.path.dirname(self.current_las_path)
                las_basename = os.path.splitext(os.path.basename(self.current_las_path))[0]
                self.volume_folder_path = os.path.join(base_dir, 'volume', las_basename)
                if hasattr(self, 'volume_folder_label'):
                    self.volume_folder_label.setText(f"Volume Folder: {os.path.basename(self.volume_folder_path)}")
            except Exception:
                self.current_las_path = None

            # Check for ITC field or compatible tree ID field
            self._source_tree_id_field = None
            if 'itc' not in self.las_data.point_format.dimension_names:
                tree_field = self._find_tree_id_field(self.las_data.point_format.dimension_names)
                if tree_field:
                    self.log_to_console(f"Found tree ID field '{tree_field}'. Mapping to 'itc' for editing and visualization...")
                    try:
                        self.las_data.add_extra_dim(laspy.ExtraBytesParams(name='itc', type=self.las_data[tree_field].dtype))
                        self.las_data['itc'] = np.array(self.las_data[tree_field])
                        self._source_tree_id_field = tree_field
                    except Exception as e:
                        self.log_to_console(f"[WARN] Failed to add 'itc' extra dimension: {e}")
                else:
                    if not silent:
                        choice_type, choice_val = self._prompt_tree_id_field(self.las_data)
                        if choice_type == 'field' and choice_val:
                            self.log_to_console(f"Mapping selected field '{choice_val}' to 'itc'...")
                            try:
                                self.las_data.add_extra_dim(laspy.ExtraBytesParams(name='itc', type=self.las_data[choice_val].dtype))
                                self.las_data['itc'] = np.array(self.las_data[choice_val])
                                self._source_tree_id_field = choice_val
                            except Exception as e:
                                self.log_to_console(f"[WARN] Failed to map field to 'itc': {e}")
                        elif choice_type == 'empty':
                            self.log_to_console("Initializing blank 'itc' field (zeros)...")
                            try:
                                self.las_data.add_extra_dim(laspy.ExtraBytesParams(name='itc', type=np.int32))
                                self.las_data['itc'] = np.zeros(num_points, dtype=np.int32)
                            except Exception as e:
                                self.log_to_console(f"[WARN] Failed to initialize 'itc': {e}")
                        elif choice_type == 'single':
                            self.log_to_console("Initializing 'itc' field with Tree ID 1...")
                            try:
                                self.las_data.add_extra_dim(laspy.ExtraBytesParams(name='itc', type=np.int32))
                                self.las_data['itc'] = np.ones(num_points, dtype=np.int32)
                            except Exception as e:
                                self.log_to_console(f"[WARN] Failed to initialize 'itc': {e}")
                        else:
                            self.las_data = None
                            return
                    else:
                        self.las_data = None
                        return
            else:
                self._source_tree_id_field = 'itc'

            # Check for stemcls field for trunk visualization
            has_stemcls = 'stemcls' in self.las_data.point_format.dimension_names
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Fields: itc=True, stemcls={has_stemcls}")

            # Extract unique ITC values
            itc_start = time.perf_counter()
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Extracting ITC values...")
            itc_values = np.array(self.las_data['itc'])
            self.unique_itc_values = np.unique(itc_values[itc_values > 0]).astype(int)  # Exclude 0 (likely unassigned) and convert to int
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Found {len(self.unique_itc_values)} unique ITC values ({time.perf_counter() - itc_start:.3f}s)")

            # Filter tree IDs by stemcls = 2 if stemcls field exists
            # For large files (>50M points), ask user if they want to filter
            perform_stemcls_filter = False
            if has_stemcls:
                LARGE_FILE_THRESHOLD = 50_000_000  # 50 million points
                if num_points > LARGE_FILE_THRESHOLD and not silent:
                    reply = QMessageBox.question(
                        self,
                        "Large File Detected",
                        f"This file has {num_points:,} points ({len(self.unique_itc_values)} trees).\n\n"
                        f"Filtering by stemcls=2 (stem points) may take a while.\n\n"
                        f"Do you want to filter trees by stem presence?\n"
                        f"• Yes = Show only trees with stem points (slower)\n"
                        f"• No = Show all trees (faster)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No  # Default to No for large files
                    )
                    perform_stemcls_filter = (reply == QMessageBox.StandardButton.Yes)
                    print(f"[LOAD] {time.strftime('%H:%M:%S')} - User chose {'to filter' if perform_stemcls_filter else 'to skip filtering'} for large file")
                else:
                    perform_stemcls_filter = True  # Always filter for smaller files
                    
            if has_stemcls and perform_stemcls_filter:
                filter_start = time.perf_counter()
                print(f"[LOAD] {time.strftime('%H:%M:%S')} - Filtering by stemcls=2...")
                self.log_to_console(f"[Wait] Filtering {len(self.unique_itc_values)} trees by stem presence... (this may take a while)")
                QApplication.processEvents()  # Update UI to show message
                
                stemcls_values = np.array(self.las_data['stemcls'])
                filtered_ids = []
                for tree_id in self.unique_itc_values:
                    mask = (itc_values == tree_id) & (stemcls_values == 2)
                    if np.any(mask):
                        filtered_ids.append(tree_id)
                self.unique_itc_values = np.array(filtered_ids)
                print(f"[LOAD] {time.strftime('%H:%M:%S')} - Filtered to {len(self.unique_itc_values)} trees with stems ({time.perf_counter() - filter_start:.3f}s)")
                self.log_to_console(f"[OK] Filtered to {len(self.unique_itc_values)} trees with stem points")
            elif has_stemcls and not perform_stemcls_filter:
                print(f"[LOAD] {time.strftime('%H:%M:%S')} - Skipped stemcls filtering (user choice)")
                self.log_to_console(f" Skipped stem filtering - showing all {len(self.unique_itc_values)} trees")

            # Populate tree list widget
            list_start = time.perf_counter()
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Populating tree list...")
            self.tree_list.clear()

            # Include Ground / Unassigned item if ground points (itc == 0 or classification == 2) exist
            has_ground = False
            if 'itc' in self.las_data.point_format.dimension_names and np.any(itc_values == 0):
                has_ground = True
            elif 'classification' in self.las_data.point_format.dimension_names and np.any(np.asarray(self.las_data.classification) == 2):
                has_ground = True

            if has_ground:
                item_ground = QListWidgetItem("0 (Ground / Unassigned)")
                item_ground.setForeground(QColor("#B0BEC5"))
                self.tree_list.addItem(item_ground)

            for value in sorted(self.unique_itc_values):
                item = QListWidgetItem(str(value))
                # Color red if this tree has volume data
                if str(value) in self.trees_with_volume_data:
                    item.setForeground(QColor('red'))
                self.tree_list.addItem(item)
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Tree list populated ({time.perf_counter() - list_start:.3f}s)")

            # Populate color field combo box
            available_fields = list(self.las_data.point_format.dimension_names)
            color_fields = [field for field in available_fields
                           if field not in ['x', 'y', 'z']]  # Include itc for multiple tree coloring

            self.color_combo.clear()
            self.color_combo.addItem("Default (green)")
            
            # Check if RGB fields exist
            has_red = 'red' in available_fields
            has_green = 'green' in available_fields
            has_blue = 'blue' in available_fields
            if has_red and has_green and has_blue:
                self.color_combo.addItem("RGB (red, green, blue)")
            
            for field in sorted(color_fields):
                self.color_combo.addItem(field)

            # Auto-detect the tree-point filter field for this data output
            self.detect_tree_point_filter()

            self.visualize_button.setEnabled(True)
            self.prev_tree_button.setEnabled(True)
            self.next_tree_button.setEnabled(True)
            self.trunk_button.setEnabled(has_stemcls)
            self.batch_analyze_button.setEnabled(has_stemcls)
            self.crown_polygon_button.setEnabled(True)  # Always enabled since it doesn't require stemcls
            self.view_neighbors_button.setEnabled(True)
            self.draw_polygon_button.setEnabled(True)
            self.radius_export_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.fit_to_screen_button.setEnabled(True)

            total_time = time.perf_counter() - load_start
            print(f"[LOAD] {time.strftime('%H:%M:%S')} - Load complete: {total_time:.2f}s total")
            self.log_to_console(f"[OK] Loaded {len(self.unique_itc_values)} trees from {num_points:,} points in {total_time:.2f}s")
            
            if not silent:
                QMessageBox.information(self, "Success",
                                      f"Loaded {len(self.unique_itc_values)} unique tree IDs from {len(self.las_data.points)} points.")

        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to load LAS file:\n{str(e)}")
            self.las_data = None
        finally:
            self.progress_bar.setVisible(False)

    # ------------------------------------------------------------------
    # GDB Import & 3D Overlay
    # ------------------------------------------------------------------

    def save_las_file(self):
        """Save the modified LAS file with a summary of changes."""
        if self.las_data is None:
            QMessageBox.warning(self, "Warning", "No LAS data available.")
            return

        if not self.itc_changes:
            QMessageBox.information(self, "No Changes", "No ITC assignments have been made in this session.")
            return

        # Create summary text
        summary_text = "ITC Assignment Summary\n"
        summary_text += "=" * 50 + "\n\n"
        summary_text += f"Total assignments: {len(self.itc_changes)}\n"
        summary_text += f"Total points modified: {sum(change['num_points'] for change in self.itc_changes)}\n\n"

        summary_text += "Assignment Details:\n"
        summary_text += "-" * 40 + "\n"
        for i, change in enumerate(self.itc_changes, 1):
            timestamp = change['timestamp'].strftime("%H:%M:%S")
            summary_text += f"{i}. {timestamp} - ITC {change['new_itc']} assigned to {change['num_points']} points\n"

        # Show summary dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save LAS File - Change Summary")
        msg_box.setText(summary_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel)

        # Add export summary button
        export_button = msg_box.addButton("Export Summary", QMessageBox.ButtonRole.ActionRole)

        result = msg_box.exec()

        if result == QMessageBox.StandardButton.Save:
            # Save LAS file
            self._save_las_file()
        elif msg_box.clickedButton() == export_button:
            # Export summary
            self._export_summary(summary_text)
            # Then ask if they want to save
            reply = QMessageBox.question(self, "Save LAS File",
                                       "Summary exported. Do you also want to save the LAS file?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._save_las_file()


    def _save_las_file(self):
        """Internal method to save the LAS file."""
        try:
            # Get save file path
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("LAS files (*.las)")
            file_dialog.setDefaultSuffix("las")
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)

            if file_dialog.exec():
                save_path = file_dialog.selectedFiles()[0]

                # Save the LAS file
                self.las_data.write(save_path)

                QMessageBox.information(self, "Success",
                                      f"LAS file saved successfully to:\n{save_path}\n\n"
                                      f"All {len(self.itc_changes)} ITC assignments have been persisted.")

                # Clear changes after successful save
                self.itc_changes = []
                self.log_to_console(f"LAS file saved to {save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save LAS file:\n{str(e)}")


