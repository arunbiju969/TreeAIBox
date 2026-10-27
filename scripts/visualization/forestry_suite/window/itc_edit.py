"""ITC assignment and noise conversion on selections. (TreeVisualizerGUI mixin)."""

import numpy as np
from PyQt6.QtWidgets import QMessageBox, QDialog
from forestry_suite.dialogs.itc_assignment import ITCAssignmentDialog


class ItcEditMixin:
    """ITC assignment and noise conversion on selections."""

    def convert_to_noise_class(self, noise_itc_value=9999, analyzed_points=None):
        """Convert the currently analyzed points to noise class with specified ITC value."""
        print(f"DEBUG: convert_to_noise_class called with analyzed_points={analyzed_points}")
        print(f"DEBUG: type(analyzed_points)={type(analyzed_points)}")
        if analyzed_points is not None:
            print(f"DEBUG: len(analyzed_points)={len(analyzed_points) if hasattr(analyzed_points, '__len__') else 'no len'}")
        print(f"DEBUG: self.current_tree_points={self.current_tree_points}")
        print(f"DEBUG: type(self.current_tree_points)={type(self.current_tree_points)}")
        if self.current_tree_points is not None:
            print(f"DEBUG: len(self.current_tree_points)={len(self.current_tree_points) if hasattr(self.current_tree_points, '__len__') else 'no len'}")
        print(f"DEBUG: hasattr(self, 'last_analyzed_points')={hasattr(self, 'last_analyzed_points')}")
        if hasattr(self, 'last_analyzed_points'):
            print(f"DEBUG: self.last_analyzed_points={self.last_analyzed_points}")
            print(f"DEBUG: type(self.last_analyzed_points)={type(self.last_analyzed_points)}")
            if self.last_analyzed_points is not None:
                print(f"DEBUG: len(self.last_analyzed_points)={len(self.last_analyzed_points) if hasattr(self.last_analyzed_points, '__len__') else 'no len'}")
        
        if self.las_data is None:
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        # Check if we have analyzed points provided, otherwise try current/last points
        points_to_convert = analyzed_points
        print(f"DEBUG: initial points_to_convert={points_to_convert}")
        if points_to_convert is None or len(points_to_convert) == 0:
            print("DEBUG: points_to_convert is None or empty, using self.current_tree_points")
            points_to_convert = self.current_tree_points
        print(f"DEBUG: after first check, points_to_convert={points_to_convert}")
        if points_to_convert is None or len(points_to_convert) == 0:
            print("DEBUG: still None or empty, checking last_analyzed_points")
            # If no current points, check if we have any points from the last analysis
            if hasattr(self, 'last_analyzed_points') and self.last_analyzed_points is not None:
                points_to_convert = self.last_analyzed_points
                print(f"DEBUG: set points_to_convert to last_analyzed_points: {points_to_convert}")
            else:
                print("DEBUG: no points available, returning")
                QMessageBox.warning(self, "Warning", "No points available for conversion.")
                return

        print(f"DEBUG: final points_to_convert={points_to_convert}, type={type(points_to_convert)}")
        if points_to_convert is not None:
            print(f"DEBUG: len(points_to_convert)={len(points_to_convert)}")

        try:
            # Store the points we're converting for reference
            self.last_analyzed_points = points_to_convert
            
            # Use all available points for noise conversion
            self.selected_point_indices = np.arange(len(points_to_convert))
            self.selected_polygon = None  # No polygon for "convert all to noise"
            
            # Assign the noise ITC value
            self.assign_itc_to_selection(noise_itc_value)
            
            self.log_to_console(f"Converted {len(self.selected_point_indices)} points to noise class (ITC {noise_itc_value})")
            
            # Show success message
            QMessageBox.information(self, "Success", 
                                  f"Successfully converted {len(self.selected_point_indices)} points to noise class with ITC value {noise_itc_value}.")
            
        except Exception as e:
            self.log_to_console(f"Error converting to noise class: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to convert to noise class: {str(e)}")


    def assign_itc_to_selection(self, new_itc=None):
        """Assign new ITC value to selected points."""
        # Basic validation
        if self.selected_point_indices is None:
            QMessageBox.warning(self, "Warning", "No points selected (selection is None).")
            return

        if len(self.selected_point_indices) == 0:
            QMessageBox.warning(self, "Warning", "No points selected (selection is empty).")
            return

        if self.las_data is None:
            QMessageBox.warning(self, "Warning", "No LAS data available.")
            return

        # Get new ITC value from user (if not provided)
        if new_itc is None:
            # Get unique ITC values from selected points
            selected_itc_values = set()
            itc_array = np.array(self.las_data['itc'])
            
            # Map merged visualization indices back to original LAS indices
            if self.all_masks is not None and len(self.all_masks) > 0:
                try:
                    per_mask_indices = [np.where(mask)[0] for mask in self.all_masks]
                    global_indices = np.concatenate(per_mask_indices)
                except Exception:
                    QMessageBox.warning(self, "Warning", "Could not determine original indices for selection.")
                    return
            elif self.current_mask is not None and len(self.current_mask) == len(self.las_data.points):
                # Single-tree case: use the boolean mask for mapping
                global_indices = np.where(self.current_mask)[0]
            else:
                QMessageBox.warning(self, "Warning", "Could not determine original LAS indices for the selection.")
                return
            
            if len(global_indices) == 0:
                QMessageBox.warning(self, "Warning", "No corresponding LAS indices found for the selection.")
                return
            
            # Ensure selected_point_indices are within range
            if np.max(self.selected_point_indices) >= len(global_indices):
                QMessageBox.warning(self, "Warning", "Internal index mapping error: selection out of range.")
                return
            
            # Map to original LAS indices
            original_indices = global_indices[self.selected_point_indices]
            
            # Get ITC values from selected points
            for idx in original_indices:
                if idx < len(itc_array):
                    selected_itc_values.add(int(itc_array[idx]))

            # Show ITC assignment dialog
            dialog = ITCAssignmentDialog(list(selected_itc_values), self)
            result = dialog.exec()

            if result == QDialog.DialogCode.Accepted and dialog.selected_itc is not None:
                new_itc = dialog.selected_itc
            else:
                return
        else:
            # Validate provided ITC value
            if not isinstance(new_itc, int) or new_itc < 1 or new_itc > 999999:
                QMessageBox.warning(self, "Warning", "Invalid ITC value provided.")
                return

        try:
            # Map merged visualization indices back to original LAS indices
            # The merged points order corresponds to concatenating mask_indices for each mask in self.all_masks
            if self.all_masks is not None and len(self.all_masks) > 0:
                try:
                    per_mask_indices = [np.where(mask)[0] for mask in self.all_masks]
                    global_indices = np.concatenate(per_mask_indices)
                except Exception:
                    # Fallback: build iteratively to avoid odd shapes
                    global_list = []
                    for mask in self.all_masks:
                        idxs = np.where(mask)[0]
                        global_list.append(idxs)
                    if len(global_list) > 0:
                        global_indices = np.concatenate(global_list)
                    else:
                        global_indices = np.array([], dtype=int)
            elif self.current_mask is not None and len(self.current_mask) == len(self.las_data.points):
                # Single-tree case: use the boolean mask for mapping
                global_indices = np.where(self.current_mask)[0]
            else:
                self.log_to_console("Warning: Could not determine original indices for selection.")
                QMessageBox.warning(self, "Warning", "Could not determine original LAS indices for the selected points.")
                return

            if len(global_indices) == 0:
                self.log_to_console("Warning: No global indices found for selection.")
                QMessageBox.warning(self, "Warning", "No corresponding LAS indices found for the selection.")
                return

            # Ensure selected_point_indices are within range
            if np.max(self.selected_point_indices) >= len(global_indices):
                self.log_to_console("Error: Selection indices exceed mapped global indices range.")
                QMessageBox.critical(self, "Error", "Internal index mapping error: selection out of range.")
                return

            # Map to original LAS indices and perform update in-memory
            original_indices = global_indices[self.selected_point_indices]
            self.las_data['itc'][original_indices] = new_itc
            if getattr(self, '_source_tree_id_field', None) and self._source_tree_id_field != 'itc':
                if self._source_tree_id_field in self.las_data.point_format.dimension_names:
                    self.las_data[self._source_tree_id_field][original_indices] = new_itc

            # Record the change for session tracking
            from datetime import datetime
            change_record = {
                'timestamp': datetime.now(),
                'original_indices': original_indices,
                'new_itc': new_itc,
                'num_points': len(original_indices)
            }
            self.itc_changes.append(change_record)

            # Also update current visualization ITC values so the view can refresh
            if self.current_itc_values is not None and len(self.current_itc_values) == len(global_indices):
                self.current_itc_values[self.selected_point_indices] = new_itc

            self.log_to_console(f"Assigned ITC value {new_itc} to {len(original_indices)} points (in-memory).")

            # Refresh visualization to reflect changes
            try:
                self.update_tree_colors(self.color_combo.currentText())
            except Exception:
                # If recoloring fails, at least clear selection markers
                pass

            # Clear selection visualization and state
            try:
                if self.plotter:
                    if 'selected_points' in self.plotter.actors:
                        self.plotter.remove_actor('selected_points')
                    if 'selection_polygon' in self.plotter.actors:
                        self.plotter.remove_actor('selection_polygon')
                    self.plotter.update()
            except Exception:
                pass

            self.cancel_polygon_selection()

            QMessageBox.information(self, "Success",
                                    f"Assigned ITC value {new_itc} to {len(original_indices)} points.\n"
                                    "Note: Changes are in memory only. Save LAS to persist changes.")

        except Exception as e:
            self.log_to_console(f"Error assigning ITC value: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to assign ITC value: {str(e)}")


