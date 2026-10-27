"""Polygon/radius/point selection and tree list navigation. (TreeVisualizerGUI mixin)."""

import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QColor
from matplotlib.path import Path as MplPath
from forestry_suite.dialogs.polygon_selection import PolygonSelectionDialog


class SelectionMixin:
    """Polygon/radius/point selection and tree list navigation."""

    def select_all_trees(self):
        """Select all trees in the list (or all visible items if search filter is active)."""
        has_filter = bool(self.tree_search_input.text().strip()) if hasattr(self, 'tree_search_input') else False
        if has_filter:
            for i in range(self.tree_list.count()):
                item = self.tree_list.item(i)
                if not item.isHidden():
                    item.setSelected(True)
        else:
            self.tree_list.selectAll()


    def clear_tree_selection(self):
        """Clear all tree selections."""
        self.tree_list.clearSelection()


    def get_selected_tree_ids(self):
        """Get list of selected tree IDs."""
        selected_items = self.tree_list.selectedItems()
        tree_ids = []
        for item in selected_items:
            try:
                txt = item.text().split()[0]
                tree_ids.append(int(txt))
            except (ValueError, IndexError):
                continue
        return tree_ids


    def next_tree(self):
        """Advance to and immediately visualize the next tree in the list."""
        if self.tree_list.count() == 0:
            return

        selected_rows = [self.tree_list.row(item) for item in self.tree_list.selectedItems()]
        current_row = selected_rows[-1] if selected_rows else -1

        # Find next visible item in tree list
        next_row = current_row
        for _ in range(self.tree_list.count()):
            next_row = (next_row + 1) % self.tree_list.count()
            item = self.tree_list.item(next_row)
            if not item.isHidden():
                self.tree_list.clearSelection()
                item.setSelected(True)
                self.tree_list.setCurrentItem(item)
                self.tree_list.scrollToItem(item)
                self.visualize_tree()
                break


    def prev_tree(self):
        """Move back to and immediately visualize the previous tree in the list."""
        if self.tree_list.count() == 0:
            return

        selected_rows = [self.tree_list.row(item) for item in self.tree_list.selectedItems()]
        current_row = selected_rows[0] if selected_rows else 0

        # Find previous visible item in tree list
        prev_row = current_row
        for _ in range(self.tree_list.count()):
            prev_row = (prev_row - 1 + self.tree_list.count()) % self.tree_list.count()
            item = self.tree_list.item(prev_row)
            if not item.isHidden():
                self.tree_list.clearSelection()
                item.setSelected(True)
                self.tree_list.setCurrentItem(item)
                self.tree_list.scrollToItem(item)
                self.visualize_tree()
                break


    def _on_show_ground_toggled(self, state):
        """Toggle the terrain context overlay without re-rendering the tree."""
        if self.current_tree_points is None:
            return
        if self._ribbon_ground_cb.isChecked() and getattr(self, 'current_ground_points', None) is None:
            # Tree was visualized with context off -- extract ground for the current view
            try:
                itc_values = np.asarray(self.las_data['itc'])
                xs = np.asarray(self.las_data.x)
                ys = np.asarray(self.las_data.y)
                zs = np.asarray(self.las_data.z)
                if self.current_tree_ids is not None and len(self.current_tree_ids) > 0:
                    sel_ids = [int(t) for t in np.unique(self.current_tree_ids)]
                elif self.current_tree_id is not None:
                    sel_ids = [int(self.current_tree_id)]
                else:
                    sel_ids = [0]
                self._extract_footprint_ground_points(self.current_tree_points, itc_values, xs, ys, zs, sel_ids)
            except Exception:
                self.current_ground_points = None
        self._update_terrain_context()
        self.plotter.render()


    def on_draw_polygon_button_clicked(self):
        """Handle draw polygon button click - toggles between start and cancel."""
        if self.polygon_selection_active:
            self.cancel_polygon_selection()
        else:
            self.start_polygon_selection()


    def on_radius_export_button_clicked(self):
        """Toggle radius-based point picking and export mode."""
        if self.radius_selection_active:
            self.cancel_radius_selection()
        else:
            self.start_radius_selection()


    def start_radius_selection(self):
        """Start radius-based point selection mode."""
        if self.las_data is None:
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            QMessageBox.warning(self, "Warning", "No tree points available for selection.")
            return

        if not self.plotter:
            QMessageBox.warning(self, "Warning", "3D viewer not initialized.")
            return

        self.radius_selection_active = True
        self.radius_selection_center = None
        self.radius_export_button.setText("Cancel Radius")
        radius_value = float(self.radius_value_input.value())
        self.log_to_console(
            f"Radius export mode enabled. Click a point in the 3D view to select points within {radius_value:.2f} m and export them."
        )


    def cancel_radius_selection(self, log_message=True):
        """Cancel radius-based point selection mode."""
        self.radius_selection_active = False
        self.radius_selection_center = None
        if hasattr(self, 'radius_export_button'):
            self.radius_export_button.setText("Pick & Export Radius")
        if log_message:
            self.log_to_console("Radius export mode cancelled.")


    def _resolve_picked_point(self, picked_info):
        """Resolve the picked point index and coordinates from a PyVista pick callback payload.
        
        Uses the 3D pick position and finds the closest point in the current tree points.
        This is more reliable than VTK's point_index, which can return a non-closest point
        due to ray-casting through dense point clouds.
        """
        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            return None, None

        # Extract the 3D pick position from the pick callback payload
        picked_point = None
        if hasattr(picked_info, 'picked_point'):
            picked_point = np.asarray(picked_info.picked_point, dtype=float)
        elif isinstance(picked_info, dict) and 'picked_point' in picked_info:
            picked_point = np.asarray(picked_info['picked_point'], dtype=float)
        elif hasattr(picked_info, 'points') and picked_info.points is not None:
            picked_point = np.asarray(picked_info.points[0], dtype=float)
        elif isinstance(picked_info, np.ndarray):
            picked_point = picked_info.ravel()
        else:
            # Last resort: try to use point_index with distance check
            point_index = None
            if hasattr(picked_info, 'point_index'):
                point_index = picked_info.point_index
            elif isinstance(picked_info, dict) and 'point_index' in picked_info:
                point_index = picked_info['point_index']
            if point_index is not None:
                try:
                    point_index = int(point_index)
                    if 0 <= point_index < len(self.current_tree_points):
                        return point_index, self.current_tree_points[point_index]
                except (TypeError, ValueError):
                    pass
            return None, None

        if picked_point is None or picked_point.shape[0] < 3:
            return None, None

        # Find the closest point in the current tree to the pick position
        # This is more accurate than VTK's point_index for dense point clouds
        squared_distances = np.sum((self.current_tree_points[:, :3] - picked_point[:3]) ** 2, axis=1)
        point_index = int(np.argmin(squared_distances))

        if point_index < 0 or point_index >= len(self.current_tree_points):
            return None, None

        return point_index, self.current_tree_points[point_index]


    def _resolve_current_visualization_global_indices(self):
        """Map the current visualization back to original LAS indices."""
        if self.las_data is None:
            return None

        if self.all_masks is not None and len(self.all_masks) > 0:
            try:
                per_mask_indices = [np.where(mask)[0] for mask in self.all_masks]
                if per_mask_indices:
                    return np.concatenate(per_mask_indices)
            except Exception:
                pass

        if self.current_mask is not None:
            try:
                return np.where(self.current_mask)[0]
            except Exception:
                return None

        return None


    def _sanitize_filename_part(self, value):
        """Convert a tree identifier into a filesystem-friendly filename fragment."""
        import re

        text = str(value) if value is not None else "tree"
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
        return text or "tree"


    def _select_points_with_radius(self, picked_point_coords):
        """Select points within the configured radius around the picked point."""
        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            raise ValueError("No tree points available for radius selection")

        radius_value = float(self.radius_value_input.value())
        center_point = np.asarray(picked_point_coords, dtype=float)
        if center_point.shape[0] < 2:
            raise ValueError("Invalid picked point coordinates")

        self.radius_selection_center = center_point[:3]
        xy_offsets = self.current_tree_points[:, :2] - center_point[:2]
        distances = np.sqrt(np.sum(xy_offsets ** 2, axis=1))
        selected_indices = np.where(distances <= radius_value)[0]

        if len(selected_indices) == 0:
            raise ValueError(f"No points found within {radius_value:.2f} m of the picked point")

        self.selected_point_indices = selected_indices
        self.selected_polygon = None
        self.visualize_selected_points()
        self.log_to_console(
            f"Selected {len(selected_indices)} points within {radius_value:.2f} m of ({center_point[0]:.2f}, {center_point[1]:.2f}, {center_point[2]:.2f})."
        )

        self._export_selected_radius_points(radius_value)


    def _export_selected_radius_points(self, radius_value):
        """Export the currently selected radius points to a tree-specific LAS file."""
        if self.las_data is None:
            raise ValueError("No LAS data available for export")

        if self.selected_point_indices is None or len(self.selected_point_indices) == 0:
            raise ValueError("No radius-selected points available for export")

        global_indices = self._resolve_current_visualization_global_indices()
        if global_indices is None or len(global_indices) == 0:
            raise ValueError("Could not determine original LAS indices for the current visualization")

        if np.max(self.selected_point_indices) >= len(global_indices):
            raise ValueError("Selection indices exceed the available LAS index mapping")

        original_indices = global_indices[self.selected_point_indices]
        if len(original_indices) == 0:
            raise ValueError("No original LAS indices resolved for export")

        import copy

        if getattr(self, 'current_las_path', None):
            las_dir = os.path.dirname(self.current_las_path)
            las_basename = os.path.splitext(os.path.basename(self.current_las_path))[0]
        else:
            las_dir = os.getcwd()
            las_basename = "tree_export"

        safe_tree_id = self._sanitize_filename_part(self.current_tree_id if self.current_tree_id is not None else "tree")
        radius_token = f"{radius_value:.2f}".replace('.', 'p')

        export_dir = os.path.join(las_dir, "tree", las_basename, f"tree_{safe_tree_id}")
        os.makedirs(export_dir, exist_ok=True)

        output_path = os.path.join(export_dir, f"tree_{safe_tree_id}_radius_{radius_token}m.las")

        subset_points = self.las_data.points[original_indices].copy()
        header = copy.deepcopy(self.las_data.header)
        export_las = laspy.LasData(header)
        export_las.points = subset_points
        export_las.write(output_path)

        self.log_to_console(
            f"[Save] Exported {len(original_indices)} points to {output_path}"
        )
        self.log_to_console(
            f" Tree folder: {export_dir}"
        )


    def on_point_picked(self, picked_info):
        """Callback for point picking - shows ITC value and trunk ID of picked point."""
        if getattr(self, 'gdb_pick_mode', False):
            self._handle_gdb_pick(picked_info)
            return

        if self.current_tree_points is None or self.current_itc_values is None:
            self.log_to_console("Warning: No tree data available for point picking")
            return

        try:
            point_index, point_coords = self._resolve_picked_point(picked_info)
            if point_index is None or point_coords is None:
                self.log_to_console("Warning: Could not resolve picked point")
                return

            if self.radius_selection_active:
                try:
                    self._select_points_with_radius(point_coords)
                except Exception as radius_error:
                    self.log_to_console(f"Error in radius export selection: {str(radius_error)}")
                    QMessageBox.warning(self, "Radius Selection Error", str(radius_error))
                finally:
                    self.cancel_radius_selection(log_message=False)
                return

            if point_index < len(self.current_itc_values):
                itc_value = self.current_itc_values[point_index]
                message = f"Point picked - ITC: {itc_value}, Coordinates: ({point_coords[0]:.2f}, {point_coords[1]:.2f}, {point_coords[2]:.2f})"

                if hasattr(self, 'trunk_assignment') and self.trunk_assignment is not None:
                    if point_index < len(self.trunk_assignment):
                        trunk_id = self.trunk_assignment[point_index]
                        if trunk_id >= 0:
                            message += f" | Trunk: {trunk_id}"
                        else:
                            message += " | Trunk: NOISE"

                self.log_to_console(message)
            else:
                self.log_to_console("Warning: Point index out of range")

        except Exception as e:
            self.log_to_console(f"Error in point picking: {str(e)}")


    def start_polygon_selection(self):
        """Start polygon selection mode for manual point cloud segmentation."""
        if self.las_data is None:
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            QMessageBox.warning(self, "Warning", "No tree points available for selection.")
            return

        try:
            # Get corresponding ITC values
            if self.current_itc_values is None or len(self.current_itc_values) != len(self.current_tree_points):
                self.log_to_console("Warning: ITC values not available for current visualization")
                QMessageBox.warning(self, "Warning", "ITC values not available for current visualization.")
                return
            
            # Create and show 2D polygon selection dialog
            self.polygon_dialog = PolygonSelectionDialog(self.current_tree_points, self.current_itc_values, self)
            self.polygon_dialog.polygon_completed.connect(self.on_polygon_completed)
            
            # Update button states
            self.draw_polygon_button.setText("Cancel Selection")
            self.draw_polygon_button.clicked.disconnect()
            self.draw_polygon_button.clicked.connect(self.cancel_polygon_selection)
            
            self.polygon_dialog.exec()
            
        except Exception as e:
            self.log_to_console(f"Error starting polygon selection: {str(e)}")


    def on_polygon_completed(self, polygon_2d):
        """Handle completion of 2D polygon selection."""
        try:
            self.selected_polygon = polygon_2d
            self.log_to_console(f"Polygon completed with {len(polygon_2d)} points.")
            
            # Find points inside polygon
            self.find_points_in_polygon()
            
            # Enable assign button
            self.assign_itc_button.setEnabled(True)
            
            # Enable volume calculation button
            self.volume_calc_button.setEnabled(True)
            
            # Visualize selected points
            self.visualize_selected_points()
            
            # Reset button
            self.draw_polygon_button.setText("Draw Selection")
            self.draw_polygon_button.clicked.disconnect()
            self.draw_polygon_button.clicked.connect(self.start_polygon_selection)
            
        except Exception as e:
            self.log_to_console(f"Error processing completed polygon: {str(e)}")


    def _on_dbh_height_changed(self, value):
        """Called when the user changes the DBH reference height via the spinbox."""
        self.dbh_reference_height = float(value)
        self.log_to_console(f"[Measure] DBH reference height set to: {self.dbh_reference_height:.2f}m (re-visualize tree to update marker)")


    def select_all_points(self):
        """Select all points in the current visualization without drawing."""
        if self.las_data is None:
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            QMessageBox.warning(self, "Warning", "No tree points available for selection.")
            return

        try:
            # Select all points in the current visualization
            self.selected_point_indices = np.arange(len(self.current_tree_points))
            self.selected_polygon = None  # No polygon for "select all"
            
            self.log_to_console(f"Selected all {len(self.selected_point_indices)} points in current visualization.")
            
            # Enable assign button
            self.assign_itc_button.setEnabled(True)
            
            # Enable volume calculation button
            self.volume_calc_button.setEnabled(True)
            
            # Visualize selected points (all points will be highlighted)
            self.visualize_selected_points()
            
        except Exception as e:
            self.log_to_console(f"Error selecting all points: {str(e)}")


    def cancel_polygon_selection(self):
        """Cancel polygon selection mode."""
        self.polygon_selection_mode = False
        self.polygon_points = []
        self.selected_polygon = None
        self.selected_point_indices = None
        
        # Close dialog if open
        if hasattr(self, 'polygon_dialog') and self.polygon_dialog.isVisible():
            self.polygon_dialog.reject()
        
        # Clear any polygon visualization
        try:
            self.plotter.remove_actor('selected_points')
            self.plotter.remove_actor('selection_polygon')
            self.plotter.update()
        except:
            pass
        
        # Reset button
        self.draw_polygon_button.setText("Draw Selection")
        self.draw_polygon_button.clicked.disconnect()
        self.draw_polygon_button.clicked.connect(self.start_polygon_selection)
        
        self.assign_itc_button.setEnabled(False)
        self.volume_calc_button.setEnabled(False)
        self.clear_volume_button.setEnabled(False)
        self.log_to_console("Polygon selection cancelled.")


    def on_left_click(self, obj, event):
        """VTK callback for left-click to add polygon point."""
        if not self.polygon_selection_mode:
            return
            
        try:
            # Project click position to Z-plane
            point_3d = self.project_click_to_plane()
            if point_3d is not None:
                self.polygon_points.append(point_3d)
                
                # Visualize the point
                self.visualize_polygon_progress()
                
                self.log_to_console(f"Added point {len(self.polygon_points)}: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})")
            
        except Exception as e:
            self.log_to_console(f"Error adding polygon point: {str(e)}")


    def on_right_click(self, obj, event):
        """VTK callback for right-click to finish polygon."""
        if not self.polygon_selection_mode or len(self.polygon_points) < 3:
            if len(self.polygon_points) < 3:
                self.log_to_console("Warning: Need at least 3 points to create a polygon.")
            return
            
        try:
            self.selected_polygon = np.array(self.polygon_points)
            self.log_to_console(f"Polygon completed with {len(self.polygon_points)} points.")
            
            # Find points inside polygon
            self.find_points_in_polygon()
            
            # Enable assign button
            self.assign_itc_button.setEnabled(True)
            
            # Visualize selected points
            self.visualize_selected_points()
            
            # Exit polygon drawing mode
            self.polygon_selection_mode = False
            self.plotter.iren.remove_observers('LeftButtonPressEvent')
            self.plotter.iren.remove_observers('RightButtonPressEvent')
            
        except Exception as e:
            self.log_to_console(f"Error finishing polygon: {str(e)}")


    def project_click_to_plane(self):
        """Project the current mouse click position to the polygon Z-plane."""
        try:
            # Get mouse position in display coordinates
            click_pos = self.plotter.iren.get_event_position()
            
            # Use PyVista's built-in click position picking
            world_pos = self.plotter.pick_click_position()
            
            if world_pos is not None and not np.allclose(world_pos, [0, 0, 0]):
                # If we got a valid pick from existing geometry, use it
                return np.array(world_pos)
            else:
                # Project to Z-plane using camera ray casting
                camera = self.plotter.camera
                renderer = self.plotter.renderer
                
                # Get display coordinates
                display_point = [click_pos[0], click_pos[1], 0.0]
                
                # Convert display coordinates to world coordinates
                renderer.set_display_point(display_point[0], display_point[1], 0.0)
                renderer.display_to_world()
                world_near = np.array(renderer.get_world_point())
                
                renderer.set_display_point(display_point[0], display_point[1], 1.0)
                renderer.display_to_world()
                world_far = np.array(renderer.get_world_point())
                
                # Calculate ray direction
                ray_direction = world_far - world_near
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                # Find intersection with Z-plane
                if abs(ray_direction[2]) > 1e-6:
                    t = (self.polygon_z_plane - world_near[2]) / ray_direction[2]
                    if t >= 0:  # Only forward intersections
                        intersection = world_near + t * ray_direction
                        return intersection
                
                # Fallback: use camera focal point projected to Z-plane
                focal_point = np.array(camera.focal_point)
                return np.array([focal_point[0], focal_point[1], self.polygon_z_plane])
                
        except Exception as e:
            self.log_to_console(f"Error projecting click to plane: {str(e)}")
            return None


    def find_points_in_polygon(self):
        """Find all points within the drawn polygon."""
        if self.selected_polygon is None or self.current_tree_points is None:
            return
            
        try:
            # Get polygon bounds
            min_x, max_x = np.min(self.selected_polygon[:, 0]), np.max(self.selected_polygon[:, 0])
            min_y, max_y = np.min(self.selected_polygon[:, 1]), np.max(self.selected_polygon[:, 1])
            
            # Get coordinates based on current view mode
            if self.polygon_dialog.view_mode == 'top' or self.polygon_dialog.view_mode == 'bottom':
                # XY coordinates
                points_2d = self.current_tree_points[:, :2]
            elif self.polygon_dialog.view_mode == 'front' or self.polygon_dialog.view_mode == 'back':
                # XZ coordinates
                points_2d = self.current_tree_points[:, [0, 2]]
            elif self.polygon_dialog.view_mode == 'left' or self.polygon_dialog.view_mode == 'right':
                # YZ coordinates
                points_2d = self.current_tree_points[:, [1, 2]]
            else:
                # Default to XY
                points_2d = self.current_tree_points[:, :2]
            
            # Find points within bounding box
            bbox_mask = (
                (points_2d[:, 0] >= min_x) & 
                (points_2d[:, 0] <= max_x) &
                (points_2d[:, 1] >= min_y) & 
                (points_2d[:, 1] <= max_y)
            )
            
            # Get candidate points
            candidate_points = points_2d[bbox_mask]
            
            if len(candidate_points) == 0:
                self.selected_point_indices = np.array([], dtype=int)
                self.log_to_console("No points found within polygon bounding box.")
                return
                
            # Point-in-polygon test using ray casting algorithm
            def point_in_polygon(point, polygon):
                x, y = point[0], point[1]
                n = len(polygon)
                inside = False
                
                p1x, p1y = polygon[0]
                for i in range(1, n + 1):
                    p2x, p2y = polygon[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                    
                return inside
            
            # Test each candidate point
            inside_mask = np.array([point_in_polygon(pt, self.selected_polygon) for pt in candidate_points])
            selected_candidates = np.where(bbox_mask)[0][inside_mask]
            
            self.selected_point_indices = selected_candidates
            self.log_to_console(f"Selected {len(self.selected_point_indices)} points within polygon.")
            
        except Exception as e:
            self.log_to_console(f"Error finding points in polygon: {str(e)}")
            self.selected_point_indices = None


    def _refresh_tree_list_colors(self):
        """Refresh the colors of items in the tree list based on volume data availability."""
        if not hasattr(self, 'unique_itc_values') or self.unique_itc_values is None or len(self.unique_itc_values) == 0:
            return

        for i in range(self.tree_list.count()):
            item = self.tree_list.item(i)
            tree_id = str(item.text())
            if tree_id in self.trees_with_volume_data:
                item.setForeground(QColor('red'))
            else:
                item.setForeground(QColor('white'))


