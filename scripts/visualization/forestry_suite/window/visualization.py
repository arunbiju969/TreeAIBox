"""Tree/trunk/crown rendering and color updates. (TreeVisualizerGUI mixin)."""

import os
import sys
import time
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class VisualizationMixin:
    """Tree/trunk/crown rendering and color updates."""

    def get_point_size(self):
        """Get the current point size from UI (default 2)."""
        if hasattr(self, 'point_size_spin'):
            return self.point_size_spin.value()
        return 2

    def get_render_spheres(self):
        """Get whether to render points as spheres (round) vs squares (default False)."""
        if hasattr(self, 'point_shape_combo'):
            return self.point_shape_combo.currentData() == 'sphere'
        return False

    def _on_point_style_changed(self):
        """Update point size and shape interactively on current actors."""
        pt_size = self.get_point_size()
        as_spheres = self.get_render_spheres()
        if hasattr(self, 'plotter') and self.plotter and hasattr(self.plotter, 'actors'):
            updated = False
            for actor_name in ['tree_point_cloud', 'ground_point_cloud', 'selected_points', 'polygon_points']:
                actor = self.plotter.actors.get(actor_name)
                if actor and hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if actor_name == 'ground_point_cloud':
                        prop.SetPointSize(max(1.0, pt_size * 0.75))
                    else:
                        prop.SetPointSize(float(pt_size))
                    prop.SetRenderPointsAsSpheres(as_spheres)
                    updated = True
            if updated:
                self.plotter.render()

    def _extract_footprint_ground_points(self, merged_points, itc_values, xs, ys, zs, selected_tree_ids):
        """Extract terrain/ground points strictly beneath the selected trees' base.

        Ground criteria (union of any, strictly at/below tree base):
          1. ASPRS classification == 2 (standard ground)
          2. itc == 0 (unassigned ground surface)
          3. treefilter == 1 (TreeAIBox non-tree class at surface level)
          4. semantic_pred == 0 (semantic segmentation ground class)
        Strictly excludes any points belonging to the selected tree(s) and any points
        above the tree base (zs <= tree_min_z + 0.5) to prevent dark ground points from
        contaminating the tree canopy, trunk, or branches.
        Stores (and returns) up to 50,000 subsampled points in self.current_ground_points.
        """
        self.current_ground_points = None
        if merged_points is None or len(merged_points) == 0:
            return None
        if list(selected_tree_ids) == [0]:
            return None

        tree_min_x = np.min(merged_points[:, 0]) - 3.0
        tree_max_x = np.max(merged_points[:, 0]) + 3.0
        tree_min_y = np.min(merged_points[:, 1]) - 3.0
        tree_max_y = np.max(merged_points[:, 1]) + 3.0
        tree_min_z = np.min(merged_points[:, 2])

        # Spatial bounding box mask
        footprint_mask = (xs >= tree_min_x) & (xs <= tree_max_x) & (ys >= tree_min_y) & (ys <= tree_max_y)

        # Ground elevation window: strictly at or below tree base (+0.5m margin for root flare)
        elevation_mask = (zs >= tree_min_z - 3.0) & (zs <= tree_min_z + 0.5)

        dims = list(self.las_data.point_format.dimension_names)
        is_ground = np.zeros(len(xs), dtype=bool)
        if 'classification' in dims:
            is_ground |= (np.asarray(self.las_data.classification) == 2)
        if 'itc' in dims:
            is_ground |= (itc_values == 0)
        if 'treefilter' in dims:
            is_ground |= (np.asarray(self.las_data['treefilter']) == 1)
        if 'semantic_pred' in dims:
            is_ground |= (np.asarray(self.las_data['semantic_pred']) == 0)

        # Explicitly exclude all points that belong to the visualized tree
        is_tree = np.isin(itc_values, list(selected_tree_ids))
        if getattr(self, 'current_mask', None) is not None:
            is_tree = is_tree | self.current_mask

        ground_mask = footprint_mask & elevation_mask & is_ground & (~is_tree)
        if not np.any(ground_mask):
            return None
        g_indices = np.where(ground_mask)[0]
        # Keep performance crisp by subsampling ground points if more than 50,000
        if len(g_indices) > 50000:
            step = max(1, len(g_indices) // 50000)
            g_indices = g_indices[::step]
        self.current_ground_points = np.column_stack((xs[g_indices], ys[g_indices], zs[g_indices]))
        return self.current_ground_points


    def _update_terrain_context(self):
        """Add/remove the terrain context actor from cached ground points.

        Cheap: only touches the ground actor, no tree re-render.
        """
        try:
            self.plotter.remove_actor('ground_point_cloud')
        except Exception:
            pass
        if getattr(self, '_ribbon_ground_cb', None) and self._ribbon_ground_cb.isChecked():
            gp = getattr(self, 'current_ground_points', None)
            if gp is not None and len(gp) > 0:
                g_cloud = pv.PolyData(gp)
                ground_color = '#E53935' if (getattr(self, 'color_combo', None) and self.color_combo.currentText().lower() == 'semantic_pred') else '#8D6E63'
                self.plotter.add_points(
                    g_cloud,
                    color=ground_color,
                    point_size=max(1.0, self.get_point_size() * 0.75),
                    opacity=0.85,
                    render_points_as_spheres=self.get_render_spheres(),
                    reset_camera=False,
                    name='ground_point_cloud'
                )


    def update_tree_colors(self, color_field, preserve_camera=True):
        """Update the coloring of the currently visualized tree."""
        if self.current_tree_points is None or self.current_tree_id is None or not self.plotter or not self.las_data or self.current_mask is None:
            print("Warning: No tree data available for coloring")
            return

        # Guard against an empty field name. This can happen transiently when
        # the color combo is cleared/repopulated while loading a new LAS, which
        # fires currentTextChanged('') while the previous tree is still loaded.
        if not color_field:
            print("Warning: Empty color field - skipping recoloring")
            return

        # Save current camera state so recoloring preserves zoom & camera position
        camera_state = None
        if preserve_camera and self.plotter and hasattr(self.plotter, 'camera') and self.plotter.camera:
            try:
                camera_state = (
                    self.plotter.camera.position,
                    self.plotter.camera.focal_point,
                    self.plotter.camera.up,
                    self.plotter.camera.view_angle,
                    self.plotter.camera.clipping_range
                )
            except Exception:
                camera_state = None

        pt_size = self.get_point_size()
        as_spheres = self.get_render_spheres()

        try:
            if color_field == "Default (green)":
                # Use default green color
                self.plotter.clear()
                self._restore_gdb_layers()
                point_cloud = pv.PolyData(self.current_tree_points)
                self.plotter.add_points(point_cloud, color='#4CAF50', point_size=pt_size, render_points_as_spheres=as_spheres, reset_camera=False, name='tree_point_cloud')
            elif color_field == "RGB (red, green, blue)":
                # Color by RGB values
                print(f"Coloring by RGB values")
                self.plotter.clear()
                self._restore_gdb_layers()
                
                # Extract RGB values from the current tree points
                available_fields = list(self.las_data.point_format.dimension_names)
                if 'red' in available_fields and 'green' in available_fields and 'blue' in available_fields:
                    # Get RGB values for the current tree(s)
                    all_rgb_values = []
                    if self.all_masks is not None:
                        # Multiple trees selected - use all_masks.
                        # Convert the full RGB arrays once, then slice per mask.
                        red_full = np.asarray(self.las_data['red'])
                        green_full = np.asarray(self.las_data['green'])
                        blue_full = np.asarray(self.las_data['blue'])
                        for mask in self.all_masks:
                            red_vals = red_full[mask]
                            green_vals = green_full[mask]
                            blue_vals = blue_full[mask]
                            
                            # Normalize RGB to 0-1 range if they're in 0-65535 range (uint16)
                            if len(red_vals) > 0 and np.max(red_vals) > 1:
                                red_vals = red_vals / 65535.0
                                green_vals = green_vals / 65535.0
                                blue_vals = blue_vals / 65535.0
                            
                            rgb_colors = np.column_stack([red_vals, green_vals, blue_vals])
                            all_rgb_values.append(rgb_colors)
                        
                        rgb_colors = np.vstack(all_rgb_values)
                    else:
                        # Single tree - use current_mask
                        red_vals = np.array(self.las_data['red'])[self.current_mask]
                        green_vals = np.array(self.las_data['green'])[self.current_mask]
                        blue_vals = np.array(self.las_data['blue'])[self.current_mask]
                        
                        # Normalize RGB to 0-1 range if they're in 0-65535 range (uint16)
                        if len(red_vals) > 0 and np.max(red_vals) > 1:
                            red_vals = red_vals / 65535.0
                            green_vals = green_vals / 65535.0
                            blue_vals = blue_vals / 65535.0
                        
                        rgb_colors = np.column_stack([red_vals, green_vals, blue_vals])
                    
                    # Create point cloud and color by RGB
                    point_cloud = pv.PolyData(self.current_tree_points)
                    point_cloud['rgb'] = rgb_colors
                    
                    # Use direct RGB coloring (rgb=True uses the 'rgb' scalar field)
                    self.plotter.add_points(point_cloud, scalars='rgb', rgb=True, point_size=pt_size, render_points_as_spheres=as_spheres, reset_camera=False, name='tree_point_cloud')
                    print(f"RGB coloring applied to {len(self.current_tree_points)} points")
                else:
                    print(f"Warning: RGB fields not found. Available fields: {available_fields}")
                    QMessageBox.warning(self, "Warning", "RGB fields (red, green, blue) not found in point cloud")
                    return
            else:
                # Color by the selected scalar field
                try:
                    # Check if we need to recalculate color values
                    if self.current_color_field != color_field or self.current_color_values is None:
                        if self.all_masks is not None:
                            color_full = np.asarray(self.las_data[color_field])
                            all_color_values = [color_full[mask] for mask in self.all_masks]
                            self.current_color_values = np.concatenate(all_color_values)
                            self.current_color_field = color_field
                        else:
                            # Fallback
                            self.current_color_values = np.array(self.las_data[color_field])[self.current_mask]

                    color_values = self.current_color_values
                    
                    # Special handling for ITC field - map to 9 cycling colors
                    is_itc_or_tree_field = (color_field == 'itc' or color_field == getattr(self, '_source_tree_id_field', None))
                    if is_itc_or_tree_field:
                        # Map ITC values to color indices 0-8 (9 colors total)
                        color_indices = (color_values - 1) % 9  # Subtract 1 since ITC starts from 1, then modulo 9
                        color_values = color_indices
                        print(f"Coloring by ITC with 9 cycling colors, {len(color_values)} values")
                    else:
                        print(f"Coloring by field '{color_field}' with {len(color_values)} values, range: {np.min(color_values):.3f} - {np.max(color_values):.3f}")

                    self.plotter.clear()
                    self._restore_gdb_layers()
                    point_cloud = pv.PolyData(self.current_tree_points)
                    
                    if is_itc_or_tree_field:
                        # Use a colormap with 9 distinct colors for ITC
                        self.plotter.add_points(point_cloud, scalars=color_values, point_size=pt_size, render_points_as_spheres=as_spheres, 
                                              cmap='tab10', n_colors=9, clim=[0, 8], reset_camera=False, name='tree_point_cloud')
                    elif color_field.lower() == 'semantic_pred':
                        # Semantic segmentation color mapping:
                        # 0: Ground  -> Red   ([229, 57, 53] / #E53935)
                        # 1: Trunk   -> Blue  ([30, 136, 229] / #1E88E5)
                        # 2: Foliage -> Green ([67, 160, 71] / #43A047)
                        # Other      -> Gray  ([158, 158, 158] / #9E9E9E)
                        sem_vals = np.asarray(color_values, dtype=np.int32)
                        rgb_colors = np.full((len(sem_vals), 3), 158, dtype=np.uint8)
                        rgb_colors[sem_vals == 0] = [229, 57, 53]   # Red: Ground
                        rgb_colors[sem_vals == 1] = [30, 136, 229]  # Blue: Trunk
                        rgb_colors[sem_vals == 2] = [67, 160, 71]   # Green: Foliage
                        point_cloud['rgb'] = rgb_colors
                        self.plotter.add_points(point_cloud, scalars='rgb', rgb=True, point_size=pt_size, render_points_as_spheres=as_spheres, reset_camera=False, name='tree_point_cloud')
                    else:
                        self.plotter.add_points(point_cloud, scalars=color_values, point_size=pt_size, render_points_as_spheres=as_spheres, cmap='cividis', reset_camera=False, name='tree_point_cloud')
                        
                except Exception as e:
                    print(f"Error coloring by field '{color_field}': {str(e)}")
                    QMessageBox.warning(self, "Warning", f"Failed to color by field '{color_field}': {str(e)}")
                    return

            # Render ground points if enabled
            self._update_terrain_context()

            # Apply current pick mode (tree vs GDB) to the freshly added actors
            self._apply_pick_mode()

            # Restore camera position and zoom level if preserving view, otherwise reset to frame tree
            if preserve_camera and camera_state is not None:
                try:
                    self.plotter.camera.position = camera_state[0]
                    self.plotter.camera.focal_point = camera_state[1]
                    self.plotter.camera.up = camera_state[2]
                    self.plotter.camera.view_angle = camera_state[3]
                    self.plotter.camera.clipping_range = camera_state[4]
                except Exception:
                    self.plotter.reset_camera()
            else:
                self.plotter.reset_camera()

            num_points = len(self.current_tree_points)
            try:
                self.plotter.remove_actor('tree_info_text')
                self.plotter.remove_actor('itc_instruction_text')
            except Exception:
                pass

            if color_field.lower() == 'semantic_pred':
                color_legend_text = f"Color: semantic_pred [Foliage: Green | Trunk: Blue | Ground: Red]"
            else:
                color_legend_text = f"Color: {color_field}"

            self.plotter.add_text(
                f"Click on points to see ITC values\nTree ID: {self.current_tree_id} ({num_points:,} points) - {color_legend_text}",
                position='lower_left',
                font_size=11,
                color='#FFFFFF',
                name='tree_info_text'
            )
            # Print trunk/branch summary table to GUI console and python console
            try:
                # Prefer trunk_volume_summary if present
                if hasattr(self, 'trunk_volume_summary') and self.trunk_volume_summary:
                    overall_total = sum(t['total_volume'] for t in self.trunk_volume_summary.values())
                    total_trunks = len(self.trunk_volume_summary)
                    header = f"{'Trunk':<8} {'Volume_m3':>12} {'#Branches':>10} {'#Points':>10}"
                    self.log_to_console('\n' + header)
                    print(header)
                    self.log_to_console('-' * len(header))
                    print('-' * len(header))
                    for tid in sorted(self.trunk_volume_summary.keys()):
                        t = self.trunk_volume_summary[tid]
                        row = f"{tid:<8} {t['total_volume']:12.4f} {t['num_branches']:10d} {t['total_points']:10d}"
                        self.log_to_console(row)
                        print(row)
                    footer = f"Total trunks: {total_trunks} | Overall volume: {overall_total:.4f} m³"
                    self.log_to_console(footer)
                    print(footer)
                else:
                    # Aggregate from accumulated_volume_sections if trunk_summary missing
                    trunk_agg = {}
                    for sec in self.accumulated_volume_sections:
                        tid = sec.get('trunk_id', None)
                        if tid is None:
                            tid = -1
                        rec = trunk_agg.setdefault(tid, {'total_volume': 0.0, 'num_branches': 0, 'total_points': 0})
                        rec['total_volume'] += float(sec.get('volume', 0.0) or 0.0)
                        rec['num_branches'] += 1
                        rec['total_points'] += len(sec.get('points', []))

                    header = f"{'Trunk':<8} {'Volume_m3':>12} {'#Sections':>10} {'#Points':>10}"
                    self.log_to_console('\n' + header)
                    print(header)
                    self.log_to_console('-' * len(header))
                    print('-' * len(header))
                    for tid in sorted(trunk_agg.keys()):
                        rec = trunk_agg[tid]
                        row = f"{tid:<8} {rec['total_volume']:12.4f} {rec['num_branches']:10d} {rec['total_points']:10d}"
                        self.log_to_console(row)
                        print(row)
                    overall = sum(r['total_volume'] for r in trunk_agg.values())
                    footer = f"Total trunks (keys): {len(trunk_agg)} | Overall volume: {overall:.4f} m³"
                    self.log_to_console(footer)
                    print(footer)
            except Exception as e:
                self.log_to_console(f"[WARN] Error building trunk summary table: {e}")
                print(f"Error building trunk summary table: {e}")

            self.plotter.update()
        except Exception as e:
            print(f"Error in update_tree_colors: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to update tree colors: {str(e)}")


    def visualize_tree(self):
        """Visualize the selected trees using PyVista."""
        if self.las_data is None:
            print("Error: No LAS file loaded")
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        if not self.plotter:
            print("Error: 3D viewer not initialized")
            QMessageBox.warning(self, "Warning", "3D viewer not initialized.")
            return

        selected_tree_ids = self.get_selected_tree_ids()
        if not selected_tree_ids:
            print("Warning: No tree IDs selected")
            QMessageBox.warning(self, "Warning", "Please select one or more tree IDs from the list.")
            return

        try:
            # Validate selected tree IDs
            invalid_ids = [tid for tid in selected_tree_ids if tid not in self.unique_itc_values and tid != 0]
            if invalid_ids:
                print(f"Warning: Invalid tree IDs selected: {invalid_ids}")
                QMessageBox.warning(self, "Warning", f"Invalid tree IDs selected: {invalid_ids}")
                return

        except ValueError as e:
            print(f"Error validating tree IDs: {str(e)}")
            QMessageBox.warning(self, "Warning", "Invalid tree IDs selected.")
            return

        # Start loading feedback (matching LAS file loading style)
        import time
        viz_start = time.perf_counter()
        num_trees = len(selected_tree_ids)
        self.log_to_console(f"[Viz] Preparing 3D visualization for {num_trees:,} selected tree(s)...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._set_progress(5, f"Visualizing {num_trees:,} tree(s)...")

        try:
            # Clear accumulated volume results when visualizing new trees
            self.clear_accumulated_volume()
            # Reset any prior selection/analysis state to avoid index mismatches
            try:
                self.selected_point_indices = None
            except Exception:
                pass
            try:
                self.radius_selection_active = False
                self.radius_selection_center = None
                if hasattr(self, 'radius_export_button'):
                    self.radius_export_button.setText("Pick & Export Radius")
            except Exception:
                pass
            try:
                self.branch_assignment = None
            except Exception:
                pass
            try:
                self.trunk_assignment = None
            except Exception:
                pass
            try:
                self.last_analyzed_points = None
            except Exception:
                pass

            self._set_progress(15, f"Filtering points ({num_trees:,} trees)...")

            # Extract coordinates and itc values once up-front
            itc_values = np.asarray(self.las_data['itc'])
            xs = np.asarray(self.las_data.x)
            ys = np.asarray(self.las_data.y)
            zs = np.asarray(self.las_data.z)

            treefilter_keep = self.get_tree_point_mask()

            # Fast unified vectorized path for all tree selections (1 to thousands of trees)
            pos_tree_ids = [tid for tid in selected_tree_ids if tid != 0]
            include_ground = (0 in selected_tree_ids)

            combined_mask = np.zeros(len(xs), dtype=bool)
            if pos_tree_ids:
                if len(pos_tree_ids) == 1:
                    pos_mask = (itc_values == pos_tree_ids[0])
                elif (hasattr(self, 'unique_itc_values') and self.unique_itc_values is not None
                        and len(self.unique_itc_values) > 0
                        and set(pos_tree_ids) >= set(self.unique_itc_values)):
                    pos_mask = (itc_values > 0)
                else:
                    pos_mask = np.isin(itc_values, pos_tree_ids)

                if treefilter_keep is not None:
                    pos_mask = pos_mask & treefilter_keep
                combined_mask |= pos_mask

            if include_ground:
                g_mask = np.zeros(len(xs), dtype=bool)
                if 'itc' in self.las_data.point_format.dimension_names:
                    g_mask |= (itc_values == 0)
                if 'classification' in self.las_data.point_format.dimension_names:
                    g_mask |= (np.asarray(self.las_data.classification) == 2)
                combined_mask |= g_mask

            if not np.any(combined_mask):
                extra = ""
                if treefilter_keep is not None:
                    extra = (f"\n\nTree-point filter ({self.tree_filter_field} == {self.tree_filter_value}) "
                             f"matches {int(treefilter_keep.sum())} points in the whole file.\n"
                             f"Check the filter Value on the View ribbon (e.g. semantic_pred: 0=ground, 1=trunk, 2=foliage).")
                print(f"Warning: No points found for selected tree IDs: {selected_tree_ids}")
                QMessageBox.warning(self, "Warning", f"No points found for selected tree IDs: {selected_tree_ids}{extra}")
                return

            merged_points = np.column_stack((xs[combined_mask], ys[combined_mask], zs[combined_mask]))
            merged_mask = combined_mask
            self.current_tree_ids = itc_values[combined_mask]
            self.all_masks = None

            self._set_progress(45, f"Extracted {len(merged_points):,} points. Preparing colors...")

            # If Show Terrain Context is enabled, extract ground points under the selected trees' XY footprint
            self.current_ground_points = None
            if getattr(self, '_ribbon_ground_cb', None) and self._ribbon_ground_cb.isChecked():
                self._extract_footprint_ground_points(merged_points, itc_values, xs, ys, zs, selected_tree_ids)

            print(f"Visualizing {len(selected_tree_ids)} trees with {len(merged_points)} total points")

            # Calculate bounding box and log dimensions
            if len(merged_points) > 0:
                bounds = [
                    np.min(merged_points[:, 0]), np.max(merged_points[:, 0]),
                    np.min(merged_points[:, 1]), np.max(merged_points[:, 1]),
                    np.min(merged_points[:, 2]), np.max(merged_points[:, 2])
                ]
                bottom_z = bounds[4]
                bounding_box_height = bounds[5] - bounds[4]

                tree_z_values = merged_points[:, 2]
                ground_level = np.min(tree_z_values)
                tree_height = bounds[5] - ground_level
                self.current_tree_height = float(tree_height)
                self.current_tree_ground_z = float(ground_level)

                print(f"[Box] Bounding Box - Bottom Z: {bottom_z:.2f}m, Bounding Box Height: {bounding_box_height:.2f}m")
                print(f"[Tree] Tree Height (from ground): {tree_height:.2f}m (ground level: {ground_level:.2f}m)")
                print(f"[Dim] Dimensions: {bounds[1]-bounds[0]:.2f}m W × {bounds[3]-bounds[2]:.2f}m L × {bounding_box_height:.2f}m H")

            # Store current tree data for recoloring
            self.current_tree_points = merged_points
            self.current_tree_id = selected_tree_ids[0] if selected_tree_ids else None
            self.current_mask = merged_mask
            self.all_masks = None

            # Collect color values for current visualization
            color_field = self.color_combo.currentText()
            if color_field == "RGB (red, green, blue)":
                self.current_color_values = None
                self.current_color_field = color_field
            elif color_field != "Default (green)":
                color_full = np.asarray(self.las_data[color_field])
                self.current_color_values = color_full[merged_mask]
                self.current_color_field = color_field
            else:
                self.current_color_values = None
                self.current_color_field = color_field

            # Collect ITC values for point picking
            self.current_itc_values = itc_values[merged_mask]

            self._set_progress(70, f"Rendering 3D scene ({len(merged_points):,} points)...")

            # Visualize with current color field (reset camera to frame tree)
            self.update_tree_colors(color_field, preserve_camera=False)

            self._set_progress(90, "Finalizing 3D overlay...")

            # Add bounding box visualization
            if len(merged_points) > 0:
                bounds = [
                    np.min(merged_points[:, 0]), np.max(merged_points[:, 0]),
                    np.min(merged_points[:, 1]), np.max(merged_points[:, 1]),
                    np.min(merged_points[:, 2]), np.max(merged_points[:, 2])
                ]
                bounding_box = pv.Cube(bounds=bounds)
                self.plotter.add_mesh(bounding_box, color='black', style='wireframe', line_width=2,
                                    name='tree_bounding_box', label='Tree Bounding Box')
                self._set_actor_pickable('tree_bounding_box', False)
                try:
                    bottom_z = bounds[4]
                    top_z = bounds[5]
                    bounding_box_height = top_z - bottom_z
                    self.plotter.add_text(f"BBox H: {bounding_box_height:.2f} m", position='upper_right', font_size=12, color='#FFFFFF')
                    top_center = ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, top_z)
                    try:
                        self.plotter.add_point_labels(np.array([top_center]), [f"{bounding_box_height:.2f} m"], point_size=0, font_size=10, text_color='yellow', always_visible=True)
                    except Exception:
                        self.plotter.add_point_labels([top_center], [f"{bounding_box_height:.2f} m"], point_size=0, font_size=10, text_color='yellow', always_visible=True)
                except Exception as e:
                    print(f"Warning: Failed to add bounding box height label: {e}")

                # Draw DBH marker only for single tree or small group
                if len(selected_tree_ids) <= 3:
                    try:
                        dbh_ref_h = getattr(self, 'dbh_reference_height', 1.3)
                        dbh_z = ground_level + dbh_ref_h
                        mid_y = (bounds[2] + bounds[3]) / 2.0
                        dbh_line_start = (bounds[0], mid_y, dbh_z)
                        dbh_line_end = (bounds[1], mid_y, dbh_z)
                        dbh_line = pv.Line(dbh_line_start, dbh_line_end)
                        self.plotter.add_mesh(dbh_line, color='orange', line_width=3,
                                            name='dbh_marker_line', label=f'DBH @ {dbh_ref_h:.1f}m')
                        self._set_actor_pickable('dbh_marker_line', False)
                        dbh_mid = ((bounds[0] + bounds[1]) / 2.0, mid_y, dbh_z)
                        self.plotter.add_point_labels([dbh_mid], [f"DBH ({dbh_ref_h:.1f}m)"],
                                                    point_size=0, font_size=9, text_color='orange',
                                                    always_visible=True,
                                                    name='dbh_marker_label')
                    except Exception as e:
                        print(f"Warning: Failed to add DBH marker: {e}")

            # Enable analyze button for tree analysis
            self.analyze_button.setEnabled(True)

            # Enable polygon selection buttons
            self.draw_polygon_button.setEnabled(True)
            self.select_all_points_button.setEnabled(True)
            self.radius_export_button.setEnabled(True)

            # Check volume folder for existing volume data if enabled (after visualization is complete)
            if (hasattr(self, 'volume_folder_checkbox') and 
                hasattr(self, 'volume_folder_path') and
                self.volume_folder_checkbox.isChecked() and 
                self.volume_folder_path and 
                isinstance(self.volume_folder_path, str)):
                self._refresh_tree_list_colors()

            self._set_progress(100, "Done")
            total_time = time.perf_counter() - viz_start
            self.log_to_console(f"[OK] Visualized {num_trees:,} tree(s) ({len(merged_points):,} points) in {total_time:.2f}s")

        except Exception as e:
            print(f"Error in visualize_tree: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to visualize trees: {str(e)}")
        finally:
            self._hide_progress()
            while QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()


    def visualize_trunk(self):
        """Visualize the trunks of the selected trees using PyVista (stemcls != 1)."""
        if self.las_data is None:
            print("Error: No LAS file loaded")
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        if not self.plotter:
            print("Error: 3D viewer not initialized")
            QMessageBox.warning(self, "Warning", "3D viewer not initialized.")
            return

        selected_tree_ids = self.get_selected_tree_ids()
        if not selected_tree_ids:
            print("Warning: No tree IDs selected")
            QMessageBox.warning(self, "Warning", "Please select one or more tree IDs from the list.")
            return

        try:
            # Validate selected tree IDs
            invalid_ids = [tid for tid in selected_tree_ids if tid not in self.unique_itc_values and tid != 0]
            if invalid_ids:
                print(f"Warning: Invalid tree IDs selected: {invalid_ids}")
                QMessageBox.warning(self, "Warning", f"Invalid tree IDs selected: {invalid_ids}")
                return

        except ValueError as e:
            print(f"Error validating tree IDs: {str(e)}")
            QMessageBox.warning(self, "Warning", "Invalid tree IDs selected.")
            return

        # Check if stemcls field exists
        if 'stemcls' not in self.las_data.point_format.dimension_names:
            print("Warning: No 'stemcls' scalar field found")
            QMessageBox.warning(self, "Warning", "No 'stemcls' scalar field found in the LAS file.")
            return

        import time
        viz_start = time.perf_counter()
        num_trees = len(selected_tree_ids)
        self.log_to_console(f"[Viz] Visualizing trunks for {num_trees:,} selected tree(s)...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._set_progress(5, f"Visualizing {num_trees:,} trunk(s)...")

        try:
            self._set_progress(20, f"Extracting trunk points ({num_trees:,} trees)...")

            # Extract coordinates and scalar fields once up-front
            itc_values = np.asarray(self.las_data['itc'])
            stemcls_values = np.asarray(self.las_data['stemcls'])
            xs = np.asarray(self.las_data.x)
            ys = np.asarray(self.las_data.y)
            zs = np.asarray(self.las_data.z)

            tree_mask = self.get_tree_point_mask()
            stem_cond = (stemcls_values != 1)
            if tree_mask is not None:
                stem_cond = stem_cond & tree_mask

            # Fast unified vectorized trunk extraction for all tree selections (1 to thousands)
            if len(selected_tree_ids) == 1:
                combined_mask = (itc_values == selected_tree_ids[0]) & stem_cond
            elif (hasattr(self, 'unique_itc_values') and self.unique_itc_values is not None
                    and len(self.unique_itc_values) > 0
                    and set(selected_tree_ids) >= set(self.unique_itc_values)):
                combined_mask = (itc_values > 0) & stem_cond
            else:
                combined_mask = np.isin(itc_values, selected_tree_ids) & stem_cond

            if not np.any(combined_mask):
                print(f"Warning: No trunk points found for selected tree IDs: {selected_tree_ids}")
                QMessageBox.warning(self, "Warning", f"No trunk points found for selected tree IDs: {selected_tree_ids}")
                return

            merged_points = np.column_stack((xs[combined_mask], ys[combined_mask], zs[combined_mask]))
            merged_mask = combined_mask
            self.current_itc_values = itc_values[combined_mask]
            all_masks = None

            print(f"Visualizing trunks of {len(selected_tree_ids)} trees with {len(merged_points)} total points")

            # Calculate bounding box height and set max height input (use the tallest tree)
            z_coords = merged_points[:, 2]
            trunk_height = np.max(z_coords) - np.min(z_coords)
            self.max_height_input.setText(f"{trunk_height:.2f}")

            # Store current tree data for recoloring
            self.current_tree_points = merged_points
            self.current_tree_id = f"Multiple Trunks ({', '.join(map(str, selected_tree_ids[:5]))}{'...' if len(selected_tree_ids) > 5 else ''})"
            self.current_mask = merged_mask
            self.all_masks = None

            # Reset any prior selection/analysis state
            try:
                self.selected_point_indices = None
                self.radius_selection_active = False
                self.radius_selection_center = None
                if hasattr(self, 'radius_export_button'):
                    self.radius_export_button.setText("Pick & Export Radius")
                self.branch_assignment = None
                self.trunk_assignment = None
                self.last_analyzed_points = None
            except Exception:
                pass

            # For trunk visualization, use default green
            self.current_color_values = None
            self.current_color_field = "Default (green)"

            self._set_progress(70, f"Rendering trunk scene ({len(merged_points):,} points)...")

            # Switch to default green for trunk visualization
            self.color_combo.setCurrentText("Default (green)")
            self.update_tree_colors(self.color_combo.currentText(), preserve_camera=False)

            self._set_progress(90, "Enabling trunk analysis tools...")

            # Enable split detection button for merged trunks
            self.analyze_button.setEnabled(True)
            self.detect_branches_button.setEnabled(True)
            self.detect_trunks_button.setEnabled(True)

            # Enable polygon selection buttons
            self.draw_polygon_button.setEnabled(True)
            self.select_all_points_button.setEnabled(True)
            self.radius_export_button.setEnabled(True)

            self._set_progress(100, "Done")
            total_time = time.perf_counter() - viz_start
            self.log_to_console(f"[OK] Visualized {num_trees:,} trunk(s) ({len(merged_points):,} points) in {total_time:.2f}s")

        except Exception as e:
            print(f"Error in visualize_trunk: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to visualize trunks: {str(e)}")
        finally:
            self._hide_progress()
            while QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()




    def create_tree_crown_polygon(self):
        """Create a 2D polygon from all tree points projected to XY plane."""
        if self.las_data is None:
            print("Error: No LAS file loaded")
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        if not self.plotter:
            print("Error: 3D viewer not initialized")
            QMessageBox.warning(self, "Warning", "3D viewer not initialized.")
            return

        selected_tree_ids = self.get_selected_tree_ids()
        if not selected_tree_ids:
            print("Warning: No tree IDs selected")
            QMessageBox.warning(self, "Warning", "Please select one or more tree IDs from the list.")
            return

        try:
            # Validate selected tree IDs
            invalid_ids = [tid for tid in selected_tree_ids if tid not in self.unique_itc_values and tid != 0]
            if invalid_ids:
                print(f"Warning: Invalid tree IDs selected: {invalid_ids}")
                QMessageBox.warning(self, "Warning", f"Invalid tree IDs selected: {invalid_ids}")
                return

            # Extract and merge ALL points for all selected trees (no height filtering)
            all_points = []
            all_masks = []
            itc_values = np.array(self.las_data['itc'])
            tree_mask = self.get_tree_point_mask()

            for tree_id in selected_tree_ids:
                mask = (itc_values == tree_id)

                # Apply tree-point filter if enabled
                if tree_mask is not None:
                    mask = mask & tree_mask

                if np.any(mask):
                    points = np.column_stack([
                        np.array(self.las_data.x)[mask],
                        np.array(self.las_data.y)[mask],
                        np.array(self.las_data.z)[mask]
                    ])
                    all_points.append(points)
                    all_masks.append(mask)

            if not all_points:
                extra = ""
                if tree_mask is not None:
                    extra = (f"\n\nTree-point filter ({self.tree_filter_field} == {self.tree_filter_value}) "
                             f"matches {int(tree_mask.sum())} points in the whole file.\n"
                             f"Check the filter Value on the View ribbon (e.g. semantic_pred: 0=ground, 1=trunk, 2=foliage).")
                print(f"Warning: No points found for selected tree IDs: {selected_tree_ids}")
                QMessageBox.warning(self, "Warning", f"No points found for selected tree IDs: {selected_tree_ids}{extra}")
                return

            # Merge all points
            merged_points = np.vstack(all_points)
            merged_mask = np.concatenate(all_masks)

            print(f"Creating crown polygon from {len(merged_points)} total points for trees: {selected_tree_ids}")

            # Project points to 2D (XY plane) - remove Z coordinate
            points_2d = merged_points[:, :2]  # Keep only X, Y coordinates

            # Debug: Check for valid points
            print(f"2D points shape: {points_2d.shape}")
            print(f"2D points range X: {np.min(points_2d[:, 0]):.2f} to {np.max(points_2d[:, 0]):.2f}")
            print(f"2D points range Y: {np.min(points_2d[:, 1]):.2f} to {np.max(points_2d[:, 1]):.2f}")

            # Create convex hull polygon from 2D points
            if len(points_2d) < 3:
                QMessageBox.warning(self, "Warning", "Need at least 3 points to create a polygon.")
                return

            # Check if points span enough area (not collinear)
            if len(points_2d) >= 3:
                # Calculate the area of the convex hull to check if points are collinear
                try:
                    from scipy.spatial import ConvexHull
                    temp_hull = ConvexHull(points_2d)
                    hull_area = temp_hull.volume  # For 2D, volume is actually area
                    print(f"Hull area: {hull_area}")
                    if hull_area < 1e-10:  # Very small area indicates collinear points
                        QMessageBox.warning(self, "Warning", "Points appear to be collinear or nearly collinear. Cannot create a meaningful polygon.")
                        return
                except Exception as area_check_error:
                    print(f"Could not compute hull area: {area_check_error}")
                    # Continue anyway, might still work

            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(points_2d)
                hull_points_2d = points_2d[hull.vertices]

                print(f"Hull computed successfully with {len(hull_points_2d)} vertices")

                # Create closed polygon (repeat first point at end)
                polygon_points = np.vstack([hull_points_2d, hull_points_2d[0]])

                # Create PyVista polygon mesh - need 3D points, so add Z=0 (or use min Z from original points)
                import pyvista as pv
                # Use minimum Z value from original points for the polygon plane
                z_value = np.min(merged_points[:, 2])
                hull_points_3d = np.column_stack([hull_points_2d, np.full(len(hull_points_2d), z_value)])
                
                # Create polygon from hull points
                polygon_mesh = pv.PolyData(hull_points_3d)
                # Add the polygon face (single face connecting all vertices)
                faces = np.array([len(hull_points_3d)] + list(range(len(hull_points_3d))))
                polygon_mesh.faces = faces

                # Clear current view and show the polygon
                self.plotter.clear()
                self._restore_gdb_layers()

                # Add the polygon as a filled mesh
                self.plotter.add_mesh(polygon_mesh, color='green', opacity=0.3, show_edges=True, edge_color='darkgreen', line_width=2)

                # Add the original 3D points as reference
                point_cloud = pv.PolyData(merged_points)
                self.plotter.add_points(point_cloud, color='blue', point_size=2, opacity=0.6)

                # Add text info
                info_text = f"Tree Crown Polygon\nTrees: {', '.join(map(str, selected_tree_ids))}\nPoints: {len(merged_points)}\nHull vertices: {len(hull_points_2d)}"
                self.plotter.add_text(info_text, position='upper_left', font_size=10, color='white')

                self.plotter.reset_camera()

                # Store data for potential export
                self.current_tree_points = merged_points
                self.current_tree_id = f"Crown Polygon ({', '.join(map(str, selected_tree_ids))})"
                self.current_mask = merged_mask
                self.current_hull_points_2d = hull_points_2d
                self.current_polygon_tree_ids = selected_tree_ids

                QMessageBox.information(self, "Success", f"Created crown polygon from {len(merged_points)} points with {len(hull_points_2d)} vertices.")

            except Exception as hull_error:
                print(f"Error creating convex hull: {hull_error}")
                QMessageBox.warning(self, "Warning", f"Could not create polygon from points: {hull_error}")

        except Exception as e:
            print(f"Error in create_tree_crown_polygon: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to create tree crown polygon: {str(e)}")


    def refresh_color_combo_box(self):
        """Refresh the color combo box to include newly added scalar fields like DBH points."""
        if self.las_data is None:
            return

        try:
            # Get current selection
            current_selection = self.color_combo.currentText()

            # Temporarily disconnect the signal to avoid triggering update_tree_colors with empty field
            self.color_combo.currentTextChanged.disconnect(self.update_tree_colors)

            # Populate color field combo box with updated fields
            available_fields = list(self.las_data.point_format.dimension_names)
            color_fields = [field for field in available_fields
                           if field not in ['x', 'y', 'z']]  # Include itc for multiple tree coloring

            self.color_combo.clear()
            self.color_combo.addItem("Default (green)")
            for field in sorted(color_fields):
                self.color_combo.addItem(field)

            # Restore previous selection if it still exists
            if current_selection and current_selection != "Default (green)":
                index = self.color_combo.findText(current_selection)
                if index >= 0:
                    self.color_combo.setCurrentIndex(index)
                else:
                    # If previous selection no longer exists, select DBH points if available
                    dbh_index = self.color_combo.findText("dbh_points")
                    if dbh_index >= 0:
                        self.color_combo.setCurrentIndex(dbh_index)
                        self.log_to_console("Color Switched to 'dbh_points' coloring")

            # Reconnect the signal
            self.color_combo.currentTextChanged.connect(self.update_tree_colors)

            self.log_to_console("[OK] Color options updated - 'dbh_points' now available for coloring")

        except Exception as e:
            # Make sure to reconnect the signal even if there's an error
            try:
                self.color_combo.currentTextChanged.connect(self.update_tree_colors)
            except:
                pass
            self.log_to_console(f"[WARN] Could not refresh color combo box: {str(e)}")


    def view_neighbors(self):
        """Find and visualize neighboring trees within the specified radius."""
        if self.las_data is None:
            print("Error: No LAS file loaded")
            QMessageBox.warning(self, "Warning", "No LAS file loaded.")
            return

        selected_tree_ids = self.get_selected_tree_ids()
        if not selected_tree_ids:
            print("Warning: No trees selected for neighbor search")
            QMessageBox.warning(self, "Warning", "Please select one or more trees first.")
            return

        try:
            radius = float(self.neighbor_radius_input.text())
            if radius <= 0 or radius > 1000:
                raise ValueError("Radius must be between 0.1 and 1000 meters")
        except ValueError as e:
            print(f"Error: Invalid radius value: {str(e)}")
            QMessageBox.warning(self, "Invalid Radius", f"Please enter a valid radius.\nError: {e}")
            return

        try:
            # Ensure centroids are pre-computed (lazy loading)
            if not self.tree_centroids_cache:
                print("Pre-computing tree centroids for neighbor search...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                self.precompute_tree_centroids()
                self.progress_bar.setVisible(False)
                print("Centroid pre-computation completed")

            # Calculate centroid of selected trees
            reference_centroid = self.calculate_trees_centroid(selected_tree_ids)
            if reference_centroid is None:
                print("Error: Could not calculate centroid of selected trees")
                QMessageBox.warning(self, "Error", "Could not calculate centroid of selected trees.")
                return

            print(f"Reference centroid: ({reference_centroid[0]:.2f}, {reference_centroid[1]:.2f})")

            # Find neighboring trees using KDTree for fast spatial queries
            neighbor_tree_ids = list(selected_tree_ids)  # Start with selected trees
            
            # Get centroids for all trees (excluding selected ones)
            all_centroids = []
            candidate_tree_ids = []
            
            for tree_id in self.unique_itc_values:
                if tree_id not in selected_tree_ids and tree_id in self.tree_centroids_cache:
                    all_centroids.append(self.tree_centroids_cache[tree_id])
                    candidate_tree_ids.append(tree_id)
            
            if self.centroids_kdtree is not None and all_centroids:
                # Use KDTree for efficient radius search
                indices = self.centroids_kdtree.query_ball_point(reference_centroid, radius)
                # Map KDTree row indices back to tree IDs using the ordering the
                # KDTree was built from (NOT candidate_tree_ids — the KDTree is
                # built from the whole cache, so the two orders differ).
                kdtree_ids = getattr(self, 'centroids_kdtree_tree_ids', None) or list(self.tree_centroids_cache.keys())
                for _i in indices:
                    if 0 <= _i < len(kdtree_ids):
                        _tid = kdtree_ids[_i]
                        if _tid not in neighbor_tree_ids:
                            neighbor_tree_ids.append(_tid)
            elif all_centroids:
                # Fallback to vectorized distance calculation if KDTree not available
                centroids_array = np.array(all_centroids)
                reference_centroid_array = np.array([reference_centroid])
                
                # Calculate distances using scipy's cdist for efficiency
                distances = cdist(reference_centroid_array, centroids_array)[0]
                
                # Find trees within radius
                within_radius = distances <= radius
                neighbor_tree_ids.extend([candidate_tree_ids[i] for i in np.where(within_radius)[0]])
            
            # Fallback for any trees not in cache (shouldn't happen with precomputation)
            for tree_id in self.unique_itc_values:
                if tree_id in selected_tree_ids or tree_id in neighbor_tree_ids:
                    continue
                    
                # Calculate centroid (fallback)
                tree_centroid = self.calculate_tree_centroid(tree_id)
                if tree_centroid is None:
                    continue

                # Check distance
                distance = np.sqrt((tree_centroid[0] - reference_centroid[0])**2 + 
                                 (tree_centroid[1] - reference_centroid[1])**2)

                if distance <= radius:
                    neighbor_tree_ids.append(tree_id)

            if len(neighbor_tree_ids) == len(selected_tree_ids):
                print(f"Warning: No additional neighbors found within {radius}m radius")
                QMessageBox.information(self, "No Neighbors", f"No additional trees found within {radius}m radius.")
                return

            print(f"Found {len(neighbor_tree_ids)} trees within {radius}m radius (including {len(selected_tree_ids)} selected)")

            # Select the neighbor trees in the list
            self.tree_list.clearSelection()
            for i in range(self.tree_list.count()):
                item = self.tree_list.item(i)
                try:
                    tree_id = int(item.text())
                    if tree_id in neighbor_tree_ids:
                        item.setSelected(True)
                except ValueError:
                    continue

            # Trigger visualization
            self.visualize_tree()

        except Exception as e:
            print(f"Error in view_neighbors: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to find neighbors: {str(e)}")


    def fit_to_screen(self):
        """
        Reset the camera to fit the original tree point cloud in the view.
        Excludes volume calculation circles to prevent excessive zoom-out.
        Also sets appropriate clipping planes to prevent disappearing objects.
        """
        try:
            if self.plotter is not None and self.current_tree_points is not None:
                # Calculate bounds of the original tree point cloud only
                bounds = np.array([
                    np.min(self.current_tree_points[:, 0]),  # x_min
                    np.max(self.current_tree_points[:, 0]),  # x_max
                    np.min(self.current_tree_points[:, 1]),  # y_min
                    np.max(self.current_tree_points[:, 1]),  # y_max
                    np.min(self.current_tree_points[:, 2]),  # z_min
                    np.max(self.current_tree_points[:, 2])   # z_max
                ])

                # Calculate diagonal for setting appropriate clipping planes
                diagonal = np.sqrt(
                    (bounds[1] - bounds[0])**2 +
                    (bounds[3] - bounds[2])**2 +
                    (bounds[5] - bounds[4])**2
                )

                # Set camera clipping planes to prevent disappearing objects
                # Near plane: much closer than default to handle small objects
                # Far plane: based on scene diagonal to include all relevant objects
                near_plane = max(diagonal * 0.001, 0.01)  # At least 1cm near plane
                far_plane = diagonal * 10.0  # 10x diagonal for far plane

                self.plotter.camera.clipping_range = (near_plane, far_plane)

                # Set camera to fit the tree bounds
                self.plotter.reset_camera(bounds=bounds)
                self.log_to_console("[View] Camera reset to fit original tree point cloud")
            elif self.plotter is not None:
                # Fallback to standard reset if no tree points available
                self.plotter.reset_camera()
                self.log_to_console("[View] Camera reset (no tree points available)")
            else:
                QMessageBox.information(self, "No Plotter", "No active 3D visualization to reset camera for.")
        except Exception as e:
            QMessageBox.warning(self, "Camera Reset Error", f"Error resetting camera: {str(e)}")
            print(f"Fit to screen error: {e}")


    def clear_plot(self):
        """Clear the 3D visualization and reset camera."""
        if self.plotter:
            self.plotter.clear()
            self._restore_gdb_layers()
            self.plotter.reset_camera()
            self.plotter.update()
        
        # Reset cluster highlighting state
        self.current_cluster_index = -1
        
        # Reset trunk and branch detection state
        self.trunk_assignment = None
        self.branch_assignment = None
        
        # Disable buttons when clearing
        self.detect_branches_button.setEnabled(False)
        self.detect_trunks_button.setEnabled(False)
    # extract buttons removed
        self.draw_polygon_button.setEnabled(False)
        self.select_all_points_button.setEnabled(False)
        self.assign_itc_button.setEnabled(False)
        self.volume_calc_button.setEnabled(False)
        self.clear_volume_button.setEnabled(False)
        
        # Cancel any active polygon selection
        if self.polygon_selection_mode:
            self.cancel_polygon_selection()


    def visualize_selected_points(self):
        """Visualize the selected points in the polygon."""
        if self.selected_point_indices is None or len(self.selected_point_indices) == 0:
            return
            
        try:
            selected_points = self.current_tree_points[self.selected_point_indices]
            
            # Get ITC values for selected points and apply 9-color cycling
            if self.current_itc_values is not None and len(self.current_itc_values) > 0:
                selected_itc_values = self.current_itc_values[self.selected_point_indices]
                # Map ITC values to color indices 0-8 (9 colors total)
                color_indices = (selected_itc_values - 1) % 9
                
                # Add selected points with ITC-based colors
                selected_cloud = pv.PolyData(selected_points)
                self.plotter.add_points(selected_cloud, scalars=color_indices, point_size=5, 
                                      render_points_as_spheres=True, name='selected_points',
                                      cmap='tab10', n_colors=9, clim=[0, 8])
            else:
                # Fallback to red if no ITC values available
                selected_cloud = pv.PolyData(selected_points)
                self.plotter.add_points(selected_cloud, color='red', point_size=5, 
                                      render_points_as_spheres=True, name='selected_points')
            
            # Add polygon outline (convert 2D polygon to 3D by adding Z coordinate)
            if self.selected_polygon is not None:
                # Use average Z height for the polygon plane
                avg_z = np.mean(self.current_tree_points[:, 2]) if len(self.current_tree_points) > 0 else 0.0
                
                # Convert 2D polygon to 3D
                polygon_3d = np.column_stack([
                    self.selected_polygon[:, 0],
                    self.selected_polygon[:, 1],
                    np.full(len(self.selected_polygon), avg_z)
                ])
                
                # Close the polygon
                closed_polygon = np.vstack([polygon_3d, polygon_3d[0]])
                self.plotter.add_lines(closed_polygon, color='yellow', width=3, name='selection_polygon')
            
            self.plotter.update()
            self.log_to_console("Selected points highlighted with ITC-based colors.")
            
        except Exception as e:
            self.log_to_console(f"Error visualizing selected points: {str(e)}")


    def visualize_polygon_progress(self):
        """Visualize the polygon as it's being drawn."""
        try:
            if len(self.polygon_points) > 0:
                # Show current points
                points_array = np.array(self.polygon_points)
                points_cloud = pv.PolyData(points_array)
                self.plotter.add_points(points_cloud, color='cyan', point_size=8, 
                                      render_points_as_spheres=True, name='polygon_points')
                
                # Show lines connecting points
                if len(self.polygon_points) > 1:
                    lines = []
                    for i in range(len(self.polygon_points) - 1):
                        lines.extend([self.polygon_points[i], self.polygon_points[i+1]])
                    if len(lines) > 0:
                        lines_array = np.array(lines)
                        self.plotter.add_lines(lines_array, color='cyan', width=2, name='polygon_lines')
                
                self.plotter.update()
                
        except Exception as e:
            self.log_to_console(f"Error visualizing polygon progress: {str(e)}")


    def _visualize_dbh_components(self, components, branching_height, dbh_points):
        """Visualize the extracted DBH components in the plotter with circle fitting."""
        if self.plotter is None:
            return

        try:
            self.plotter.clear()
            self._restore_gdb_layers()

            # Create point clouds for each component
            if len(components['trunk']) > 0:
                trunk_cloud = pv.PolyData(components['trunk'])
                self.plotter.add_points(trunk_cloud, color='brown', point_size=4, opacity=0.9,
                                      name='dbh_trunk', label='DBH Trunk')

            # Visualize branches with different colors
            branch_colors = ['green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            for i, branch_points in enumerate(components['branches']):
                if len(branch_points) > 0:
                    branch_cloud = pv.PolyData(branch_points)
                    color = branch_colors[i % len(branch_colors)]
                    self.plotter.add_points(branch_cloud, color=color, point_size=4, opacity=0.8,
                                          name=f'dbh_branch_{i+1}', label=f'DBH Branch {i+1}')

            # Visualize unassigned points
            if len(components['unassigned']) > 0:
                unassigned_cloud = pv.PolyData(components['unassigned'])
                self.plotter.add_points(unassigned_cloud, color='gray', point_size=3, opacity=0.6,
                                      name='dbh_unassigned', label='DBH Unassigned')

            # Add circle fitting for DBH points
            self._add_dbh_circle_fitting(dbh_points)

            # Add reference plane at branching height
            min_z = np.min(dbh_points[:, 2])
            branching_z = min_z + branching_height
            reference_plane = pv.Plane(center=(0, 0, branching_z), direction=(0, 0, 1),
                                     i_size=50, j_size=50)

            self.plotter.add_mesh(reference_plane, color='red', opacity=0.3,
                                name='branching_plane', label=f'Branching Height: {branching_height:.2f}m')

            # Add information text
            info_text = (f"DBH Components Analysis with Circle Fitting\n"
                        f"Branching Height: {branching_height:.2f}m\n"
                        f"Total DBH Points: {len(dbh_points)}\n"
                        f"Components: {1 + len(components['branches']) + (1 if len(components['unassigned']) > 0 else 0)}")

            self.plotter.add_text(info_text, position='upper_left', font_size=10, color='#FFFFFF')

            self.plotter.reset_camera()

        except Exception as e:
            self.log_to_console(f"[Error] Error visualizing DBH components: {str(e)}")
            print(f"Error in _visualize_dbh_components: {e}")


    def _visualize_advanced_components(self, components, branching_height, all_points):
        """
        Visualize the advanced extracted components (trunk, branches, unassigned).

        Args:
            components: Dictionary with trunk, branches, and unassigned points
            branching_height: Height where branching occurs
            all_points: All original stem points
        """
        # Clear existing plot
        self.plotter.clear()
        self._restore_gdb_layers()

        trunk_points = components['trunk']
        branches = components['branches']
        unassigned_points = components['unassigned']

        # Visualize trunk in brown
        if len(trunk_points) > 0:
            trunk_cloud = pv.PolyData(trunk_points)
            self.plotter.add_points(trunk_cloud, color='brown', point_size=4, opacity=0.9,
                                  label=f'Trunk ({len(trunk_points)} pts)')

        # Visualize branches with different colors
        branch_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, branch_points in enumerate(branches):
            if len(branch_points) > 0:
                color = branch_colors[i % len(branch_colors)]
                branch_cloud = pv.PolyData(branch_points)
                self.plotter.add_points(branch_cloud, color=color, point_size=4, opacity=0.8,
                                      label=f'Branch {i+1} ({len(branch_points)} pts)')

        # Visualize unassigned points in gray
        if len(unassigned_points) > 0:
            unassigned_cloud = pv.PolyData(unassigned_points)
            self.plotter.add_points(unassigned_cloud, color='gray', point_size=3, opacity=0.6,
                                  label=f'Unassigned ({len(unassigned_points)} pts)')

        # Add reference plane at branching height
        if len(trunk_points) > 0:
            min_z = np.min(trunk_points[:, 2])
            branching_z = min_z + branching_height

            plane_bounds = [
                np.min(all_points[:, 0]) - 1, np.max(all_points[:, 0]) + 1,
                np.min(all_points[:, 1]) - 1, np.max(all_points[:, 1]) + 1,
                branching_z, branching_z
            ]

            reference_plane = pv.Plane(center=[float(np.mean([plane_bounds[0], plane_bounds[1]])),
                                             float(np.mean([plane_bounds[2], plane_bounds[3]])),
                                             float(branching_z)],
                                     i_size=plane_bounds[1]-plane_bounds[0],
                                     j_size=plane_bounds[3]-plane_bounds[2])
            self.plotter.add_mesh(reference_plane, color='red', opacity=0.3,
                                label=f'Branching Height: {branching_height:.1f}m')

        # Add info text about advanced components
        total_points = len(trunk_points) + sum(len(b) for b in branches) + len(unassigned_points)
        info_text = f"Advanced Tree Component Analysis\n"
        info_text += f"Total stem points: {total_points}\n"
        info_text += f"Trunk: {len(trunk_points)} points\n"
        for i, branch in enumerate(branches):
            info_text += f"Branch {i+1}: {len(branch)} points\n"
        if len(unassigned_points) > 0:
            info_text += f"Unassigned: {len(unassigned_points)} points\n"
        info_text += f"Branching height: {branching_height:.2f}m"

        self.plotter.add_text(info_text, position='upper_left', font_size=10, color='#FFFFFF')

        # Reset camera to show all components
        self.plotter.reset_camera()






