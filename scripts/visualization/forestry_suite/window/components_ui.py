"""Component extraction UI, DBH circle fitting, extrapolation. (TreeVisualizerGUI mixin)."""

import os
import sys
import numpy as np
import laspy
import pyvista as pv
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt
from forestry_suite.analysis.components import extract_components
from forestry_suite.analysis.volume import fit_section_circles, calculate_cylinder_volume


class ComponentsUiMixin:
    """Component extraction UI, DBH circle fitting, extrapolation."""

    def extract_component(self):
        """
        Extract individual tree components (trunk and branches) from currently visualized trunk points
        using advanced connectivity-based branch extension algorithm with DBSCAN clustering.
        """
        if not hasattr(self, 'current_tree_points') or self.current_tree_points is None:
            QMessageBox.warning(self, "No Trunk Points", "No trunk points available. Please visualize a trunk first.")
            return

        try:
            # Get the points data - use all stem points from LAS data for proper analysis
            if self.las_data is None:
                QMessageBox.warning(self, "No LAS Data", "LAS data not available for advanced analysis.")
                return

            # Get all stem points (stemcls != 1) for the selected trees
            selected_tree_ids = self.get_selected_tree_ids()
            if not selected_tree_ids:
                QMessageBox.warning(self, "No Trees Selected", "Please select tree IDs first.")
                return

            itc_values = np.array(self.las_data['itc'])
            stemcls_values = np.array(self.las_data['stemcls'])

            # Get all stem points for selected trees
            all_stem_points = []
            for tree_id in selected_tree_ids:
                mask = (itc_values == tree_id) & (stemcls_values != 1)
                tree_mask = self.get_tree_point_mask()
                if tree_mask is not None:
                    mask = mask & tree_mask

                if np.any(mask):
                    points = np.column_stack([
                        np.array(self.las_data.x)[mask],
                        np.array(self.las_data.y)[mask],
                        np.array(self.las_data.z)[mask]
                    ])
                    all_stem_points.append(points)

            if not all_stem_points:
                QMessageBox.warning(self, "No Stem Points", "No stem points found for selected trees.")
                return

            # Combine all stem points
            stem_points = np.vstack(all_stem_points)

            # Estimate branching height from current trunk visualization
            # Use the maximum height of currently visualized trunk points as branching height
            branching_height = np.max(self.current_tree_points[:, 2]) - np.min(self.current_tree_points[:, 2])

            # Extract components using the advanced algorithm
            components = extract_components(stem_points, branching_height)

            # Visualize the advanced components
            self._visualize_advanced_components(components, branching_height, stem_points)

            # Count total components (trunk + branches + unassigned if any)
            total_components = 1 + len(components['branches']) + (1 if len(components['unassigned']) > 0 else 0)
            QMessageBox.information(self, "Advanced Components Extracted",
                                  f"Successfully extracted {total_components} components:\n"
                                  f"• Trunk: {len(components['trunk'])} points\n"
                                  f"• Branches: {len(components['branches'])} branches\n"
                                  f"• Unassigned: {len(components['unassigned'])} points")

        except Exception as e:
            QMessageBox.warning(self, "Extraction Error", f"Error extracting components: {str(e)}")
            print(f"Component extraction error: {e}")


    def extract_dbh_components(self):
        """
        Extract components from DBH points only (below first split) using connectivity-based analysis.
        This analyzes the trunk structure below the first branching point.
        """
        if not hasattr(self, 'current_tree_points') or self.current_tree_points is None:
            QMessageBox.warning(self, "No Tree Points", "No tree points available. Please visualize a tree first.")
            return

        if self.las_data is None or 'dbh_points' not in self.las_data.point_format.dimension_names:
            QMessageBox.warning(self, "No DBH Points", "DBH points field not found. Please run split detection first.")
            return

        try:
            # Get DBH points for the current tree
            if self.current_mask is None:
                QMessageBox.warning(self, "No Tree Selected", "No tree is currently selected.")
                return
                
            tree_indices = np.where(self.current_mask)[0]
            dbh_values = np.array(self.las_data['dbh_points'])[tree_indices]
            dbh_mask = dbh_values == 1

            if not np.any(dbh_mask):
                QMessageBox.warning(self, "No DBH Points", "No DBH points found for the current tree.")
                return

            # Extract DBH point coordinates
            dbh_indices = tree_indices[dbh_mask]
            dbh_points = np.column_stack([
                np.array(self.las_data.x)[dbh_indices],
                np.array(self.las_data.y)[dbh_indices],
                np.array(self.las_data.z)[dbh_indices]
            ])

            self.log_to_console(f"[Metrics] Extracting components from {len(dbh_points)} DBH points")

            # For DBH component extraction, we treat the entire DBH section as one unit
            # Use a small branching height to separate any potential sub-components
            min_z = np.min(dbh_points[:, 2])
            max_z = np.max(dbh_points[:, 2])
            branching_height = (max_z - min_z) * 0.5  # Use middle height as branching point

            # Extract components using connectivity-based algorithm
            components = extract_components(dbh_points, branching_height)

            # Create scalar field for DBH components
            self.create_dbh_components_field(components, dbh_indices)

            # Visualize the DBH components
            self._visualize_dbh_components(components, branching_height, dbh_points)

            # Count total components
            total_components = 1 + len(components['branches']) + (1 if len(components['unassigned']) > 0 else 0)
            QMessageBox.information(self, "DBH Components Extracted",
                                  f"Successfully extracted {total_components} components from DBH points:\n"
                                  f"• Trunk: {len(components['trunk'])} points\n"
                                  f"• Branches: {len(components['branches'])} branches\n"
                                  f"• Unassigned: {len(components['unassigned'])} points")

        except Exception as e:
            QMessageBox.warning(self, "DBH Extraction Error", f"Error extracting DBH components: {str(e)}")
            print(f"DBH component extraction error: {e}")




    def _add_dbh_circle_fitting_for_section(self, dbh_points, section_id, circle_color, avg_color, measurement_label="DBH", run_id=None):
        """Add circle fitting visualization for section points with section-specific naming and colors."""
        if len(dbh_points) == 0:
            return {'circles': [], 'avg_radius': 0.0, 'height_range': 0.0}

        # Get height range for DBH points
        min_z = np.min(dbh_points[:, 2])
        max_z = np.max(dbh_points[:, 2])
        height_range = max_z - min_z

        self.log_to_console(f"[Measure] Section {section_id} - Z range {min_z:.2f} to {max_z:.2f}m (height: {height_range:.2f}m)")

        # Calculate circle centers (trajectory points) at regular height intervals
        circle_centers = []
        circle_radii = []
        circle_heights = []  # Store heights for DBH extraction
        step = 0.1  # 10cm height steps
        current_z = min_z

        while current_z <= max_z:
            # Extract points within thickness around current height
            thickness = 0.05  # 5cm thickness for more precise fitting
            height_mask = (dbh_points[:, 2] >= current_z - thickness/2) & \
                         (dbh_points[:, 2] <= current_z + thickness/2)
            height_points = dbh_points[height_mask]

            if len(height_points) >= 3:  # Need at least 3 points for circle fitting
                # Calculate bounding circle using centroid method
                cluster_xy = height_points[:, :2]  # Only X,Y coordinates
                centroid = np.mean(cluster_xy, axis=0)
                distances = np.linalg.norm(cluster_xy - centroid, axis=1)
                radius = np.max(distances)

                # Only include circles within reasonable radius bounds (3cm to 50cm for DBH)
                if 0.03 <= radius <= 0.5:
                    circle_centers.append([centroid[0], centroid[1], current_z])
                    circle_radii.append(radius)
                    circle_heights.append(current_z - min_z)  # Height from min_z
                    self.log_to_console(f"[OK] Section {section_id} - Z={current_z:.2f}m, center=({centroid[0]:.3f}, {centroid[1]:.3f}), radius={radius:.3f}m")

            current_z += step

        # Display the center trajectory points
        if len(circle_centers) > 0:
            centers_array = np.array(circle_centers)
            centers_cloud = pv.PolyData(centers_array)

            # Use white spheres for trajectory points
            trajectory_name = f'dbh_trajectory_section_{section_id}' if run_id is None else f'dbh_trajectory_run_{run_id}_section_{section_id}'
            self.plotter.add_mesh(centers_cloud, color='white', point_size=8,
                                render_points_as_spheres=True, opacity=1.0,
                                name=trajectory_name, label=f'Section {section_id} Trajectory ({len(circle_centers)} pts)',
                                pickable=False)

            # Compute average center and radius for all circles
            avg_x = np.mean([c[0] for c in circle_centers])
            avg_y = np.mean([c[1] for c in circle_centers])
            final_center = [avg_x, avg_y, 0]  # Z not used for center
            final_radius = np.mean(circle_radii)

            # Display the circles at each height level as filled disks
            if len(circle_centers) > 0:
                # Create all individual circle meshes first
                individual_circles = []
                for i, (center, radius) in enumerate(zip(circle_centers, circle_radii)):
                    # Log the first circle's centroid (lowest Z) before drawing - this is the "last circle" for extrapolation
                    if i == 0:
                        self.log_to_console(f" LAST CIRCLE (lowest Z): Section {section_id}, Z={center[2]:.2f}m, centroid=({center[0]:.3f}, {center[1]:.3f}), radius={radius:.3f}m")
                    
                    # Create a circle at this height
                    circle = pv.Circle(radius=radius, resolution=32)
                    # Translate circle to the center position
                    circle.translate([center[0], center[1], center[2]], inplace=True)
                    individual_circles.append(circle)

                # Merge all individual circles into a single mesh for better performance
                merged_individual = individual_circles[0].copy()
                for circle in individual_circles[1:]:
                    merged_individual = merged_individual + circle

                # Add the merged mesh to the plotter
                individual_name = f'dbh_individual_circles_section_{section_id}' if run_id is None else f'dbh_individual_circles_run_{run_id}_section_{section_id}'
                self.plotter.add_mesh(merged_individual, color=circle_color, opacity=0.3,
                                    name=individual_name,
                                    label=f'Section {section_id} Individual Circles ({len(circle_centers)} circles)',
                                    pickable=False)

            # Add average circle at EVERY height level where individual circles exist
            self._add_average_dbh_circles_for_section(circle_centers, final_center, final_radius, section_id, avg_color, run_id=run_id)

            self.log_to_console(f"[Metrics] Added section {section_id} with {len(circle_centers)} {measurement_label} circles")

            # Extract DBH at the configured reference height above the tree base using the AVERAGE radius
            # Only compute DBH when this section contains the global DBH height.
            dbh_at_1_3m = None
            dbh_center_at_1_3m = None
            dbh_z_at_1_3m = None
            target_height = getattr(self, 'dbh_reference_height', 1.3)
            tree_base_z = float(np.min(self.current_tree_points[:, 2])) if hasattr(self, 'current_tree_points') and self.current_tree_points is not None else float(min_z)
            dbh_target_z = tree_base_z + target_height

            if len(circle_heights) > 0:
                section_min_z = float(min_z)
                section_max_z = float(max_z)

                if section_min_z <= dbh_target_z <= section_max_z:
                    # Use the fitted circle closest to the global DBH target height.
                    circle_zs = np.array([center[2] for center in circle_centers], dtype=float)
                    target_idx = int(np.argmin(np.abs(circle_zs - dbh_target_z)))
                    avg_radius_at_1_3m = float(circle_radii[target_idx])
                    dbh_at_1_3m = avg_radius_at_1_3m * 2  # Diameter

                    # Use the selected circle center for visualization
                    dbh_center_at_1_3m = np.array([circle_centers[target_idx][0], circle_centers[target_idx][1], dbh_target_z])
                    dbh_z_at_1_3m = dbh_target_z

                    self.log_to_console(f"[Tree] DBH at {target_height:.1f}m above base: {dbh_at_1_3m*100:.1f} cm (radius at target Z: {avg_radius_at_1_3m*100:.1f} cm)")

                    # DBH visualization at section level has been removed to avoid duplicate/green overlays.
                    # DBH value is still calculated and returned in section data; trunk-level (blue) overlays
                    # are used for display instead.
                elif measurement_label.lower() == "diameter":
                    self.log_to_console(f"[WARN] Section height range ({height_range:.2f}m) does not include DBH height at {dbh_target_z:.2f}m")
                else:
                    self.log_to_console(f"[WARN] Section height range ({height_range:.2f}m) doesn't reach {target_height:.1f}m for {measurement_label} measurement")

            # Return circle data for volume calculation
            return {
                'circles': list(zip(circle_centers, circle_radii)),
                'avg_radius': final_radius,
                'avg_center': [final_center[0], final_center[1]],  # Save the average center coordinates
                'height_range': height_range,
                'min_z': min_z,
                'max_z': max_z,
                'dbh_at_1_3m': dbh_at_1_3m,  # Diameter at 1.3m in meters (using avg radius)
                'dbh_center_at_1_3m': dbh_center_at_1_3m.tolist() if dbh_center_at_1_3m is not None else None,
                'dbh_z_at_1_3m': dbh_z_at_1_3m,
                'dbh_target_z': dbh_target_z
            }
        else:
            self.log_to_console(f"[WARN] No valid circles found for section {section_id}")
            return {'circles': [], 'avg_radius': 0.0, 'height_range': 0.0, 'dbh_at_1_3m': None, 'dbh_target_z': None}


    def _calculate_section_volume(self, section_data):
        """Calculate the volume of a section based on fitted circles using cylinder approximation."""
        if not section_data or 'avg_radius' not in section_data or section_data['avg_radius'] <= 0:
            return 0.0
        return calculate_cylinder_volume(section_data['avg_radius'], section_data.get('height_range', 0.0))


    def _check_global_ground_proximity_and_extrapolate(self):
        """Check ground proximity across all accumulated sections and extrapolate if needed."""
        if not self.accumulated_volume_sections:
            return

        self.log_to_console(f" Checking global ground proximity across {len(self.accumulated_volume_sections)} sections")

        # Determine ground level from the global LAS dataset bounding box (true ground)
        if self.las_data is not None:
            # Use the global bounding box min z as ground level
            all_z_values = np.array(self.las_data.z)
            ground_level = np.min(all_z_values)
            self.log_to_console(f" Global ground level from LAS dataset: {ground_level:.2f}m (true ground)")
            
            # Stop extrapolation 0.5m above ground to avoid ground interference
            extrapolation_stop_level = ground_level + 0.5
            self.log_to_console(f"[Measure] Extrapolation will stop at Z: {extrapolation_stop_level:.2f}m (0.5m above ground)")
        else:
            self.log_to_console(f"[WARN] No LAS data available for ground level detection")
            return

        # Find the lowest fitted circle across all sections (not just the min Z of input points)
        global_min_fitted_z = float('inf')
        lowest_section = None
        
        for section in self.accumulated_volume_sections:
            if 'circle_data' in section and section['circle_data']:
                circles = section['circle_data'].get('circles', [])
                if circles:
                    # Get the minimum Z from actual fitted circles
                    section_min_fitted_z = min(center[2] for center, _ in circles)
                    if section_min_fitted_z < global_min_fitted_z:
                        global_min_fitted_z = section_min_fitted_z
                        lowest_section = section

        if global_min_fitted_z == float('inf'):
            self.log_to_console(f"[WARN] No fitted circles found in any section")
            return

        global_min_z = global_min_fitted_z  # Use the actual lowest fitted circle Z

        if global_min_z == float('inf'):
            self.log_to_console(f"[WARN] No valid circle data found in any section")
            return

        self.log_to_console(f"[Measure] Global lowest circle Z: {global_min_z:.2f}m (from section {lowest_section['section_id'] if lowest_section else 'unknown'})")

        # Check proximity to stop level
        proximity_threshold = 0.0  # Always extrapolate if below stop level
        distance_to_stop = global_min_z - extrapolation_stop_level
        
        self.log_to_console(f"[Metrics] Global distance to stop level: {distance_to_stop:.2f}m (stopping 0.5m above ground)")

        if distance_to_stop <= proximity_threshold:
            self.log_to_console(f"[OK] Global lowest circle already at or above stop level - no extrapolation needed")
            return

        self.log_to_console(f" Global extrapolation needed - gap: {distance_to_stop:.2f}m")
        self.log_to_console(f" Extrapolating downward from global lowest point...")

        # Use the lowest section's data for extrapolation
        if not lowest_section or 'circle_data' not in lowest_section:
            self.log_to_console(f"[WARN] No valid lowest section data for extrapolation")
            return

        lowest_circle_data = lowest_section['circle_data']
        existing_circles = lowest_circle_data.get('circles', [])
        
        if not existing_circles:
            self.log_to_console(f"[WARN] No circles in lowest section")
            return

        # Get the step size
        step = 0.1  # 10cm steps

        # Calculate how many additional circles we need
        num_additional_circles = int(np.ceil(distance_to_stop / step))
        
        # Apply safety limit to prevent excessive extrapolation
        max_extrapolated_circles = 100
        if num_additional_circles > max_extrapolated_circles:
            num_additional_circles = max_extrapolated_circles
            self.log_to_console(f"[Measure] Limiting extrapolation to {max_extrapolated_circles} circles for safety")
        
        self.log_to_console(f"[Measure] Planning to add {num_additional_circles} global extrapolated circles (until ground level)")

        # Use the centroid from the lowest individual circle (same as the last red avg circle position)
        lowest_circle_center, lowest_circle_radius = existing_circles[0]  # First circle is lowest Z
        extrapolation_centroid = [lowest_circle_center[0], lowest_circle_center[1]]
        
        self.log_to_console(f" LOWEST CIRCLE (for extrapolation): Section {lowest_section['section_id']}, Z={lowest_circle_center[2]:.2f}m, centroid=({lowest_circle_center[0]:.3f}, {lowest_circle_center[1]:.3f}), radius={lowest_circle_radius:.3f}m")
        self.log_to_console(f"[Point] Using lowest circle centroid for extrapolation (matches red avg circle position): ({extrapolation_centroid[0]:.3f}, {extrapolation_centroid[1]:.3f})m")

        # Use the average radius across all circles from the lowest section for consistency
        all_radii = []
        for section in self.accumulated_volume_sections:
            if 'circle_data' in section and section['circle_data']:
                section_circles = section['circle_data'].get('circles', [])
                for _, radius in section_circles:
                    all_radii.append(radius)
        
        global_avg_radius = np.mean(all_radii) if all_radii else 0.1  # Default fallback
        self.log_to_console(f"[Measure] Global average radius: {global_avg_radius:.3f}m (from {len(all_radii)} total circles)")

        # Start extrapolation from the lowest measured circle Z
        current_z = global_min_z

        # Create a new global extrapolated section
        extrapolated_circles = []

        # Calculate extrapolation z levels
        extrapolation_z_levels = []
        current_z = global_min_z - step
        for _ in range(num_additional_circles):
            if current_z < extrapolation_stop_level:
                break
            extrapolation_z_levels.append(current_z)
            current_z -= step

        # If the last level is above stop level, add stop level
        if extrapolation_z_levels and extrapolation_z_levels[-1] > extrapolation_stop_level:
            extrapolation_z_levels.append(extrapolation_stop_level)

        # Now add circles for each z in extrapolation_z_levels
        extrapolated_circle_meshes = []
        for i, z in enumerate(extrapolation_z_levels):
            extrapolated_center = [extrapolation_centroid[0], extrapolation_centroid[1], z]
            extrapolated_circles.append((extrapolated_center, global_avg_radius))

            # Log the first extrapolated circle's centroid
            if i == 0:
                self.log_to_console(f" FIRST EXTRAPOLATED CIRCLE: Z={z:.2f}m, centroid=({extrapolation_centroid[0]:.3f}, {extrapolation_centroid[1]:.3f}), radius={global_avg_radius:.3f}m")

            self.log_to_console(f" Added global extrapolated circle at Z={z:.2f}m, center=({extrapolation_centroid[0]:.3f}, {extrapolation_centroid[1]:.3f}), radius={global_avg_radius:.3f}m")

            # Create circle mesh but don't add to plotter yet
            circle = pv.Circle(radius=global_avg_radius, resolution=32)
            circle.translate(extrapolated_center, inplace=True)
            extrapolated_circle_meshes.append(circle)

        # Merge all extrapolated circles into a single mesh for better performance
        if extrapolated_circle_meshes:
            merged_extrapolated = extrapolated_circle_meshes[0].copy()
            for circle in extrapolated_circle_meshes[1:]:
                merged_extrapolated = merged_extrapolated + circle

            # Add the merged mesh to the plotter
            self.plotter.add_mesh(merged_extrapolated, color='cyan', opacity=0.3,
                                name=f'dbh_global_extrapolated_circles',
                                label=f'Global Extrapolated Circles ({len(extrapolated_circle_meshes)} circles)',
                                pickable=False)

        if extrapolated_circles:
            # Create a new global extrapolated section
            global_section_id = "global_extrapolated"
            
            # Calculate averages for extrapolated circles
            all_centers = [center for center, _ in extrapolated_circles]
            all_radii = [radius for _, radius in extrapolated_circles]
            
            global_avg_x = np.mean([c[0] for c in all_centers])
            global_avg_y = np.mean([c[1] for c in all_centers])
            global_avg_radius = np.mean(all_radii)
            
            global_min_z_new = min(c[2] for c in all_centers)
            global_max_z_new = max(c[2] for c in all_centers)
            global_height_range = global_max_z_new - global_min_z_new

            # Create extrapolated section data
            extrapolated_section_data = {
                'circles': extrapolated_circles,
                'avg_radius': global_avg_radius,
                'avg_center': [global_avg_x, global_avg_y],
                'height_range': global_height_range,
                'min_z': global_min_z_new,
                'max_z': global_max_z_new
            }

            # Add average circles for global extrapolated section
            self._add_average_dbh_circles_for_section(all_centers,
                                                    [global_avg_x, global_avg_y, 0],
                                                    global_avg_radius,
                                                    global_section_id,
                                                    'red')

            # Add to accumulated sections
            global_section_info = {
                'tree_id': lowest_section['tree_id'] if lowest_section else self.current_tree_id,
                'section_id': global_section_id,
                'points': np.array([]),  # No original points for extrapolated
                'point_color': 'cyan',
                'circle_color': 'cyan',
                'volume': self._calculate_section_volume(extrapolated_section_data),
            'circle_data': extrapolated_section_data,
            'extrapolated': True,
            'original_height_range': 0.0,
            'original_min_z': global_min_z,
            'original_max_z': global_min_z,
            'extrapolated_height_range': global_height_range,
            'extrapolated_min_z': global_min_z_new,
            'extrapolated_max_z': global_max_z_new,
            'num_extrapolated_circles': len(extrapolated_circles),
            'ground_level_used': ground_level
            }
            self.accumulated_volume_sections.append(global_section_info)
            # Calculate total cylinder height across all sections
            all_z = []
            for section in self.accumulated_volume_sections:
                if 'circle_data' in section and section['circle_data']:
                    circles = section['circle_data'].get('circles', [])
                    for center, _ in circles:
                        all_z.append(center[2])
            
            if all_z:
                total_min_z = min(all_z)
                total_max_z = max(all_z)
                total_height = total_max_z - total_min_z
                self.log_to_console(f"[Dim] Total cylinder height: {total_height:.2f}m (from Z {total_min_z:.2f}m to {total_max_z:.2f}m)")


    def _add_average_dbh_circles_for_section(self, circle_centers, avg_center, avg_radius, section_id, avg_color, run_id=None):
        """Add average circle at every height level for a specific section."""
        if len(circle_centers) == 0:
            return

        try:
            # Create all average circle meshes first
            avg_circles = []
            for i, center in enumerate(circle_centers):
                height_z = center[2]  # Z coordinate of this height level

                # Create average circle at this height
                avg_circle = pv.Circle(radius=avg_radius, resolution=64)
                avg_circle.translate([center[0], center[1], height_z], inplace=True)

                # Make it wireframe (boundary only)
                avg_circle_wireframe = pv.PolyData(avg_circle.points)
                avg_circle_wireframe.lines = avg_circle.lines
                
                avg_circles.append(avg_circle_wireframe)

            # Merge all average circles into a single mesh for better performance
            if len(avg_circles) > 0:
                merged_average = avg_circles[0].copy()
                for circle in avg_circles[1:]:
                    merged_average = merged_average + circle

                # Add the merged mesh to the plotter
                actor_name = f'dbh_avg_circles_section_{section_id}' if run_id is None else f'dbh_avg_circles_run_{run_id}_section_{section_id}'
                self.plotter.add_mesh(merged_average, color=avg_color, line_width=3, opacity=0.9,
                                    name=actor_name,
                                    label=f'Section {section_id} Avg Circles ({len(circle_centers)} circles)',
                                    pickable=False)

            # Add a single center marker for the average center trajectory
            centers_array = np.array([[center[0], center[1], center[2]] for center in circle_centers])
            avg_centers_cloud = pv.PolyData(centers_array)
            traj_name = f'dbh_avg_trajectory_section_{section_id}' if run_id is None else f'dbh_avg_trajectory_run_{run_id}_section_{section_id}'
            self.plotter.add_mesh(avg_centers_cloud, color=avg_color, point_size=6,
                                render_points_as_spheres=True, opacity=1.0,
                                name=traj_name, label=f'Section {section_id} Avg Trajectory ({len(circle_centers)} pts)',
                                pickable=False)

            self.log_to_console(f" Added section {section_id} average circles at {len(circle_centers)} height levels (R={avg_radius:.3f}m)")

        except Exception as e:
            self.log_to_console(f"[Error] Error adding average DBH circles for section {section_id}: {str(e)}")
            print(f"Error in _add_average_dbh_circles_for_section: {e}")


    def create_dbh_components_field(self, components, dbh_indices):
        """Create scalar field for DBH components: 0=trunk, 1+=branch IDs, 255=unassigned."""
        if self.las_data is None:
            return

        try:
            # Initialize field with zeros
            dbh_components = np.zeros(len(self.las_data), dtype=np.uint8)

            # Assign trunk points (value = 0)
            if len(components['trunk']) > 0:
                # Find indices of trunk points in the full LAS data
                trunk_points = components['trunk']
                for i, trunk_point in enumerate(trunk_points):
                    # Find the closest point in dbh_indices
                    diffs = np.column_stack([
                        np.array(self.las_data.x)[dbh_indices] - trunk_point[0],
                        np.array(self.las_data.y)[dbh_indices] - trunk_point[1],
                        np.array(self.las_data.z)[dbh_indices] - trunk_point[2]
                    ])
                    distances = np.linalg.norm(diffs, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.01:  # Very close match
                        dbh_components[dbh_indices[closest_idx]] = 0

            # Assign branch points (values 1, 2, 3, ...)
            for branch_id, branch_points in enumerate(components['branches'], 1):
                for branch_point in branch_points:
                    # Find the closest point in dbh_indices
                    diffs = np.column_stack([
                        np.array(self.las_data.x)[dbh_indices] - branch_point[0],
                        np.array(self.las_data.y)[dbh_indices] - branch_point[1],
                        np.array(self.las_data.z)[dbh_indices] - branch_point[2]
                    ])
                    distances = np.linalg.norm(diffs, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.01:  # Very close match
                        dbh_components[dbh_indices[closest_idx]] = branch_id

            # Assign unassigned points (value = 255)
            if len(components['unassigned']) > 0:
                unassigned_points = components['unassigned']
                for unassigned_point in unassigned_points:
                    # Find the closest point in dbh_indices
                    diffs = np.column_stack([
                        np.array(self.las_data.x)[dbh_indices] - unassigned_point[0],
                        np.array(self.las_data.y)[dbh_indices] - unassigned_point[1],
                        np.array(self.las_data.z)[dbh_indices] - unassigned_point[2]
                    ])
                    distances = np.linalg.norm(diffs, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 0.01:  # Very close match
                        dbh_components[dbh_indices[closest_idx]] = 255

            # Add or update the scalar field
            field_name = "dbh_components"
            if field_name in self.las_data.point_format.dimension_names:
                self.las_data[field_name] = dbh_components
                self.log_to_console("[OK] Updated existing 'dbh_components' scalar field")
            else:
                try:
                    self.las_data.add_extra_dim(laspy.ExtraBytesParams(name=field_name, type=np.uint8))
                    self.las_data[field_name] = dbh_components
                    self.log_to_console("[OK] Added new 'dbh_components' scalar field to LAS data")
                except Exception as e:
                    self.log_to_console(f"[WARN] Could not create 'dbh_components' field: {str(e)}")
                    return

            # Refresh color combo box to include the new field
            self.refresh_color_combo_box()

        except Exception as e:
            self.log_to_console(f"[Error] Error creating DBH components field: {str(e)}")
            print(f"Error in create_dbh_components_field: {str(e)}")


    def _add_dbh_circle_fitting(self, dbh_points):
        """Add circle fitting visualization for DBH points at multiple height levels."""
        if len(dbh_points) == 0:
            return

        # Get height range for DBH points
        min_z = np.min(dbh_points[:, 2])
        max_z = np.max(dbh_points[:, 2])
        height_range = max_z - min_z

        self.log_to_console(f"[Measure] DBH Circle Fitting: Z range {min_z:.2f} to {max_z:.2f}m (height: {height_range:.2f}m)")

        # Calculate circle centers (trajectory points) at regular height intervals
        circle_centers = []
        circle_radii = []
        step = 0.1  # 10cm height steps
        current_z = min_z

        while current_z <= max_z:
            # Extract points within thickness around current height
            thickness = 0.05  # 5cm thickness for more precise fitting
            height_mask = (dbh_points[:, 2] >= current_z - thickness/2) & \
                         (dbh_points[:, 2] <= current_z + thickness/2)
            height_points = dbh_points[height_mask]

            if len(height_points) >= 3:  # Need at least 3 points for circle fitting
                # Calculate bounding circle using centroid method
                cluster_xy = height_points[:, :2]  # Only X,Y coordinates
                centroid = np.mean(cluster_xy, axis=0)
                distances = np.linalg.norm(cluster_xy - centroid, axis=1)
                radius = np.max(distances)

                # Only include circles within reasonable radius bounds (3cm to 50cm for DBH)
                if 0.03 <= radius <= 0.5:
                    circle_centers.append([centroid[0], centroid[1], current_z])
                    circle_radii.append(radius)
                    self.log_to_console(f"[OK] DBH Circle: Z={current_z:.2f}m, center=({centroid[0]:.3f}, {centroid[1]:.3f}), radius={radius:.3f}m")

            current_z += step

        # Display the center trajectory points
        if len(circle_centers) > 0:
            centers_array = np.array(circle_centers)
            centers_cloud = pv.PolyData(centers_array)

            # Use white spheres for trajectory points
            self.plotter.add_mesh(centers_cloud, color='white', point_size=8,
                                render_points_as_spheres=True, opacity=1.0,
                                name='dbh_trajectory', label=f'DBH Trajectory ({len(circle_centers)} pts)',
                                pickable=False)

            # Compute average center and radius for all circles
            avg_x = np.mean([c[0] for c in circle_centers])
            avg_y = np.mean([c[1] for c in circle_centers])
            final_center = [avg_x, avg_y, 0]  # Z not used for center
            final_radius = np.mean(circle_radii)

            # Display the circles at each height level as filled disks
            for i, (center, radius) in enumerate(zip(circle_centers, circle_radii)):
                # Create a circle at this height
                circle = pv.Circle(radius=radius, resolution=32)
                # Translate circle to the center position
                circle.translate([center[0], center[1], center[2]], inplace=True)

                # Use filled circles (disks) for individual circles
                self.plotter.add_mesh(circle, color='cyan', opacity=0.3,
                                    name=f'dbh_circle_{i}', label=f'DBH Circle Z={center[2]:.1f}m',
                                    pickable=False)

            # Add average circle at EVERY height level where individual circles exist
            self._add_average_dbh_circles_at_all_levels(circle_centers, final_center, final_radius)

            self.log_to_console(f"[Metrics] Added {len(circle_centers)} DBH circles for trajectory visualization")

        else:
            self.log_to_console("[WARN] No valid circles found for DBH trajectory")


    def _add_average_dbh_circles_at_all_levels(self, circle_centers, avg_center, avg_radius):
        """Add average circle at every height level where individual circles were calculated."""
        if len(circle_centers) == 0:
            return

        try:
            # Add average circle at EACH height level
            for i, center in enumerate(circle_centers):
                height_z = center[2]  # Z coordinate of this height level

                # Create average circle at this height
                avg_circle = pv.Circle(radius=avg_radius, resolution=64)
                avg_circle.translate([avg_center[0], avg_center[1], height_z], inplace=True)

                # Make it wireframe (boundary only)
                avg_circle_wireframe = pv.PolyData(avg_circle.points)
                avg_circle_wireframe.lines = avg_circle.lines

                # Use red color for average circles, slightly thicker
                self.plotter.add_mesh(avg_circle_wireframe, color='red', line_width=3, opacity=0.9,
                                    name=f'dbh_avg_circle_{i}', label=f'DBH Avg Circle Z={height_z:.1f}m',
                                    pickable=False)

            # Add a single center marker for the average center trajectory
            centers_array = np.array([[avg_center[0], avg_center[1], center[2]] for center in circle_centers])
            avg_centers_cloud = pv.PolyData(centers_array)
            self.plotter.add_mesh(avg_centers_cloud, color='red', point_size=6,
                                render_points_as_spheres=True, opacity=1.0,
                                name='dbh_avg_trajectory', label=f'DBH Avg Trajectory ({len(circle_centers)} pts)',
                                pickable=False)

            self.log_to_console(f" Added DBH Average Circles at {len(circle_centers)} height levels (R={avg_radius:.3f}m)")

        except Exception as e:
            self.log_to_console(f"[Error] Error adding average DBH circles at all levels: {str(e)}")
            print(f"Error in _add_average_dbh_circles_at_all_levels: {e}")


