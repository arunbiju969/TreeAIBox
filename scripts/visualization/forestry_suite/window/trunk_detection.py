"""Slice-based trunk and branch detection and rendering. (TreeVisualizerGUI mixin)."""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import matplotlib.pyplot as plt

from forestry_suite.dialogs.slice_dbscan import SliceDBSCANViewer
from forestry_suite.analysis.trunk import (
    build_z_slices,
    cluster_slice_xy,
    extract_slice_data,
    assign_points_to_nearest_trajectory,
)


class TrunkDetectionMixin:
    """Slice-based trunk and branch detection and rendering."""

    SliceDBSCANViewer = SliceDBSCANViewer

    def _build_z_slices(self, points, slice_height_m):
        """Build z-slice boundaries for a given slice height."""
        return build_z_slices(points, slice_height_m)

    def _cluster_xy_in_slice(self, slice_points_xy, eps_m, min_samples):
        """Run DBSCAN on 2D XY coordinates in a horizontal slice."""
        return cluster_slice_xy(slice_points_xy, eps_m, min_samples)


    def detect_trunks_in_slices(self):
        """Detect individual trunks in the base zone using horizontal slicing + DBSCAN + vertical tracking.
        
        Populates self.trunk_assignment with trunk IDs in the base zone.
        """
        if self.current_tree_points is None:
            QMessageBox.warning(self, "Warning", "No trunk visualized. Please visualize a trunk first.")
            return

        try:
            self.log_to_console(f"\n{'='*60}")
            self.log_to_console(f"[Tree] PHASE 1: Detecting individual trunks in base zone...")
            self.log_to_console(f"{'='*60}")

            params = self.trunk_detection_params
            params['slice_height_cm'] = self.trunk_slice_height_input.value()
            params['base_zone_height_m'] = float(self.trunk_base_zone_input.value())
            params['eps_m'] = float(self.trunk_eps_input.value())
            params['min_samples'] = int(self.trunk_min_samples_input.value())
            params['merge_dist_m'] = float(self.trunk_merge_dist_input.value())
            params['min_trunk_points'] = int(self.trunk_min_points_input.value())
            params['min_vertical_span_m'] = float(self.trunk_min_span_input.value())
            params['full_height_assign_dist_m'] = float(self.trunk_full_assign_input.value())

            # Record the parameter set used for this detection run and compare to last run.
            try:
                current_trunk_params = {
                    'slice_height_cm': int(params.get('slice_height_cm', 0)),
                    'base_zone_height_m': float(params.get('base_zone_height_m', 0.0)),
                    'eps_m': float(params.get('eps_m', 0.0)),
                    'min_samples': int(params.get('min_samples', 0)),
                    'merge_dist_m': float(params.get('merge_dist_m', 0.0)),
                    'min_trunk_points': int(params.get('min_trunk_points', 0)),
                    'min_vertical_span_m': float(params.get('min_vertical_span_m', 0.0)),
                    'full_height_assign_dist_m': float(params.get('full_height_assign_dist_m', 0.0))
                }
            except Exception:
                current_trunk_params = None

            slice_height_m = params['slice_height_cm'] / 100.0
            base_zone_height_m = params['base_zone_height_m']
            eps_m = params['eps_m']
            min_samples = params['min_samples']
            merge_dist_m = params['merge_dist_m']
            min_trunk_points = params['min_trunk_points']
            min_vertical_span_m = params['min_vertical_span_m']

            points = self.current_tree_points
            z_coords = points[:, 2]
            min_z = np.min(z_coords)
            max_z = np.max(z_coords)
            trunk_height = max_z - min_z

            self.log_to_console(f"[Measure] Trunk height: {trunk_height:.2f}m | Base zone: {base_zone_height_m:.2f}m")
            self.log_to_console(f" Slice height: {params['slice_height_cm']}cm | DBSCAN eps: {eps_m:.2f}m, min_samples: {min_samples}")

            # Use full height when base zone is set to 0.0, otherwise cap by the requested value.
            zone_height = trunk_height if base_zone_height_m <= 0 else min(base_zone_height_m, trunk_height)
            zone_max_z = min_z + zone_height
            self.log_to_console(f" Base zone Z range: {min_z:.2f}m to {zone_max_z:.2f}m")

            # Build slices in base zone only
            z_coords_zone, _, _, slice_boundaries, num_slices = self._build_z_slices(
                points[z_coords <= zone_max_z], slice_height_m
            )
            
            self.log_to_console(f"[Metrics] Total base slices: {num_slices}")

            # Initialize trunk assignment for base zone
            base_zone_mask = z_coords <= zone_max_z
            base_zone_indices = np.where(base_zone_mask)[0]
            base_zone_points = points[base_zone_indices]
            
            point_to_trunk_base = np.full(len(base_zone_indices), -1, dtype=int)
            next_trunk_id = 0
            prev_slice_clusters = {}  # Maps trunk_id to centroid
            self.log_to_console(f"\n Running DBSCAN per base slice with vertical tracking...")

            # Collect per-slice debug data for viewer
            slices_debug = []

            # Process each base zone slice
            for slice_idx in range(num_slices):
                z_start = slice_boundaries[slice_idx]
                z_end = slice_boundaries[slice_idx + 1]

                # Get points in this slice from base zone
                slice_mask = (z_coords_zone >= z_start) & (z_coords_zone < z_end)
                if slice_idx == num_slices - 1:
                    slice_mask = (z_coords_zone >= z_start) & (z_coords_zone <= z_end)

                slice_indices_local = np.where(slice_mask)[0]
                slice_indices_global = base_zone_indices[slice_indices_local]
                slice_points = base_zone_points[slice_indices_local]

                if len(slice_points) < min_samples:
                    self.log_to_console(f"  Slice {slice_idx}: < {min_samples} points, skipping")
                    # Save empty debug entry
                    slices_debug.append({
                        'slice_idx': slice_idx,
                        'z_start': z_start,
                        'z_end': z_end,
                        'points': np.empty((0, 2)),
                        'labels': np.array([], dtype=int),
                        'num_clusters': 0,
                        'num_noise': 0
                    })
                    continue

                # Compute radial coordinates from slice center
                center_x = np.mean(slice_points[:, 0])
                center_y = np.mean(slice_points[:, 1])
                radial_coords = np.column_stack([
                    slice_points[:, 0] - center_x,
                    slice_points[:, 1] - center_y
                ])

                # Run DBSCAN
                local_labels, num_clusters, num_noise = self._cluster_xy_in_slice(radial_coords, eps_m, min_samples)
                self.log_to_console(f"  Slice {slice_idx}: {num_clusters} clusters, {num_noise} noise")

                # Save debug data (use absolute XY for display)
                slices_debug.append({
                    'slice_idx': slice_idx,
                    'z_start': z_start,
                    'z_end': z_end,
                    'points': slice_points[:, :2].copy(),
                    'labels': local_labels.copy(),
                    'num_clusters': num_clusters,
                    'num_noise': num_noise
                })

                # Assign trunk IDs with vertical tracking
                current_slice_clusters = {}
                for local_id in set(local_labels):
                    if local_id == -1:
                        continue

                    cluster_mask = local_labels == local_id
                    cluster_indices_local = slice_indices_local[cluster_mask]
                    cluster_indices_global = slice_indices_global[cluster_mask]
                    cluster_points = slice_points[cluster_mask]

                    centroid = np.mean(cluster_points[:, :2], axis=0)

                    # Match to previous slice cluster
                    trunk_id = None
                    if prev_slice_clusters:
                        min_dist = float('inf')
                        closest_prev_id = None
                        for prev_id, prev_centroid in prev_slice_clusters.items():
                            dist = np.linalg.norm(centroid - prev_centroid)
                            if dist < min_dist:
                                min_dist = dist
                                closest_prev_id = prev_id

                        if closest_prev_id is not None and min_dist < merge_dist_m:
                            trunk_id = closest_prev_id
                            self.log_to_console(f"    Cluster {local_id}: Merged with trunk {trunk_id} (dist={min_dist:.3f}m)")

                    if trunk_id is None:
                        trunk_id = next_trunk_id
                        next_trunk_id += 1
                        self.log_to_console(f"    Cluster {local_id}: New trunk {trunk_id}")

                    # Assign trunk ID to points in base zone
                    point_to_trunk_base[cluster_indices_local] = trunk_id
                    current_slice_clusters[trunk_id] = centroid

                prev_slice_clusters = current_slice_clusters

            # Only show the per-slice viewer in Raw mode so it does not interrupt normal detection.
            try:
                show_raw_view = bool(getattr(self, 'raw_trunk_view_checkbox', None) and self.raw_trunk_view_checkbox.isChecked())
                if show_raw_view and slices_debug:
                    viewer = self.SliceDBSCANViewer(self, slices_debug)
                    viewer.exec()
            except Exception:
                # If viewer fails, continue silently but log
                self.log_to_console("[WARN] Failed to open slice DBSCAN viewer")

            # Save raw base-zone assignment (before filtering) for QA visualization
            try:
                self.raw_trunk_base_assignment = point_to_trunk_base.copy()
                self.raw_base_zone_indices = base_zone_indices.copy()
                self.raw_slices_debug = slices_debug
            except Exception:
                pass

            # Filter trunks by quality criteria
            unique_trunks = np.unique(point_to_trunk_base)
            valid_trunks = []

            self.log_to_console(f"\n[OK] Base zone clustering complete. Filtering by quality criteria...")
            self.log_to_console(f"   Min points: {min_trunk_points} | Min vertical span: {min_vertical_span_m:.2f}m")

            for trunk_id in unique_trunks:
                if trunk_id == -1:
                    continue

                mask = point_to_trunk_base == trunk_id
                trunk_points = base_zone_points[mask]
                trunk_size = len(trunk_points)
                z_span = np.max(trunk_points[:, 2]) - np.min(trunk_points[:, 2])

                if trunk_size >= min_trunk_points and z_span >= min_vertical_span_m:
                    valid_trunks.append(trunk_id)
                    self.log_to_console(f"  Trunk {trunk_id}: {trunk_size} points, z_span={z_span:.2f}m ")
                else:
                    self.log_to_console(f"  Trunk {trunk_id}: {trunk_size} points, z_span={z_span:.2f}m  (rejected)")

            # Re-index valid trunks to 0..N-1
            trunk_remap = {old_id: new_id for new_id, old_id in enumerate(valid_trunks)}
            remapped_base = np.full(len(point_to_trunk_base), -1, dtype=int)
            for old_id, new_id in trunk_remap.items():
                remapped_base[point_to_trunk_base == old_id] = new_id

            point_to_trunk_base = remapped_base

            # Store base zone assignment and prepare for full height extension
            self.trunk_assignment = np.full(len(points), -1, dtype=int)
            self.trunk_assignment[base_zone_indices] = point_to_trunk_base

            self.log_to_console(f"\n Base zone assignment complete: {len(valid_trunks)} valid trunks detected")
            self.log_to_console(f"   Noise points in base: {np.sum(point_to_trunk_base == -1)}")

            # Extend assignment to full height
            self._assign_points_to_trunks_full_height()

            # Auto-visualize detected trunks immediately after detection
            self.log_to_console(f"\n[OK] Trunk detection complete. Auto-visualizing trunks...")
            self.visualize_trunks_from_assignment()
            # Compare current params with last used params; only bump version when parameters changed
            try:
                if current_trunk_params is not None and current_trunk_params != getattr(self, '_last_trunk_detection_params', None):
                    self._last_trunk_detection_params = current_trunk_params
                    try:
                        self._trunk_detection_version = int(getattr(self, '_trunk_detection_version', 0)) + 1
                    except Exception:
                        self._trunk_detection_version = getattr(self, '_trunk_detection_version', 0) + 1
                    self.log_to_console(f" Trunk detection parameters changed; version updated: {self._trunk_detection_version}")
                else:
                    self.log_to_console("ℹ️ Trunk detection run used same parameters as last run; detection version unchanged.")
            except Exception:
                # On any error, conservatively bump version
                try:
                    self._trunk_detection_version = int(getattr(self, '_trunk_detection_version', 0)) + 1
                except Exception:
                    self._trunk_detection_version = getattr(self, '_trunk_detection_version', 0) + 1
                self.log_to_console(f" Trunk detection version updated (fallback): {self._trunk_detection_version}")

        except Exception as e:
            self.log_to_console(f"[Error] Error detecting trunks in base zone: {str(e)}")
            import traceback
            self.log_to_console(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to detect trunks:\n{str(e)}")


    def _assign_points_to_trunks_full_height(self):
        """Extend trunk assignment from base zone to full tree height using nearest neighbor.
        
        Uses self.trunk_assignment (populated in base zone by detect_trunks_in_slices).
        Assigns remaining points (-1) to nearest trunk trajectory.
        """
        if self.trunk_assignment is None or self.current_tree_points is None:
            self.log_to_console("[WARN] Cannot extend trunk assignment: base zone assignment not ready")
            return

        points = self.current_tree_points
        unassigned_mask = self.trunk_assignment == -1

        if not np.any(unassigned_mask):
            self.log_to_console("ℹ️ All points assigned in base zone. No extension needed.")
            return

        self.log_to_console(f"\n Extending trunk assignment to full height...")
        max_assign_dist = self.trunk_detection_params['full_height_assign_dist_m']

        # Build KDTree of assigned points per trunk
        assigned_mask = self.trunk_assignment >= 0
        assigned_points = points[assigned_mask]
        assigned_trunk_ids = self.trunk_assignment[assigned_mask]
        assigned_tree = KDTree(assigned_points[:, :2])  # XY only

        unassigned_indices = np.where(unassigned_mask)[0]
        unassigned_points = points[unassigned_indices]

        # For each unassigned point, find nearest assigned point and use its trunk ID
        distances, indices = assigned_tree.query(unassigned_points[:, :2], k=1)

        extended_count = 0
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist <= max_assign_dist:
                nearest_trunk = assigned_trunk_ids[idx]
                self.trunk_assignment[unassigned_indices[i]] = nearest_trunk
                extended_count += 1

        self.log_to_console(f"   Extended {extended_count}/{len(unassigned_indices)} unassigned points")
        self.log_to_console(f"   Points still unassigned (noise): {np.sum(self.trunk_assignment == -1)}")


    def visualize_trunks_from_assignment(self):
        """Visualize trunks using colors from self.trunk_assignment (Phase 2 QA visualization).
        
        Each trunk gets a distinct color.
        """
        if self.trunk_assignment is None:
            self.log_to_console("[WARN] No trunk assignment available for visualization")
            return

        try:
            self.log_to_console(f"\nColor Visualizing trunks...")
            show_raw = bool(getattr(self, 'raw_trunk_view_checkbox', None) and self.raw_trunk_view_checkbox.isChecked())

            # Clear current visualization
            self.plotter.clear()
            self._restore_gdb_layers()

            # Show raw detections when requested, otherwise show filtered trunks.
            if show_raw and hasattr(self, 'raw_trunk_base_assignment'):
                self.log_to_console("   Raw view enabled: showing raw base-zone detections")

                raw_assign = self.raw_trunk_base_assignment
                raw_indices = self.raw_base_zone_indices
                raw_unique = np.unique(raw_assign)
                raw_unique = raw_unique[raw_unique >= 0]
                num_raw = len(raw_unique)

                # Create colors
                import matplotlib.cm as cm
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if num_raw <= 10:
                        cmap = plt.get_cmap('tab10')
                    elif num_raw <= 20:
                        cmap = plt.get_cmap('tab20')
                    else:
                        cmap = plt.get_cmap('hsv')

                colors = cmap(np.linspace(0, 1, max(1, num_raw)))

                # Render each raw detected trunk (base zone only)
                for i, raw_id in enumerate(raw_unique):
                    mask = raw_assign == raw_id
                    global_inds = raw_indices[mask]
                    trunk_points = self.current_tree_points[global_inds]

                    if len(trunk_points) > 0:
                        color = colors[i % len(colors)]
                        rgb_color = [float(color[0]), float(color[1]), float(color[2])]
                        self.plotter.add_points(
                            trunk_points,
                            color=rgb_color,
                            point_size=5,
                            name=f'RawTrunk_{raw_id}'
                        )
                        self.log_to_console(f"RAW {raw_id:<8} {len(trunk_points):<8} (base-zone)")

                # Add raw noise in base zone
                noise_mask = raw_assign == -1
                if np.any(noise_mask):
                    noise_global = raw_indices[noise_mask]
                    noise_points = self.current_tree_points[noise_global]
                    self.plotter.add_points(
                        noise_points,
                        color=[0.5, 0.5, 0.5],
                        point_size=3,
                        name='RawNoise'
                    )
                    self.log_to_console(f"{'RAW_NOISE':<10} {len(noise_points):<8} (base-zone)")

                # Update viewer
                self.plotter.reset_camera()
                self.plotter.update()
                self.log_to_console("\n[OK] Raw base-zone trunk visualization complete (unfiltered results).")
                return

            # Get unique trunk IDs
            unique_trunks = np.unique(self.trunk_assignment)
            unique_trunks = unique_trunks[unique_trunks >= 0]

            num_trunks = len(unique_trunks)
            self.log_to_console(f"   Rendering {num_trunks} trunks...")

            # If no filtered trunks found, but raw base-zone detections exist, show them for QA
            if num_trunks == 0 and hasattr(self, 'raw_trunk_base_assignment'):
                self.log_to_console("   No filtered trunks — falling back to raw base-zone clusters for QA visualization")

                raw_assign = self.raw_trunk_base_assignment
                raw_indices = self.raw_base_zone_indices
                raw_unique = np.unique(raw_assign)
                raw_unique = raw_unique[raw_unique >= 0]
                num_raw = len(raw_unique)

                # Create colors
                import matplotlib.cm as cm
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if num_raw <= 10:
                        cmap = plt.get_cmap('tab10')
                    elif num_raw <= 20:
                        cmap = plt.get_cmap('tab20')
                    else:
                        cmap = plt.get_cmap('hsv')

                colors = cmap(np.linspace(0, 1, max(1, num_raw)))

                # Render each raw detected trunk (base zone only)
                for i, raw_id in enumerate(raw_unique):
                    mask = raw_assign == raw_id
                    global_inds = raw_indices[mask]
                    trunk_points = self.current_tree_points[global_inds]

                    if len(trunk_points) > 0:
                        color = colors[i % len(colors)]
                        rgb_color = [float(color[0]), float(color[1]), float(color[2])]
                        self.plotter.add_points(
                            trunk_points,
                            color=rgb_color,
                            point_size=5,
                            name=f'RawTrunk_{raw_id}'
                        )
                        self.log_to_console(f"RAW {raw_id:<8} {len(trunk_points):<8} (base-zone)")

                # Add raw noise in base zone
                noise_mask = raw_assign == -1
                if np.any(noise_mask):
                    noise_global = raw_indices[noise_mask]
                    noise_points = self.current_tree_points[noise_global]
                    self.plotter.add_points(
                        noise_points,
                        color=[0.5, 0.5, 0.5],
                        point_size=3,
                        name='RawNoise'
                    )
                    self.log_to_console(f"{'RAW_NOISE':<10} {len(noise_points):<8} (base-zone)")

                # Update viewer
                self.plotter.reset_camera()
                self.plotter.update()
                self.log_to_console("\n[OK] Raw base-zone trunk visualization complete (unfiltered results).")
                return

            # Create color palette
            import matplotlib.cm as cm
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if num_trunks <= 10:
                    cmap = plt.get_cmap('tab10')
                elif num_trunks <= 20:
                    cmap = plt.get_cmap('tab20')
                else:
                    cmap = plt.get_cmap('hsv')

            colors = cmap(np.linspace(0, 1, num_trunks))

            # Log trunk statistics
            self.log_to_console(f"\n[Summary] TRUNK DETAILS:")
            self.log_to_console(f"{'Trunk ID':<10} {'Size':<8} {'Min Z':<8} {'Max Z':<8} {'Height':<8}")
            self.log_to_console(f"-" * 50)

            # Render each trunk
            for i, trunk_id in enumerate(unique_trunks):
                mask = self.trunk_assignment == trunk_id
                trunk_points = self.current_tree_points[mask]

                if len(trunk_points) > 0:
                    size = len(trunk_points)
                    min_z = np.min(trunk_points[:, 2])
                    max_z = np.max(trunk_points[:, 2])
                    height = max_z - min_z

                    color = colors[i % len(colors)]
                    rgb_color = [float(color[0]), float(color[1]), float(color[2])]

                    self.plotter.add_points(
                        trunk_points,
                        color=rgb_color,
                        point_size=5,
                        name=f'Trunk_{trunk_id}'
                    )

                    # Add trunk ID label at the centroid of the trunk
                    centroid = np.mean(trunk_points, axis=0)
                    try:
                        self.plotter.add_point_labels(
                            np.array([centroid]),
                            [f'T{trunk_id}'],
                            font_size=14,
                            text_color='white',
                            always_visible=True,
                            name=f'Label_Trunk_{trunk_id}'
                        )
                    except Exception:
                        pass  # Label failed; skip silently

                    self.log_to_console(f"{trunk_id:<10} {size:<8} {min_z:<8.2f} {max_z:<8.2f} {height:<8.2f}")

            # Add legend
            try:
                self.plotter.add_legend()
            except:
                pass

            self.plotter.reset_camera()
            self.plotter.update()
            
            # Noise points
            noise_mask = self.trunk_assignment == -1
            if np.any(noise_mask):
                self.log_to_console(f"{'NOISE':<10} {np.sum(noise_mask):<8}")

            self.log_to_console(f"\n[OK] Trunk visualization complete!")

        except Exception as e:
            self.log_to_console(f"[Error] Error visualizing trunks: {str(e)}")
            import traceback
            self.log_to_console(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to visualize trunks:\n{str(e)}")

    # ======================== EXISTING BRANCH DETECTION (to be refactored in Phase 3) ========================


    def detect_branch_clusters_in_slices(self):
        """Detect individual branches using DBSCAN on radial coordinates within each height slice."""
        if self.current_tree_points is None:
            QMessageBox.warning(self, "Warning", "No trunk visualized. Please visualize a trunk first.")
            return
        if self.trunk_assignment is None or len(self.trunk_assignment) != len(self.current_tree_points):
            QMessageBox.warning(self, "Warning", "No trunk assignment available. Please run trunk detection first.")
            return
        
        try:
            self.log_to_console(f"\n{'='*60}")
            self.log_to_console(f"[Tree] Detecting branches using radial DBSCAN clustering per trunk...")
            self.log_to_console(f"{'='*60}")
            
            # Get slice height parameters (same as visualization)
            slice_height_cm = self.slice_height_input.value()
            slice_height_m = slice_height_cm / 100.0
            branch_eps = float(self.branch_eps_input.value())
            branch_min_samples = int(self.branch_min_samples_input.value())
            branch_merge_dist = float(self.branch_merge_dist_input.value())

            # Capture current branch detection params and compare to last run
            try:
                current_branch_params = {
                    'slice_height_cm': int(slice_height_cm),
                    'branch_eps': float(branch_eps),
                    'branch_min_samples': int(branch_min_samples),
                    'branch_merge_dist': float(branch_merge_dist)
                }
            except Exception:
                current_branch_params = None
            
            # Extract coordinates
            points = self.current_tree_points
            trunk_ids = np.unique(self.trunk_assignment)
            trunk_ids = trunk_ids[trunk_ids >= 0]

            if len(trunk_ids) == 0:
                QMessageBox.warning(self, "Warning", "No trunk IDs found. Please run trunk detection first.")
                return
            
            # Initialize branch assignment array (-1 = noise, >=0 = branch_id)
            point_to_branch = np.full(len(points), -1, dtype=int)
            self.branch_meta = {}
            next_branch_id = 0

            total_branches_detected = 0
            total_classified_points = 0

            self.log_to_console(f"[Measure] Slice height: {slice_height_cm}cm | Branch DBSCAN eps={branch_eps:.2f}m | min_samples={branch_min_samples}")
            self.log_to_console(f" Processing {len(trunk_ids)} trunk(s) independently...")

            # Process each trunk independently so branch IDs belong to a specific trunk.
            for trunk_id in trunk_ids:
                trunk_mask = self.trunk_assignment == trunk_id
                trunk_indices = np.where(trunk_mask)[0]
                trunk_points = points[trunk_indices]

                if len(trunk_points) < branch_min_samples:
                    self.log_to_console(f"  Trunk {trunk_id}: < {branch_min_samples} points, skipping")
                    continue

                trunk_z = trunk_points[:, 2]
                min_z = np.min(trunk_z)
                max_z = np.max(trunk_z)
                slice_boundaries = np.arange(min_z, max_z + slice_height_m, slice_height_m)
                if len(slice_boundaries) < 2:
                    slice_boundaries = np.array([min_z, max_z + slice_height_m], dtype=float)
                num_slices = len(slice_boundaries) - 1

                self.log_to_console(f"  Trunk {trunk_id}: slice height {slice_height_cm}cm | slices {num_slices} | z=[{min_z:.2f}, {max_z:.2f}]")

                # Track clusters from previous slice for vertical connectivity within this trunk only.
                prev_slice_clusters = {}
                trunk_local_branch_id = 0

                for slice_idx in range(num_slices):
                    z_start = slice_boundaries[slice_idx]
                    z_end = slice_boundaries[slice_idx + 1]

                    slice_mask = (trunk_z >= z_start) & (trunk_z < z_end)
                    if slice_idx == num_slices - 1:
                        slice_mask = (trunk_z >= z_start) & (trunk_z <= z_end)

                    slice_local_indices = np.where(slice_mask)[0]
                    slice_points = trunk_points[slice_local_indices]

                    if len(slice_points) < branch_min_samples:
                        self.log_to_console(f"    Slice {slice_idx}: < {branch_min_samples} points, skipping")
                        continue

                    center_x = np.mean(slice_points[:, 0])
                    center_y = np.mean(slice_points[:, 1])

                    radial_coords = np.column_stack([
                        slice_points[:, 0] - center_x,
                        slice_points[:, 1] - center_y
                    ])

                    dbscan = DBSCAN(eps=branch_eps, min_samples=branch_min_samples)
                    local_clusters = dbscan.fit_predict(radial_coords)

                    num_clusters = len(set(local_clusters)) - (1 if -1 in local_clusters else 0)
                    num_noise = np.sum(local_clusters == -1)
                    self.log_to_console(f"    Slice {slice_idx}: {num_clusters} clusters, {num_noise} noise points")

                    current_slice_clusters = {}
                    for local_id in np.unique(local_clusters):
                        if local_id == -1:
                            continue

                        cluster_mask = local_clusters == local_id
                        cluster_local_indices = slice_local_indices[cluster_mask]
                        cluster_indices = trunk_indices[cluster_local_indices]
                        cluster_points = slice_points[cluster_mask]

                        centroid = np.mean(cluster_points[:, :2], axis=0)
                        cluster_z_min = float(np.min(cluster_points[:, 2]))
                        cluster_z_max = float(np.max(cluster_points[:, 2]))

                        branch_id = None
                        if prev_slice_clusters:
                            min_dist = float('inf')
                            closest_prev_id = None
                            for prev_id, prev_centroid in prev_slice_clusters.items():
                                dist = np.linalg.norm(centroid - prev_centroid)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_prev_id = prev_id

                            if closest_prev_id is not None and min_dist < branch_merge_dist:
                                branch_id = closest_prev_id
                                meta = self.branch_meta.get(branch_id, {})
                                local_branch_id = meta.get('local_id', branch_id)
                                self.log_to_console(
                                    f"      Cluster {local_id}: Merged with trunk {trunk_id} branch B{local_branch_id} "
                                    f"(global {branch_id}, dist={min_dist:.3f}m)"
                                )

                        if branch_id is None:
                            branch_id = next_branch_id
                            next_branch_id += 1
                            local_branch_id = trunk_local_branch_id
                            trunk_local_branch_id += 1
                            self.branch_meta[branch_id] = {
                                'trunk_id': int(trunk_id),
                                'local_id': int(local_branch_id),
                                'point_count': 0,
                                'z_span': (cluster_z_min, cluster_z_max)
                            }
                            self.log_to_console(
                                f"      Cluster {local_id}: New branch T{trunk_id}.B{local_branch_id} (global {branch_id})"
                            )

                        point_to_branch[cluster_indices] = branch_id
                        current_slice_clusters[branch_id] = centroid

                        meta = self.branch_meta.setdefault(branch_id, {
                            'trunk_id': int(trunk_id),
                            'local_id': int(branch_id),
                            'point_count': 0,
                            'z_span': (cluster_z_min, cluster_z_max)
                        })
                        meta['trunk_id'] = int(trunk_id)
                        meta.setdefault('local_id', int(branch_id))
                        meta['point_count'] = int(meta.get('point_count', 0) + len(cluster_indices))
                        old_min_z, old_max_z = meta.get('z_span', (cluster_z_min, cluster_z_max))
                        meta['z_span'] = (min(float(old_min_z), cluster_z_min), max(float(old_max_z), cluster_z_max))

                    prev_slice_clusters = current_slice_clusters

                trunk_branch_ids = sorted(int(branch_id) for branch_id in np.unique(point_to_branch[trunk_indices]) if int(branch_id) >= 0)
                trunk_classified = int(np.sum(point_to_branch[trunk_indices] >= 0))
                total_classified_points += trunk_classified
                total_branches_detected += len(trunk_branch_ids)
                self.log_to_console(
                    f"  Trunk {trunk_id}: {len(trunk_branch_ids)} branch(es), {trunk_classified} classified point(s)"
                )
            
            # Store branch assignment
            self.branch_assignment = point_to_branch
            
            self.log_to_console(f"\n[OK] Branch detection complete!")
            self.log_to_console(f"   Total unique branches: {len([b for b in np.unique(point_to_branch) if b >= 0])}")
            self.log_to_console(f"   Total trunk-scoped branches: {total_branches_detected}")
            self.log_to_console(f"   Classified branch points: {total_classified_points}")
            self.log_to_console(f"   Noise points: {np.sum(point_to_branch == -1)}")
            
            # Store branch assignment in memory for visualization and export
            # (LAS field modification may be handled during save operation)
            self.log_to_console(f"[OK] Branch assignment stored in memory for visualization")

            # ── Merge overlapping branches within each trunk ──
            point_to_branch, merge_report = self._merge_overlapping_branches(point_to_branch)
            self.branch_assignment = point_to_branch  # update with merged branch IDs
            self.log_to_console(merge_report)
            
            # Enable selection and volume tools now that branch detection exists
            try:
                if hasattr(self, 'select_all_points_button'):
                    self.select_all_points_button.setEnabled(True)
                if hasattr(self, 'volume_calc_button'):
                    self.volume_calc_button.setEnabled(True)
            except Exception:
                pass
            
            # Visualize branches with different colors
            self.visualize_branches_from_assignment()
            # Only bump branch detection version if parameters changed since last run
            try:
                if current_branch_params is not None and current_branch_params != getattr(self, '_last_branch_detection_params', None):
                    self._last_branch_detection_params = current_branch_params
                    try:
                        self._branch_detection_version = int(getattr(self, '_branch_detection_version', 0)) + 1
                    except Exception:
                        self._branch_detection_version = getattr(self, '_branch_detection_version', 0) + 1
                    self.log_to_console(f" Branch detection parameters changed; version updated: {self._branch_detection_version}")
                else:
                    self.log_to_console("ℹ️ Branch detection run used same parameters as last run; detection version unchanged.")
            except Exception:
                try:
                    self._branch_detection_version = int(getattr(self, '_branch_detection_version', 0)) + 1
                except Exception:
                    self._branch_detection_version = getattr(self, '_branch_detection_version', 0) + 1
                self.log_to_console(f" Branch detection version updated (fallback): {self._branch_detection_version}")
            
        except Exception as e:
            self.log_to_console(f"[Error] Error detecting branch clusters: {str(e)}")
            import traceback
            self.log_to_console(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to detect branches:\n{str(e)}")


    def visualize_branches_from_assignment(self):
        """Visualize the branch assignment using colors from branch_cls."""
        if not hasattr(self, 'branch_assignment'):
            self.log_to_console("[WARN] No branch assignment available")
            return
        
        try:
            self.log_to_console(f"\nColor Visualizing branches...")
            
            # Clear current visualization
            self.plotter.clear()
            self._restore_gdb_layers()
            
            # Get unique branch IDs
            unique_branches = np.unique(self.branch_assignment)
            unique_branches = unique_branches[unique_branches >= 0]  # Exclude noise (-1)
            
            num_branches = len(unique_branches)
            self.log_to_console(f"   Rendering {num_branches} branches...")
            
            # Create color palette for branches
            import matplotlib.cm as cm
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if num_branches <= 10:
                    cmap = plt.get_cmap('tab10')
                elif num_branches <= 20:
                    cmap = plt.get_cmap('tab20')
                else:
                    cmap = plt.get_cmap('hsv')

            colors = cmap(np.linspace(0, 1, num_branches))

            # Calculate branch statistics for console output
            self.log_to_console(f"\n[Summary] BRANCH DETAILS:")
            self.log_to_console(f"{'Branch ID':<12} {'Trunk':<8} {'Local':<8} {'Size':<8} {'Min Z':<8} {'Max Z':<8} {'Height':<8}")
            self.log_to_console(f"-" * 74)
            
            branch_info = {}
            
            # Add each branch with unique color and information
            for i, branch_id in enumerate(unique_branches):
                mask = self.branch_assignment == branch_id
                branch_points = self.current_tree_points[mask]
                
                if len(branch_points) > 0:
                    # Calculate statistics
                    size = len(branch_points)
                    min_z = np.min(branch_points[:, 2])
                    max_z = np.max(branch_points[:, 2])
                    height = max_z - min_z
                    centroid = np.mean(branch_points, axis=0)
                    branch_meta = self.branch_meta.get(int(branch_id), {}) if hasattr(self, 'branch_meta') else {}
                    trunk_id = branch_meta.get('trunk_id', '?')
                    local_id = branch_meta.get('local_id', branch_id)
                    
                    # Store for labeling
                    branch_info[branch_id] = {
                        'size': size,
                        'min_z': min_z,
                        'max_z': max_z,
                        'height': height,
                        'centroid': centroid
                    }
                    
                    # Log statistics
                    self.log_to_console(
                        f"{branch_id:<12} {trunk_id:<8} {local_id:<8} {size:<8} {min_z:<8.2f} {max_z:<8.2f} {height:<8.2f}"
                    )
                    
                    # Add points with color
                    color = colors[i % len(colors)]
                    rgb_color = [float(color[0]), float(color[1]), float(color[2])]
                    
                    self.plotter.add_points(
                        branch_points,
                        color=rgb_color,
                        point_size=5,
                        name=f'Branch_{branch_id}'
                    )
                    
                    # Add label at branch centroid
                    try:
                        if trunk_id == '?':
                            label_text = f'B{branch_id}'
                        else:
                            label_text = f'T{trunk_id}.B{local_id}'
                        self.plotter.add_point_labels(
                            np.array([centroid]),
                            [label_text],
                            font_size=12,
                            text_color='white',
                            always_visible=True,
                            name=f'Label_Branch_{branch_id}'
                        )
                    except Exception as label_error:
                        self.log_to_console(f"  (Label for branch {branch_id} skipped: {str(label_error)})")
            
            # Add statistics line
            self.log_to_console(f"-" * 66)
            
            # Add noise points in gray if any
            noise_mask = self.branch_assignment == -1
            if np.any(noise_mask):
                noise_points = self.current_tree_points[noise_mask]
                self.plotter.add_points(
                    noise_points,
                    color=[0.5, 0.5, 0.5],
                    point_size=3,
                    name='Noise'
                )
                self.log_to_console(f"{'Noise':<12} {len(noise_points):<8} (filtered points)")
            
            # Summary statistics
            self.log_to_console(f"\n[Metrics] SUMMARY:")
            total_classified = np.sum(self.branch_assignment >= 0)
            total_noise = np.sum(self.branch_assignment == -1)
            total_points = len(self.branch_assignment)
            
            self.log_to_console(f"  Total points classified: {total_classified} ({100*total_classified/total_points:.1f}%)")
            self.log_to_console(f"  Noise points: {total_noise} ({100*total_noise/total_points:.1f}%)")
            self.log_to_console(f"  Unique branches: {num_branches}")
            
            if branch_info:
                sizes = [b['size'] for b in branch_info.values()]
                self.log_to_console(f"  Branch size range: {min(sizes)}-{max(sizes)} points")
                self.log_to_console(f"  Average branch size: {np.mean(sizes):.1f} points")

            # ── Vertical proximity analysis per trunk ──
            self._analyze_branch_vertical_proximity()

            # Update and display
            self.plotter.reset_camera()
            self.plotter.update()
            
            self.log_to_console(f"\n[OK] Branch visualization complete!")
            self.log_to_console(f"   (Branch labels shown as 'B#' in 3D view)\n")
            
        except Exception as e:
            self.log_to_console(f"[Error] Error visualizing branches: {str(e)}")
            import traceback
            self.log_to_console(traceback.format_exc())


    def _merge_overlapping_branches(self, point_to_branch: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Merge branches within each trunk that have overlapping Z ranges.
        Two branches overlap if: next.Z_min < current.Z_max (negative vertical gap).
        
        Returns:
            (updated_point_to_branch, merge_report_string)
        """
        if not hasattr(self, 'branch_meta') or not self.branch_meta:
            return point_to_branch, ""

        # Group branches by trunk with their Z spans
        trunk_branches: Dict[int, List[Dict[str, Any]]] = {}
        for branch_id, meta in self.branch_meta.items():
            trunk_id = meta.get('trunk_id', '?')
            if trunk_id == '?':
                continue
            trunk_id = int(trunk_id)
            z_min, z_max = meta.get('z_span', (0.0, 0.0))
            trunk_branches.setdefault(trunk_id, []).append({
                'branch_id': int(branch_id),
                'z_min': float(z_min),
                'z_max': float(z_max),
            })

        # Build remapping: old_branch_id -> merged_into_branch_id
        remap: Dict[int, int] = {}
        total_merged = 0
        merge_details: List[str] = []

        for trunk_id in sorted(trunk_branches.keys()):
            branches = trunk_branches[trunk_id]
            if len(branches) < 2:
                continue

            # Sort by Z_min
            branches.sort(key=lambda b: b['z_min'])

            # Walk through and find overlapping groups
            i = 0
            while i < len(branches):
                group = [branches[i]]
                j = i + 1
                while j < len(branches):
                    # Check if branch[j] overlaps with the combined group's Z_max
                    group_z_max = max(b['z_max'] for b in group)
                    if branches[j]['z_min'] < group_z_max:  # Overlap
                        group.append(branches[j])
                        j += 1
                    else:
                        break

                if len(group) >= 2:
                    # Merge group: keep the first branch_id, remap others
                    keeper_id = group[0]['branch_id']
                    merged_ids = []
                    for br in group[1:]:
                        remap[br['branch_id']] = keeper_id
                        merged_ids.append(br['branch_id'])

                    # Update meta: merge point counts and Z spans
                    if keeper_id in self.branch_meta:
                        keeper_meta = self.branch_meta[keeper_id]
                        for br in group[1:]:
                            if br['branch_id'] in self.branch_meta:
                                old_meta = self.branch_meta[br['branch_id']]
                                keeper_meta['point_count'] += old_meta.get('point_count', 0)
                                old_zmin, old_zmax = keeper_meta.get('z_span', (float('inf'), float('-inf')))
                                new_zmin = min(old_zmin, old_meta.get('z_span', (float('inf'), float('-inf')))[0])
                                new_zmax = max(old_zmax, old_meta.get('z_span', (float('-inf'), float('inf')))[1])
                                keeper_meta['z_span'] = (new_zmin, new_zmax)
                                # Delete the merged branch's meta
                                del self.branch_meta[br['branch_id']]

                    total_merged += len(merged_ids)
                    z_range = f"{group[0]['z_min']:.2f}–{group_z_max:.2f}m"
                    merged_str = ", ".join(f"B{b}" for b in merged_ids)
                    merge_details.append(f"     T{trunk_id}: B{keeper_id} ← merged {merged_str} | Z range: {z_range}")

                i = j  # Move to next unprocessed branch

        # Apply remapping to point_to_branch
        if remap:
            remap_array = np.full(point_to_branch.max() + 1, -1, dtype=int)
            for old_id, new_id in remap.items():
                remap_array[old_id] = new_id
            # Fill self-references
            unique_ids = np.unique(point_to_branch)
            for bid in unique_ids:
                if bid >= 0 and bid < len(remap_array) and remap_array[bid] == -1:
                    remap_array[bid] = bid

            valid_mask = point_to_branch >= 0
            point_to_branch[valid_mask] = remap_array[point_to_branch[valid_mask]]

        # Build report
        if total_merged == 0:
            return point_to_branch, ""

        report_lines = [""]
        report_lines.append(f" MERGED OVERLAPPING BRANCHES: {total_merged} branch(es) merged into existing branches")
        report_lines.extend(merge_details)
        report_lines.append(f"   Remaining unique branches after merge: {len([b for b in np.unique(point_to_branch) if b >= 0])}")
        return point_to_branch, "\n".join(report_lines)


    def _analyze_branch_vertical_proximity(self):
        """Analyze vertical gaps between branches within each trunk and print a proximity table."""
        if not hasattr(self, 'branch_meta') or not self.branch_meta:
            self.log_to_console("[WARN] No branch metadata available for vertical proximity analysis.")
            return

        # Group branches by trunk_id
        trunk_branches: Dict[int, List[Dict[str, Any]]] = {}
        for branch_id, meta in self.branch_meta.items():
            trunk_id = meta.get('trunk_id', '?')
            if trunk_id == '?':
                continue
            trunk_id = int(trunk_id)
            z_min, z_max = meta.get('z_span', (0.0, 0.0))
            trunk_branches.setdefault(trunk_id, []).append({
                'branch_id': int(branch_id),
                'local_id': meta.get('local_id', branch_id),
                'z_min': float(z_min),
                'z_max': float(z_max),
                'height': float(z_max) - float(z_min),
                'point_count': meta.get('point_count', 0),
            })

        if not trunk_branches:
            self.log_to_console("[WARN] No trunk-scoped branches found for vertical proximity analysis.")
            return

        self.log_to_console(f"\n{'='*90}")
        self.log_to_console(f"[Dim] BRANCH VERTICAL PROXIMITY ANALYSIS")
        self.log_to_console(f"{'='*90}")

        for trunk_id in sorted(trunk_branches.keys()):
            branches = trunk_branches[trunk_id]
            if len(branches) < 2:
                self.log_to_console(f"\n  Trunk T{trunk_id}: only 1 branch — no gaps to compare.")
                continue

            # Sort by min Z (bottom of branch)
            branches.sort(key=lambda b: b['z_min'])

            self.log_to_console(f"\n  ┌─ Trunk T{trunk_id} ({len(branches)} branches) ───────────────────────────────┐")
            header = f"  │ {'Branch':<10} {'Z Min (m)':>10} {'Z Max (m)':>10} {'Height (m)':>10} {'Gap Below (m)':>14} {'Status':<16} │"
            self.log_to_console(header)
            self.log_to_console(f"  │{'-' * 86}│")

            for i, br in enumerate(branches):
                if i == 0:
                    # First (lowest) branch — gap to ground
                    gap_below = br['z_min']  # Distance from ground (min Z of branch) - we'd need tree base
                    # Get tree base from current_tree_points if available
                    tree_base = None
                    if hasattr(self, 'current_tree_points') and self.current_tree_points is not None:
                        tree_base = float(np.min(self.current_tree_points[:, 2]))
                    if tree_base is not None:
                        gap_below = br['z_min'] - tree_base
                        status = "↳ from tree base"
                    else:
                        status = "↳ lowest branch"
                else:
                    # Gap between this branch's bottom and previous branch's top
                    gap_below = br['z_min'] - branches[i - 1]['z_max']
                    if gap_below <= 0.05:
                        status = "[WARN] overlapping"
                    elif gap_below < 0.15:
                        status = "● very close"
                    elif gap_below < 0.30:
                        status = "● close"
                    elif gap_below < 0.50:
                        status = "◐ moderate"
                    else:
                        status = "○ far apart"

                branch_label = f"T{trunk_id}.B{br['local_id']}"
                self.log_to_console(
                    f"  │ {branch_label:<10} {br['z_min']:>10.2f} {br['z_max']:>10.2f} "
                    f"{br['height']:>10.2f} {gap_below:>14.2f} {status:<16} │"
                )

            # Summary stats for this trunk
            gaps = []
            for i in range(1, len(branches)):
                gaps.append(branches[i]['z_min'] - branches[i - 1]['z_max'])
            if gaps:
                min_gap = min(gaps)
                max_gap = max(gaps)
                avg_gap = np.mean(gaps)
                close_count = sum(1 for g in gaps if g < 0.15)
                overlap_count = sum(1 for g in gaps if g <= 0.05)
                self.log_to_console(f"  └{'─' * 86}┘")
                self.log_to_console(f"     Total gaps: {len(gaps)} | Avg: {avg_gap:.2f}m | Min: {min_gap:.2f}m | Max: {max_gap:.2f}m")
                if close_count > 0:
                    self.log_to_console(f"     [WARN] {close_count} branch pair(s) vertically close (<15cm)")
                if overlap_count > 0:
                    self.log_to_console(f"      {overlap_count} branch pair(s) overlapping vertically (≤5cm)")

        self.log_to_console(f"{'='*90}\n")


