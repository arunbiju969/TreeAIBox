"""Volume calculation, accumulation, save/load, summaries. (TreeVisualizerGUI mixin)."""

import os
import sys
import glob
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QWidget, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


class VolumeMixin:
    """Volume calculation, accumulation, save/load, summaries."""

    def select_volume_folder(self):
        """Allow user to pre-select a folder for batch volume saving."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder for Volume Saves",
            ""
        )
        if folder:
            self.volume_folder_path = folder
            self.log_to_console(f" Volume folder selected: {folder}")
            QMessageBox.information(self, "Success", f"Volume folder set to:\n{folder}")
        else:
            self.log_to_console("ℹ️ No folder selected")


    def calculate_volume_from_selection(self):
        """
        Perform circle fitting on selected points for volume calculation.
        OVERLAYS results on existing tree visualization - does NOT clear the view.
        Accumulates results from multiple selections for comprehensive volume analysis.
        """
        if not hasattr(self, 'current_tree_points') or self.current_tree_points is None:
            QMessageBox.warning(self, "No Tree Points", "No tree points available. Please visualize a tree first.")
            return

        if self.selected_point_indices is None or len(self.selected_point_indices) == 0:
            n_points = len(self.current_tree_points)
            has_branch_detection = (
                hasattr(self, 'branch_assignment') and
                self.branch_assignment is not None and
                len(self.branch_assignment) == n_points
            )
            has_trunk_detection = (
                hasattr(self, 'trunk_assignment') and
                self.trunk_assignment is not None and
                len(self.trunk_assignment) == n_points
            )

            if has_branch_detection or has_trunk_detection:
                self.selected_point_indices = np.arange(n_points, dtype=int)
                mode_msg = "branches/trunks" if has_branch_detection and has_trunk_detection else ("branches" if has_branch_detection else "trunks")
                self.log_to_console(f"ℹ️ No manual selection found. Using all {n_points} current tree points because {mode_msg} are already detected.")
            else:
                QMessageBox.warning(self, "No Selection", "No points selected. Please use the polygon selection tool first.")
                return

        # Validate selected indices are within the range of current visualization points.
        # If they appear to be original LAS indices (out-of-range), try remapping to visualization indices.
        try:
            sel = np.asarray(self.selected_point_indices)
            if sel.size > 0 and len(self.current_tree_points) > 0 and np.max(sel) >= len(self.current_tree_points):
                # Attempt to build mapping from original LAS indices -> visualization indices
                global_indices = None
                try:
                    if hasattr(self, 'all_masks') and self.all_masks is not None and len(self.all_masks) > 0:
                        per_mask_indices = [np.where(mask)[0] for mask in self.all_masks]
                        global_indices = np.concatenate(per_mask_indices)
                    elif hasattr(self, 'current_mask') and self.current_mask is not None and len(self.current_mask) == len(self.las_data.points):
                        global_indices = np.where(self.current_mask)[0]
                except Exception:
                    global_indices = None

                if global_indices is not None:
                    mapping = {int(orig): int(viz) for viz, orig in enumerate(global_indices)}
                    remapped = [mapping[int(x)] for x in sel if int(x) in mapping]
                    if len(remapped) == 0:
                        QMessageBox.warning(self, "Selection Index Error", "Selected indices appear out of range and could not be remapped to the current view. Please re-select points.")
                        return
                    self.selected_point_indices = np.array(remapped, dtype=int)
                    self.log_to_console(" Remapped selection indices from original LAS indices to current visualization indices.")
                else:
                    QMessageBox.warning(self, "Selection Index Error", "Selected indices are out of range for the current view and no index mapping is available. Please re-select points.")
                    return
        except Exception as e:
            # In case of any unexpected failure during remapping, abort gracefully
            self.log_to_console(f"[WARN] Error validating selection indices: {e}")
            QMessageBox.warning(self, "Selection Index Error", f"Error validating selection indices: {e}")
            return

        try:
            # Only calculate volume if trunk/branch detection state has changed since last calculation
            current_sig = (getattr(self, 'current_tree_id', None),
                           getattr(self, '_trunk_detection_version', 0),
                           getattr(self, '_branch_detection_version', 0))
            if getattr(self, '_last_volume_calc_signature', None) == current_sig:
                self.log_to_console("ℹ️ Volume calculation skipped: no change in trunk/branch detection since last calculation.")
                return

            volume_groups, is_branch_mode = self._get_volume_groups_from_selection()
            if not volume_groups:
                QMessageBox.warning(self, "No Valid Groups", "No valid point groups found for volume calculation.")
                return

            # Start a new overlay run so every section in this run gets unique actor names.
            self._volume_calc_run_id = int(getattr(self, '_volume_calc_run_id', 0)) + 1
            current_run_id = self._volume_calc_run_id
            self.log_to_console(f" Starting volume overlay run {current_run_id}")
            
            # Determine tree_id for this section based on selected points
            if hasattr(self, 'current_tree_ids') and self.current_tree_ids is not None:
                selected_tree_ids = self.current_tree_ids[self.selected_point_indices]
                # Use the tree ID that appears most frequently in the selection
                unique_ids, counts = np.unique(selected_tree_ids, return_counts=True)
                section_tree_id = unique_ids[np.argmax(counts)]
                
                # Log if points from multiple trees were selected
                if len(unique_ids) > 1:
                    self.log_to_console(f"[WARN] Selected points from {len(unique_ids)} trees, assigning section to tree {section_tree_id} (most points)")
            else:
                # Fallback to current_tree_id
                section_tree_id = self.current_tree_id
            
            # Remove existing accumulated sections for this tree so each successful run is fresh
            try:
                prev_count = len(self.accumulated_volume_sections)
                self.accumulated_volume_sections = [s for s in self.accumulated_volume_sections if str(s.get('tree_id', '')) != str(section_tree_id)]
                removed = prev_count - len(self.accumulated_volume_sections)
                if removed > 0:
                    self.log_to_console(f"[Clean] Cleared {removed} previous section(s) for tree {section_tree_id} to start a fresh volume calculation")
            except Exception:
                pass

            # Start section numbering fresh for this tree
            section_id = 1

            total_selected_points = sum(len(group_points) for _, _, group_points in volume_groups)
            if is_branch_mode:
                self.log_to_console(
                    f"[Measure] Branch-wise volume mode: {len(volume_groups)} branches, {total_selected_points} selected points (tree {section_tree_id})"
                )
            else:
                self.log_to_console(
                    f"[Measure] Standard volume mode: section {section_id}, {total_selected_points} selected points (tree {section_tree_id})"
                )

            # Debug: Log accumulated sections count
            self.log_to_console(f" Current accumulated sections before adding: {len(self.accumulated_volume_sections)}")

            # Define colors for different sections (cycling through a palette)
            section_colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'purple', 'brown']
            point_color = section_colors[(section_id - 1) % len(section_colors)]
            circle_color = section_colors[(section_id - 1) % len(section_colors)]
            avg_color = 'red'  # Keep average circles red for consistency

            last_section_id = None
            last_section_points = 0
            last_section_volume = 0.0
            last_section_data = None
            section_metric_values_cm = []

            if self.plotter is not None:
                # NEVER clear the plotter - overlay volume results on existing tree visualization
                # This preserves the tree visualization while adding volume analysis

                # Suppress rendering during batch additions for performance
                self.plotter.suppress_rendering = True

                for group_branch_id, _, group_points in volume_groups:
                    point_color = section_colors[(section_id - 1) % len(section_colors)]
                    circle_color = section_colors[(section_id - 1) % len(section_colors)]
                    branch_suffix = f" | B{group_branch_id}" if group_branch_id is not None else ""
                    branch_trunk_id = None
                    if is_branch_mode and group_branch_id is not None and hasattr(self, 'branch_meta'):
                        branch_meta = self.branch_meta.get(int(group_branch_id), {})
                        branch_trunk_id = branch_meta.get('trunk_id')

                    # Show selected points for this section
                    selected_cloud = pv.PolyData(group_points)
                    self.plotter.add_points(
                        selected_cloud,
                        color=point_color,
                        point_size=4,
                        opacity=0.6,
                        name=f'volume_section_{section_id}_points',
                        label=f'Section {section_id}{branch_suffix} Points ({len(group_points)} pts)',
                        pickable=False
                    )

                    # Add circle fitting to selected points with section-specific naming
                    measurement_label = "Diameter" if is_branch_mode else "DBH"
                    section_data = self._add_dbh_circle_fitting_for_section(
                        group_points,
                        section_id,
                        circle_color,
                        avg_color,
                        measurement_label=measurement_label,
                        run_id=current_run_id
                    )

                    # Calculate volume for this section
                    section_volume = self._calculate_section_volume(section_data)

                    # Store this section's data including volume
                    section_info = {
                        'tree_id': section_tree_id,
                        'section_id': section_id,
                        'trunk_id': int(branch_trunk_id) if branch_trunk_id is not None else None,
                        'branch_id': int(group_branch_id) if group_branch_id is not None else None,
                        'points': group_points.copy(),
                        'point_color': point_color,
                        'circle_color': circle_color,
                        'volume': section_volume,
                        'circle_data': section_data
                    }

                    # Avoid appending duplicate sections when re-running calculation without changes.
                    # Consider duplicate if tree_id, trunk_id, branch_id match and
                    # point count, centroid and volume are effectively identical.
                    is_dup = False
                    try:
                        new_n = len(section_info['points'])
                        new_centroid = np.mean(section_info['points'], axis=0) if new_n > 0 else np.array([0.0, 0.0, 0.0])
                        new_vol = float(section_info.get('volume', 0.0) or 0.0)

                        for ex in self.accumulated_volume_sections:
                            try:
                                if ex.get('tree_id') != section_info.get('tree_id'):
                                    continue
                                if ex.get('trunk_id') != section_info.get('trunk_id'):
                                    continue
                                if ex.get('branch_id') != section_info.get('branch_id'):
                                    continue

                                ex_n = len(ex.get('points') or [])
                                if ex_n != new_n:
                                    continue

                                ex_centroid = np.mean(ex.get('points'), axis=0) if ex_n > 0 else np.array([0.0, 0.0, 0.0])
                                if np.linalg.norm(ex_centroid - new_centroid) > 1e-3:
                                    continue

                                ex_vol = float(ex.get('volume', 0.0) or 0.0)
                                if abs(ex_vol - new_vol) > 1e-6:
                                    continue

                                # All checks passed — treat as duplicate
                                is_dup = True
                                break
                            except Exception:
                                continue
                    except Exception:
                        is_dup = False

                    if is_dup:
                        self.log_to_console(f" Skipped duplicate section {section_id} for tree {section_tree_id} (no changes detected)")
                    else:
                        self.accumulated_volume_sections.append(section_info)

                    last_section_id = section_id
                    last_section_points = len(group_points)
                    last_section_volume = section_volume
                    last_section_data = section_data

                    if group_branch_id is not None:
                        self.log_to_console(
                            f"[Measure] Section {section_id} (Branch B{group_branch_id}): {len(group_points)} pts, {section_volume:.4f} m³"
                        )
                    else:
                        self.log_to_console(
                            f"[Measure] Section {section_id}: {len(group_points)} pts, {section_volume:.4f} m³"
                        )

                    if is_branch_mode:
                        avg_radius = section_data.get('avg_radius') if section_data else None
                        diameter_cm = (avg_radius * 2 * 100) if avg_radius else None
                        branch_label = f"B{group_branch_id}" if group_branch_id is not None else "N/A"
                        metric_text = f"{diameter_cm:.1f} cm" if diameter_cm is not None else "N/A"
                        self.log_to_console(
                            f"   ↳ Diameter (section avg) | Section {section_id} | Branch {branch_label}: {metric_text}"
                        )
                    else:
                        dbh_at_1_3m = section_data.get('dbh_at_1_3m') if section_data else None
                        dbh_cm = (dbh_at_1_3m * 100) if dbh_at_1_3m else None
                        if dbh_cm is not None:
                            section_metric_values_cm.append(dbh_cm)
                        metric_text = f"{dbh_cm:.1f} cm" if dbh_cm is not None else "N/A"
                        self.log_to_console(
                            f"   ↳ DBH @{getattr(self, 'dbh_reference_height', 1.3):.1f}m | Section {section_id}: {metric_text}"
                        )

                    section_id += 1

                # Re-enable rendering after batch additions
                self.plotter.suppress_rendering = False
                self.plotter.render()

                # Perform global ground proximity check in all modes.
                # In branch mode this helps DBH estimation when a trunk misses the exact DBH plane.
                self._check_global_ground_proximity_and_extrapolate()

                # Debug: Log accumulated sections count after adding
                self.log_to_console(f"[OK] Total accumulated sections now: {len(self.accumulated_volume_sections)}")

                # Build and emit trunk summary table for the current accumulated results
                trunk_summary, trunk_summary_lines, trunk_total_volume = self._build_trunk_volume_summary_table()
                trunk_dbh_summary, trunk_dbh_lines, trunk_dbh_details = self._build_trunk_dbh_summary_table() if is_branch_mode else ({}, [], {})

                if is_branch_mode and trunk_dbh_details:
                    self._add_trunk_dbh_overlays(trunk_dbh_details)

                if trunk_summary:
                    self.log_to_console(f"[Box] Total trunks used for volume: {len(trunk_summary)}")
                    self.log_to_console("[Summary] Trunk volume summary:")
                    for line in trunk_summary_lines:
                        self.log_to_console(line)
                        print(line)
                    self.log_to_console(f"[Metrics] Combined trunk volume: {trunk_total_volume:.4f} m³")
                    print(f"Combined trunk volume: {trunk_total_volume:.4f} m³")
                else:
                    self.log_to_console("[Box] Total trunks used for volume: 0")
                    self.log_to_console("[Summary] Trunk volume summary: none")
                    print("Total trunks used for volume: 0")
                    print("Trunk volume summary: none")

                if is_branch_mode:
                    if trunk_dbh_summary:
                        dbh_msg = f"[Dim] Trunk DBH summary ({getattr(self, 'dbh_reference_height', 1.3):.1f}m above base):"
                        self.log_to_console(dbh_msg)
                        for line in trunk_dbh_lines:
                            self.log_to_console(line)
                            print(line)
                    else:
                        dbh_msg = f"[Dim] Trunk DBH summary ({getattr(self, 'dbh_reference_height', 1.3):.1f}m above base): none"
                        self.log_to_console(dbh_msg)

                # Update info text to show accumulated results
                total_sections = len(self.accumulated_volume_sections)
                total_points = sum(len(section['points']) for section in self.accumulated_volume_sections)
                total_volume = sum(section['volume'] for section in self.accumulated_volume_sections)

                # Show measurement text based on mode
                dbh_h = getattr(self, 'dbh_reference_height', 1.3)
                if is_branch_mode:
                    if trunk_dbh_summary:
                        dbh_lines = []
                        for key, dbh in sorted(trunk_dbh_summary.items()):
                            parts = str(key).rsplit('_', 1)
                            if len(parts) == 2:
                                tree_id, trunk_id = parts[0], parts[1]
                                dbh_lines.append(f"Tree{tree_id}/T{trunk_id}: {dbh:.1f} cm")
                            else:
                                dbh_lines.append(f"T{key}: {dbh:.1f} cm")
                        dbh_text = "\nDBH by trunk: " + "; ".join(dbh_lines)
                    else:
                        dbh_text = "\nDBH by trunk: N/A"
                else:
                    dbh_at_1_3m = last_section_data.get('dbh_at_1_3m') if last_section_data else None
                    dbh_text = f"\nLast DBH @{dbh_h:.1f}m: {dbh_at_1_3m*100:.1f} cm" if dbh_at_1_3m else f"\nLast DBH @{dbh_h:.1f}m: N/A"

                metric_summary_text = ""
                if is_branch_mode and trunk_dbh_summary:
                    metric_summary_text = (
                        f"\nDBH @{dbh_h:.1f}m (all trunks): min {np.min(list(trunk_dbh_summary.values())):.1f}, "
                        f"max {np.max(list(trunk_dbh_summary.values())):.1f}, avg {np.mean(list(trunk_dbh_summary.values())):.1f} cm"
                    )
                elif section_metric_values_cm:
                    metric_name = f"DBH @{dbh_h:.1f}m (all sections)" if not is_branch_mode else "Diameter (all sections)"
                    metric_summary_text = (
                        f"\n{metric_name}: min {np.min(section_metric_values_cm):.1f}, "
                        f"max {np.max(section_metric_values_cm):.1f}, avg {np.mean(section_metric_values_cm):.1f} cm"
                    )

                mode_text = "Mode: Branch-wise" if is_branch_mode else "Mode: Standard"

                trunk_summary_overlay = ""
                if trunk_summary:
                    compact_rows = []
                    for tid in sorted(trunk_summary.keys()):
                        data = trunk_summary[tid]
                        compact_rows.append(f"T{tid}: {data['total_volume']:.4f} m³ / {data['num_branches']} branches")
                    trunk_summary_overlay = "\nTrunks Used: " + str(len(trunk_summary)) + "\n" + "\n".join(compact_rows)

                info_text = (f"Accumulated Volume Calculation\n"
                            f"{mode_text}\n"
                            f"Sections: {total_sections}\n"
                            f"Total Points: {total_points}\n"
                            f"Combined Trunk Volume: {total_volume:.4f} m³\n"
                            f"Trunks Used: {len(trunk_summary) if trunk_summary else 0}"
                            f"{trunk_summary_overlay}"
                            f"{metric_summary_text}"
                            f"{dbh_text}\n"
                            f"Last Section: {last_section_id} ({last_section_points} pts, {last_section_volume:.4f} m³)")

                # Remove previous info text if it exists
                try:
                    self.plotter.remove_actor('volume_info_text')
                except:
                    pass

                self.plotter.add_text(info_text, position='upper_left', font_size=10, color='#FFFFFF',
                                    name='volume_info_text')
                self._set_volume_overlay_actors_unpickable()

                # Ensure plotter updates to show all accumulated sections
                self.plotter.update()
                # NOTE: Don't reset_camera() here - preserve user's current view of the tree

                # Mark that volume has been calculated for this detection state
                try:
                    self._last_volume_calc_signature = (
                        getattr(self, 'current_tree_id', None),
                        getattr(self, '_trunk_detection_version', 0),
                        getattr(self, '_branch_detection_version', 0)
                    )
                    self.log_to_console(f"[OK] Recorded volume calculation state: tree={self._last_volume_calc_signature[0]}, trunk_v={self._last_volume_calc_signature[1]}, branch_v={self._last_volume_calc_signature[2]}")
                except Exception:
                    pass

            # Enable clear button now that we have results
            self.clear_volume_button.setEnabled(True)
            self.save_volume_button.setEnabled(True)
            self.export_trunks_button.setEnabled(True)

        except Exception as e:
            QMessageBox.warning(self, "Volume Calculation Error", f"Error performing volume calculation: {str(e)}")
            print(f"Volume calculation error: {e}")


    def _get_volume_groups_from_selection(self):
        """Return grouped points for volume calculation as (branch_id, indices, points)."""
        selected_indices = np.asarray(self.selected_point_indices)
        selected_points = self.current_tree_points[selected_indices]

        # Default behavior: all selected points as one section
        default_group = [(None, selected_indices, selected_points)]

        if not hasattr(self, 'branch_assignment') or self.branch_assignment is None:
            # Ensure no trunk grouping leftover
            self.volume_groups_by_trunk = {}
            return default_group, False

        if len(self.branch_assignment) != len(self.current_tree_points):
            self.log_to_console("[WARN] Branch assignment size mismatch with current tree points. Falling back to standard volume mode.")
            return default_group, False

        selected_branch_ids = self.branch_assignment[selected_indices]
        unique_branch_ids = sorted(int(branch_id) for branch_id in np.unique(selected_branch_ids) if int(branch_id) >= 0)

        if not unique_branch_ids:
            self.log_to_console("[WARN] No valid branch IDs in current selection. Falling back to standard volume mode.")
            return default_group, False

        grouped = []
        for branch_id in unique_branch_ids:
            branch_mask = selected_branch_ids == branch_id
            branch_indices = selected_indices[branch_mask]
            if len(branch_indices) < 3:
                continue
            branch_points = self.current_tree_points[branch_indices]
            grouped.append((branch_id, branch_indices, branch_points))

        if not grouped:
            self.log_to_console("[WARN] Branch groups were too small for fitting. Falling back to standard volume mode.")
            self.volume_groups_by_trunk = {}
            return default_group, False

        # Build trunk-scoped grouping for downstream per-trunk aggregation
        try:
            self.volume_groups_by_trunk = {}
            for branch_id, branch_indices, branch_points in grouped:
                trunk_id = None
                # Prefer explicit branch_meta mapping if available
                if hasattr(self, 'branch_meta') and branch_id in self.branch_meta:
                    trunk_id = int(self.branch_meta[branch_id].get('trunk_id', -1))
                else:
                    # Fallback: infer trunk by majority vote of trunk_assignment within branch indices
                    if hasattr(self, 'trunk_assignment') and self.trunk_assignment is not None:
                        trunks_in_branch = self.trunk_assignment[branch_indices]
                        valid = trunks_in_branch[trunks_in_branch >= 0]
                        if len(valid) > 0:
                            vals, counts = np.unique(valid, return_counts=True)
                            trunk_id = int(vals[np.argmax(counts)])
                if trunk_id is None or trunk_id < 0:
                    trunk_id = -1

                self.volume_groups_by_trunk.setdefault(int(trunk_id), []).append((branch_id, branch_indices, branch_points))
        except Exception:
            self.volume_groups_by_trunk = {}

        return grouped, True


    def _build_trunk_volume_summary_table(self):
        """Build trunk volume totals and formatted console rows from accumulated section data."""
        trunk_summary = {}

        # Prefer explicit trunk ids saved in sections; fall back to section grouping if needed.
        for section in self.accumulated_volume_sections:
            trunk_id = section.get('trunk_id', None)
            branch_id = section.get('branch_id', None)
            volume = float(section.get('volume', 0.0) or 0.0)
            # Use the saved point count when available (reloaded data has an
            # empty points array but retains num_points from the JSON).
            _saved_num_points = section.get('num_points')
            num_points = int(_saved_num_points) if _saved_num_points is not None else len(section.get('points', []))

            if trunk_id is None:
                # If no trunk id is present, skip from the trunk table rather than inventing one.
                continue

            trunk_key = int(trunk_id)
            rec = trunk_summary.setdefault(trunk_key, {
                'total_volume': 0.0,
                'num_branches': 0,
                'total_points': 0,
                'branch_ids': set()
            })
            rec['total_volume'] += volume
            rec['total_points'] += num_points
            if branch_id is not None and branch_id not in rec['branch_ids']:
                rec['branch_ids'].add(int(branch_id))
                rec['num_branches'] = len(rec['branch_ids'])

        # Convert to display-friendly structure and format rows
        total_volume = sum(rec['total_volume'] for rec in trunk_summary.values())
        rows = []
        header = f"{'Trunk':<8} {'Volume_m3':>12} {'#Branches':>10} {'#Points':>10}"
        rows.append(header)
        rows.append("-" * len(header))

        for trunk_id in sorted(trunk_summary.keys()):
            rec = trunk_summary[trunk_id]
            row = f"{trunk_id:<8} {rec['total_volume']:12.4f} {rec['num_branches']:10d} {rec['total_points']:10d}"
            rows.append(row)

        return trunk_summary, rows, total_volume


    def _to_local_branch_id(self, global_branch_id):
        """Convert global branch ID to local (per-trunk) branch ID using branch_meta."""
        if global_branch_id is None or not hasattr(self, 'branch_meta') or not self.branch_meta:
            return global_branch_id
        try:
            meta = self.branch_meta.get(int(global_branch_id), {})
            return meta.get('local_id', global_branch_id)
        except (ValueError, TypeError):
            return global_branch_id


    def _build_trunk_dbh_summary_table(self):
        """Build trunk-level DBH totals using section-fitted circles (same pipeline as volume sections)."""
        trunk_dbh: Dict[int, Dict[str, Any]] = {}
        target_height = getattr(self, 'dbh_reference_height', 1.3)

        if hasattr(self, 'current_tree_points') and self.current_tree_points is not None:
            tree_base_z = float(np.min(self.current_tree_points[:, 2]))
            dbh_target_z = tree_base_z + target_height
        else:
            tree_base_z = 0.0
            dbh_target_z = target_height

        debug_logs = [
            f" DBH DEBUG: target_height={target_height:.2f}m, tree_base_z={tree_base_z:.2f}m, dbh_target_z={dbh_target_z:.2f}m",
            f" DBH DEBUG: scanning {len(self.accumulated_volume_sections)} accumulated section(s)",
        ]

        # Track the closest section per trunk as a fallback when no section's
        # Z range contains dbh_target_z (e.g. no trunk points near that height).
        closest_fallback: Dict[str, Dict[str, Any]] = {}

        for section in self.accumulated_volume_sections:
            trunk_id = section.get('trunk_id', None)
            if trunk_id is None:
                continue

            circle_data = section.get('circle_data') or {}
            section_min_z = circle_data.get('min_z')
            section_max_z = circle_data.get('max_z')
            avg_radius = circle_data.get('avg_radius')
            avg_center = circle_data.get('avg_center')
            tree_id = section.get('tree_id', '?')
            branch_id = section.get('branch_id', None)
            sec_id = section.get('section_id', '?')

            avg_r_str = f"{avg_radius:.3f}" if avg_radius is not None else "None"
            z_min_str = f"{section_min_z:.2f}" if section_min_z is not None else "?"
            z_max_str = f"{section_max_z:.2f}" if section_max_z is not None else "?"

            local_branch = self._to_local_branch_id(branch_id)
            global_str = f"(global={branch_id})" if branch_id is not None and branch_id != local_branch else ""
            debug_logs.append(
                f" DBH DEBUG: Section sec={sec_id} tree={tree_id} trunk={trunk_id} branch={local_branch}{global_str} "
                f"Z=[{z_min_str}, {z_max_str}] avg_R={avg_r_str}"
            )

            # Per-section DBH target Z: prefer the tree base Z recorded when the
            # section was processed (saved in the JSON), so reloaded data
            # reproduces the original DBH reference height even when no tree is
            # currently visualized. Falls back to the saved circle_data target Z,
            # then to the global tree base Z.
            _saved_base = section.get('tree_base_z')
            if _saved_base is not None:
                try:
                    section_tree_base_z = float(_saved_base)
                except (ValueError, TypeError):
                    section_tree_base_z = tree_base_z
            else:
                _saved_target = circle_data.get('dbh_target_z')
                if _saved_target is not None:
                    try:
                        section_tree_base_z = float(_saved_target) - target_height
                    except (ValueError, TypeError):
                        section_tree_base_z = tree_base_z
                else:
                    section_tree_base_z = tree_base_z
            section_dbh_target_z = section_tree_base_z + target_height

            candidate = None

            if (avg_radius is not None and avg_center is not None and len(avg_center) >= 2
                    and section_min_z is not None and section_max_z is not None):
                try:
                    _sec_num = int(str(sec_id))
                except (ValueError, TypeError):
                    _sec_num = 0

                # Use XY of the individual circle closest to section_dbh_target_z
                # so the blue circle aligns with the red circle at that height.
                circles = circle_data.get('circles', [])
                best_xy = avg_center  # fallback to section average XY
                best_z = section_dbh_target_z  # fallback to the DBH height
                best_dist = float('inf')
                if circles and len(circles) > 0:
                    for center, _radius in circles:
                        dist = abs(center[2] - section_dbh_target_z)
                        if dist < best_dist:
                            best_dist = dist
                            best_xy = [center[0], center[1]]
                            best_z = float(center[2])

                if section_min_z <= section_dbh_target_z <= section_max_z:
                    # Trunk has points at the DBH height: draw the blue circle
                    # at the DBH height itself.
                    candidate = {
                        'tree_id': str(tree_id),
                        'dbh_cm': float(avg_radius) * 2.0 * 100.0,
                        'radius_m': float(avg_radius),
                        'center_xyz': [float(best_xy[0]), float(best_xy[1]), float(section_dbh_target_z)],
                        'score': 0.0,
                        'priority': 0,
                        'section_id': f"section_avg@{sec_id}",
                        'branch_id': self._to_local_branch_id(branch_id),
                        'dbh_target_z': float(section_dbh_target_z),
                        '_sec_num': int(_sec_num),
                    }
                    debug_logs.append(
                        f"   [OK] QUALIFIES: Z range contains dbh_target_z → candidate (priority=0, sec_num={_sec_num})"
                    )
                else:
                    # Record as a fallback candidate using the closest circle to
                    # the DBH height, so trees without trunk points near that
                    # height still get a blue DBH circle drawn. The circle is
                    # drawn at the actual closest circle's Z for realism.
                    fallback_key = f"{str(tree_id)}_{int(trunk_id)}"
                    fallback_dist = abs(section_min_z - section_dbh_target_z) if section_dbh_target_z < section_min_z else abs(section_max_z - section_dbh_target_z)
                    fallback_candidate = {
                        'tree_id': str(tree_id),
                        'dbh_cm': float(avg_radius) * 2.0 * 100.0,
                        'radius_m': float(avg_radius),
                        'center_xyz': [float(best_xy[0]), float(best_xy[1]), float(best_z)],
                        'score': float(best_dist),
                        'priority': 1,
                        'section_id': f"section_avg@{sec_id}",
                        'branch_id': self._to_local_branch_id(branch_id),
                        'dbh_target_z': float(section_dbh_target_z),
                        '_sec_num': int(_sec_num),
                        '_fallback_dist': float(fallback_dist),
                    }
                    existing_fb = closest_fallback.get(fallback_key)
                    if existing_fb is None or fallback_dist < existing_fb.get('_fallback_dist', float('inf')):
                        closest_fallback[fallback_key] = fallback_candidate
                    debug_logs.append(
                        f"   [Wait] FALLBACK candidate for {fallback_key}: Z range [{z_min_str}, {z_max_str}] does NOT contain "
                        f"dbh_target_z={section_dbh_target_z:.2f} (closest circle dist={best_dist:.3f}m)"
                    )
            elif avg_radius is None:
                debug_logs.append("   [Error] SKIPPED: no avg_radius")
            else:
                debug_logs.append("   [Error] SKIPPED: missing avg_center or Z range")

            if candidate is not None:
                tree_id_str = str(tree_id)
                composite_key = f"{tree_id_str}_{int(trunk_id)}"
                rec = trunk_dbh.get(composite_key)
                if rec is None:
                    trunk_dbh[composite_key] = candidate
                    debug_logs.append(
                        f"    SELECTED for {composite_key}: branch={branch_id}, DBH={candidate['dbh_cm']:.1f}cm, sec_num={candidate['_sec_num']}"
                    )
                else:
                    new_sort = (candidate['priority'], -candidate['_sec_num'], candidate['score'])
                    old_sort = (rec['priority'], -rec.get('_sec_num', 0), rec['score'])
                    if new_sort < old_sort:
                        trunk_dbh[composite_key] = candidate
                        debug_logs.append(
                            f"    REPLACED {composite_key}: old branch={rec.get('branch_id')} sec_num={rec.get('_sec_num')} → new branch={branch_id} sec_num={candidate['_sec_num']}"
                        )
                    else:
                        debug_logs.append(
                            f"    IGNORED for {composite_key}: existing has better sort (old_sec_num={rec.get('_sec_num')}, new_sec_num={candidate['_sec_num']})"
                        )

        # Apply closest-circle fallback for trunks that had no section containing
        # the DBH height, so a blue DBH circle is still drawn.
        for fallback_key, fallback_candidate in closest_fallback.items():
            if fallback_key not in trunk_dbh:
                trunk_dbh[fallback_key] = fallback_candidate
                debug_logs.append(
                    f"    FALLBACK SELECTED for {fallback_key}: no section at DBH height, "
                    f"using closest circle (branch={fallback_candidate.get('branch_id')}, "
                    f"DBH={fallback_candidate['dbh_cm']:.1f}cm, dist={fallback_candidate.get('_fallback_dist', 0):.3f}m)"
                )

        # No trunk-point refit fallback here by design:
        # DBH should come from section-fitted circles used by the same branch/volume pipeline.

        trunk_dbh_values = {key: rec['dbh_cm'] for key, rec in trunk_dbh.items()}
        debug_logs.append(f" DBH DEBUG: final candidates = {list(trunk_dbh.keys())}")
        for key, rec in trunk_dbh.items():
            debug_logs.append(
                f"   {key}: branch={rec.get('branch_id')} DBH={rec['dbh_cm']:.1f}cm source={rec.get('section_id')} sec_num={rec.get('_sec_num', '?')}"
            )
        self.log_many_to_console(debug_logs)
        rows = []
        header = f"{'Tree/Trunk':<16} {'DBH_cm':>12} {'Source':>24}"
        rows.append(header)
        rows.append("-" * len(header))

        for composite_key in sorted(trunk_dbh.keys()):
            rec = trunk_dbh[composite_key]
            tree_id = rec.get('tree_id', '?')
            # Parse trunk_id from composite key for display
            parts = composite_key.rsplit('_', 1)
            trunk_display = parts[-1] if len(parts) == 2 else str(composite_key)
            label = f"Tree{tree_id}/T{trunk_display}"
            rows.append(f"{label:<16} {rec['dbh_cm']:12.1f} {str(rec['section_id']):>24}")

        return trunk_dbh_values, rows, trunk_dbh


    def _add_trunk_dbh_overlays(self, trunk_dbh_details: Dict[str, Dict[str, Any]]):
        """Render trunk-level DBH circles and labels in blue for all trunks with DBH estimates.

        Circles and labels are batched into two actors. Adding one mesh + one
        label actor per trunk is much slower because each add_mesh/add_point_labels
        rebuilds VTK state and can trigger a render.
        """
        if self.plotter is None or not trunk_dbh_details:
            return

        import re as _re

        circle_specs = []
        label_points = []
        label_texts = []

        for composite_key, rec in trunk_dbh_details.items():
            center = rec.get('center_xyz')
            radius_m = rec.get('radius_m')
            dbh_cm = rec.get('dbh_cm')
            tree_id = rec.get('tree_id', '?')
            section_id = rec.get('section_id', '?')
            branch_id = rec.get('branch_id', None)

            if center is None or radius_m is None or dbh_cm is None:
                continue

            parts = composite_key.rsplit('_', 1)
            trunk_display = parts[-1] if len(parts) == 2 else str(composite_key)

            sec_display = ""
            if section_id and section_id != '?':
                m = _re.search(r'@(\d+)', str(section_id))
                if m:
                    sec_display = f"/S{m.group(1)}"
                else:
                    sec_display = f"/{section_id}"

            source_info = ""
            if branch_id is not None:
                source_info += f"/B{branch_id}"
            source_info += sec_display

            try:
                center_xyz = [float(center[0]), float(center[1]), float(center[2])]
                radius_val = float(radius_m)
                circle_specs.append((center_xyz, radius_val))
                label_points.append([center_xyz[0] + radius_val * 1.25, center_xyz[1], center_xyz[2]])
                label_texts.append(f'T{tree_id}/T{trunk_display}{source_info}: {float(dbh_cm):.1f}cm')
            except Exception as e:
                self.log_to_console(f"[WARN] Failed to prepare blue trunk DBH overlay for tree {tree_id} trunk {trunk_display}: {e}")

        if not circle_specs:
            return

        was_suppressed = bool(getattr(self.plotter, 'suppress_rendering', False))
        self.plotter.suppress_rendering = True
        try:
            merged_circles = self._build_circles_mesh(circle_specs, resolution=64)
            self.plotter.add_mesh(
                merged_circles,
                color='blue',
                opacity=0.9,
                line_width=4,
                style='wireframe',
                name='trunk_dbh_circles',
                label=f'Trunk DBH Circles ({len(circle_specs)})',
                pickable=False
            )

            if label_points:
                self.plotter.add_point_labels(
                    label_points,
                    label_texts,
                    font_size=12,
                    text_color='blue',
                    point_size=0,
                    always_visible=True,
                    name='trunk_dbh_labels',
                    pickable=False,
                    render=False
                )
        except Exception as e:
            self.log_to_console(f"[WARN] Failed to draw blue trunk DBH overlays: {e}")
        finally:
            if not was_suppressed:
                self.plotter.suppress_rendering = False
                self.plotter.render()


    def _log_section_diameter_breakdown(self):
        """Log a per-section diameter breakdown to the console (only when ≤3 trees loaded)."""
        if not self.accumulated_volume_sections:
            return
        unique_trees = set(str(s.get('tree_id', '')) for s in self.accumulated_volume_sections) - {''}
        if len(unique_trees) > 3:
            return

        self.log_to_console("=" * 60)
        self.log_to_console("[Dim] SECTION DIAMETER BREAKDOWN (section avg radius × 2)")
        self.log_to_console("=" * 60)

        # Group by tree → trunk → branch (deduplicate by (tree_id, section_id))
        tree_groups: dict = {}
        seen = set()
        for s in self.accumulated_volume_sections:
            tid = str(s.get('tree_id', '?'))
            sid = s.get('section_id', '?')
            dedup_key = (tid, str(sid))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            tid = str(s.get('tree_id', '?'))
            trid = s.get('trunk_id')
            bid = s.get('branch_id')
            sid = s.get('section_id', '?')
            cd = s.get('circle_data') or {}
            ar = cd.get('avg_radius')
            diam_cm = None if ar is None else float(ar) * 2.0 * 100.0
            zmin = cd.get('min_z')
            zmax = cd.get('max_z')
            tree_groups.setdefault(tid, {}).setdefault(trid, {}).setdefault(bid, []).append({
                'section': sid,
                'diam_cm': diam_cm,
                'zmin': zmin,
                'zmax': zmax,
            })

        for tree_id in sorted(tree_groups):
            self.log_to_console(f"\n[Tree] Tree {tree_id}")
            for trunk_id in sorted(tree_groups[tree_id], key=lambda x: (isinstance(x, str), x if x is not None else -1)):
                branches = tree_groups[tree_id][trunk_id]
                self.log_to_console(f"  └─ Trunk {trunk_id}")
                for branch_id in sorted(branches, key=lambda x: (isinstance(x, str), x if x is not None else -1)):
                    sections = branches[branch_id]
                    branch_label = f"B{branch_id}" if branch_id is not None else "(no branch)"
                    # Compute branch average diameter across its sections
                    branch_diams = [s['diam_cm'] for s in sections if s['diam_cm'] is not None]
                    branch_avg = np.mean(branch_diams) if branch_diams else None
                    avg_str = f" → branch avg: {branch_avg:.1f} cm" if branch_avg is not None else ""
                    self.log_to_console(f"     ├─ {branch_label}{avg_str}")
                    for s in sections:
                        d_str = f"{s['diam_cm']:.1f} cm" if s['diam_cm'] is not None else "N/A"
                        z_str = f"Z=[{s['zmin']:.2f}, {s['zmax']:.2f}]" if s['zmin'] is not None and s['zmax'] is not None else ""
                        self.log_to_console(f"     │   S{s['section']}: {d_str}  {z_str}")
        self.log_to_console("=" * 60)


    def _add_section_info_labels(self):
        """Add 3D labels for each accumulated volume section showing trunk/branch/section info.

        Labels are placed at the average center XY with Z at the section's min_z.
        Only rendered when the load contains 3 or fewer trees; larger loads skip
        the white section labels to keep the scene readable and responsive.
        """
        if self.plotter is None:
            return

        unique_tree_ids = set(str(s.get('tree_id', '')) for s in self.accumulated_volume_sections) - {''}
        if len(unique_tree_ids) > 3:
            self.log_to_console(
                f" Skipping section labels: {len(unique_tree_ids)} trees loaded (>3 threshold)"
            )
            return

        seen = set()
        label_points = []
        label_texts = []
        for section in self.accumulated_volume_sections:
            dedup_key = (str(section.get('tree_id', '?')), str(section.get('section_id', '?')))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            circle_data = section.get('circle_data') or {}
            avg_center = circle_data.get('avg_center')
            if avg_center is None or len(avg_center) < 2:
                continue

            min_z = circle_data.get('min_z', 0.0)
            tree_id = section.get('tree_id', '?')
            trunk_id = section.get('trunk_id', None)
            branch_id = section.get('branch_id', None)
            sec_id = section.get('section_id', '?')

            parts = [f"T{tree_id}"]
            if trunk_id is not None:
                try:
                    parts.append(f"Tr{int(trunk_id)}")
                except (ValueError, TypeError):
                    parts.append(f"Tr{trunk_id}")
            if branch_id is not None:
                try:
                    parts.append(f"B{int(branch_id)}")
                except (ValueError, TypeError):
                    parts.append(f"B{branch_id}")
            parts.append(f"S{sec_id}")
            avg_r = circle_data.get('avg_radius')
            if avg_r is not None and avg_r > 0:
                diam_cm = float(avg_r) * 2.0 * 100.0
                label_text = f"{'/'.join(parts)}: {diam_cm:.1f}cm"
            else:
                label_text = "/".join(parts)

            label_points.append([float(avg_center[0]), float(avg_center[1]), float(min_z)])
            label_texts.append(label_text)

        if not label_points:
            return

        try:
            self.plotter.add_point_labels(
                label_points,
                label_texts,
                font_size=9,
                text_color='white',
                point_size=5,
                always_visible=True,
                name='section_info_labels',
                pickable=False,
                render=False
            )
            self.log_to_console(f"[Tag]  Added {len(label_texts)} section labels")
        except Exception as e:
            self.log_to_console(f"[WARN] Failed to add section labels: {e}")


    def clear_accumulated_volume(self):
        """
        Clear all accumulated volume calculation results and reset the visualization.
        """
        try:
            # Clear accumulated data
            self.accumulated_volume_sections = []
            self.trees_with_volume_data.clear()
            
            # Reset volume calculation signature so a new tree can be calculated
            self._last_volume_calc_signature = None
            
            # Refresh tree list colors
            self._refresh_tree_list_colors()
            
            # Clear volume-related actors from plotter
            if self.plotter is not None:
                # Suppress rendering during batch removal for performance
                self.plotter.suppress_rendering = True
                
                # Remove all volume-related and DBH-related meshes and actors.
                # Snapshot the actor dict ONCE (iterating self.plotter.actors
                # repeatedly is O(n²) and very slow with thousands of actors).
                try:
                    actor_names = list(self.plotter.actors)
                except Exception:
                    actor_names = []
                actors_to_remove = [
                    name for name in actor_names
                    if ('volume' in name.lower() or
                        'dbh_' in name.lower() or
                        'trajectory' in name.lower() or
                        'circle' in name.lower() or
                        'section_info_label' in name.lower())
                ]

                for actor_name in actors_to_remove:
                    try:
                        self.plotter.remove_actor(actor_name)
                    except:
                        pass
                
                # Remove volume info text
                try:
                    self.plotter.remove_actor('volume_info_text')
                except:
                    pass
                
                # Also try to clear any text by adding empty text at the same position
                try:
                    self.plotter.add_text("", position='upper_left', font_size=10, color='#FFFFFF',
                                        name='volume_info_text')
                except:
                    pass
                
                # Re-enable rendering after batch removal
                self.plotter.suppress_rendering = False
                self.plotter.render()
                
                self.plotter.update()
            
            # Disable clear button
            self.clear_volume_button.setEnabled(False)
            self.export_trunks_button.setEnabled(False)
            
            self.log_to_console("[Clean] Cleared all accumulated volume calculation results")
            
        except Exception as e:
            QMessageBox.warning(self, "Clear Error", f"Error clearing volume results: {str(e)}")
            print(f"Clear accumulated volume error: {e}")


    def save_volume_data(self):
        """Save current volume calculation data to a JSON file."""
        if not self.accumulated_volume_sections:
            # Log to GUI console instead of popping up a modal dialog
            self.log_to_console("[WARN] No volume calculation data to save.")
            return

        try:
            # Group sections by tree_id
            trees_data = {}
            for section in self.accumulated_volume_sections:
                # Skip sections with non-numeric section_ids except for global_extrapolated
                section_id = section['section_id']
                if section_id != "global_extrapolated":
                    try:
                        int(section_id)
                    except (ValueError, TypeError):
                        self.log_to_console(f"[WARN] Skipping section with non-numeric ID: {section_id}")
                        continue
                    
                tree_id = section['tree_id']
                if tree_id not in trees_data:
                    trees_data[tree_id] = {
                        'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now()),
                        'sections': []
                    }

                # Record the tree base Z used for DBH (derived from the saved
                # DBH target Z minus the reference height) so reloaded data can
                # reproduce the original DBH reference height without needing
                # the tree visualized again.
                _cd = section.get('circle_data') or {}
                _dbh_target_z = _cd.get('dbh_target_z')
                try:
                    _base_z = float(_dbh_target_z) - float(getattr(self, 'dbh_reference_height', 1.3)) if _dbh_target_z is not None else None
                except (ValueError, TypeError):
                    _base_z = None
                if _base_z is None:
                    try:
                        _base_z = float(_cd.get('min_z')) if _cd.get('min_z') is not None else None
                    except (ValueError, TypeError):
                        _base_z = None

                section_data = {
                    'section_id': section['section_id'],
                    'trunk_id': section.get('trunk_id'),
                    'branch_id': section.get('branch_id'),
                    'volume': section['volume'],
                    'point_color': section['point_color'],
                    'circle_color': section['circle_color'],
                    'num_points': len(section['points']),
                    'circle_data': section['circle_data'],
                    'tree_base_z': _base_z,
                    'extrapolated': section.get('extrapolated', False),
                    'original_height_range': section.get('original_height_range'),
                    'extrapolated_height_range': section.get('extrapolated_height_range'),
                    'original_min_z': section.get('original_min_z'),
                    'extrapolated_min_z': section.get('extrapolated_min_z'),
                    'num_extrapolated_circles': section.get('num_extrapolated_circles'),
                    'ground_level_used': section.get('ground_level_used')
                }
                trees_data[tree_id]['sections'].append(section_data)

            # Calculate totals for each tree
            for tree_id, tree_data in trees_data.items():
                tree_data['total_sections'] = len(tree_data['sections'])
                tree_data['total_volume'] = sum(s['volume'] for s in tree_data['sections'])
                tree_data['total_points'] = sum(s['num_points'] for s in tree_data['sections'])

            # Prepare overall data structure
            save_data = {
                'trees': trees_data,
                'overall_totals': {
                    'total_sections': sum(tree_data['total_sections'] for tree_data in trees_data.values()),
                    'total_volume': sum(tree_data['total_volume'] for tree_data in trees_data.values()),
                    'total_points': sum(tree_data['total_points'] for tree_data in trees_data.values()),
                    'total_trees': len(trees_data)
                }
            }

            # Resolve a save folder automatically so the save button does not prompt for a folder.
            if self.volume_folder_path:
                folder = self.volume_folder_path
                self.log_to_console(f"[Save] Using configured volume folder: {folder}")
            else:
                base_dir = None
                las_basename = None
                try:
                    if getattr(self, 'current_las_path', None):
                        base_dir = os.path.dirname(self.current_las_path)
                        las_basename = os.path.splitext(os.path.basename(self.current_las_path))[0]
                except Exception:
                    base_dir = None
                    las_basename = None

                if not base_dir:
                    base_dir = os.getcwd()
                    las_basename = "volume_data"

                folder = os.path.join(base_dir, 'volume', las_basename)
                self.log_to_console(f"[Save] Auto-selected volume folder: {folder}")

            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                self.log_to_console(f"[Error] Could not create volume folder {folder}: {e}")
                return

            self.volume_folder_path = folder
            if hasattr(self, 'volume_folder_label'):
                self.volume_folder_label.setText(f"Volume Folder: {os.path.basename(folder)}")

            if folder:
                import json
                
                saved_files = []
                for tree_id, tree_data in trees_data.items():
                    filename = os.path.join(folder, f"tree_{tree_id}.json")
                    
                    # Check if file exists and load existing data if it does
                    existing_data = None
                    if os.path.exists(filename):
                        try:
                            with open(filename, 'r') as f:
                                existing_data = json.load(f)
                            self.log_to_console(f"[File] Found existing file for tree {tree_id}, will merge data")
                        except Exception as e:
                            self.log_to_console(f"[WARN] Could not read existing file for tree {tree_id}: {e}, will create new file")
                    
                    # Merge with existing data if available
                    if existing_data and 'sections' in existing_data:
                        # Append new sections to existing tree
                        existing_data['sections'].extend(tree_data['sections'])
                        # Recalculate totals
                        existing_data['total_sections'] = len(existing_data['sections'])
                        existing_data['total_volume'] = sum(s['volume'] for s in existing_data['sections'])
                        existing_data['total_points'] = sum(s['num_points'] for s in existing_data['sections'])
                        save_data = existing_data
                        action = "merged with"
                    else:
                        save_data = tree_data
                        action = "saved to"
                    
                    with open(filename, 'w') as f:
                        json.dump(save_data, f, indent=2)
                    
                    saved_files.append(filename)
                    self.log_to_console(f"[Save] Volume data for tree {tree_id} {action} {filename}")

                # Log success to GUI console instead of showing a popup
                total_sections = sum(tree_data['total_sections'] for tree_data in trees_data.values())
                total_volume = sum(tree_data['total_volume'] for tree_data in trees_data.values())
                self.log_to_console(
                    f"[OK] Saved {len(saved_files)} tree files to {folder} | Total trees: {len(trees_data)} | "
                    f"Total sections: {total_sections} | Total volume: {total_volume:.4f} m³"
                )

        except Exception as e:
            # Log error to GUI console rather than showing a modal dialog
            self.log_to_console(f"[Error] Error saving volume data: {e}")
            print(f"Save volume data error: {e}")


    def load_volume_data(self):
        """Load volume calculation data from tree JSON files in a folder."""
        try:
            self.progress_bar.setVisible(True)
            self._start_volume_load_profile()
            self._set_progress(2, "Resolving volume folder")
            self._profile_volume_load('start')

            # Resolve the volume folder corresponding to the currently active LAS file first.
            folder = None
            if getattr(self, 'current_las_path', None):
                base_dir = os.path.dirname(self.current_las_path)
                las_basename = os.path.splitext(os.path.basename(self.current_las_path))[0]
                expected_folder = os.path.join(base_dir, 'volume', las_basename)
                if os.path.exists(expected_folder):
                    folder = expected_folder
                elif hasattr(self, 'volume_folder_path') and isinstance(self.volume_folder_path, str) and self.volume_folder_path and os.path.exists(self.volume_folder_path):
                    folder = self.volume_folder_path
                else:
                    folder = expected_folder
                self.log_to_console(f"[File] Resolved volume folder for active LAS: {folder}")
            elif hasattr(self, 'volume_folder_path') and isinstance(self.volume_folder_path, str) and self.volume_folder_path and os.path.exists(self.volume_folder_path):
                folder = self.volume_folder_path
                self.log_to_console(f"[File] Using configured volume folder: {folder}")
            else:
                folder = os.path.join(os.getcwd(), 'volume', 'volume_data')
                self.log_to_console(f"[File] Auto-resolved volume folder: {folder}")

            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Volume Folder Error", f"Could not create volume folder:\n{folder}\n\n{e}")
                self._finish_volume_load_profile('folder_error')
                return

            import json
            import glob

            self.volume_folder_path = folder
            if hasattr(self, 'volume_folder_label'):
                self.volume_folder_label.setText(f"Volume Folder: {os.path.basename(folder)}")

            # Find all tree_*.json files in the folder
            all_tree_files = sorted(glob.glob(os.path.join(folder, "tree_*.json")))
            self._profile_volume_load('resolve_folder_and_list_files', extra=f'{len(all_tree_files)} files in {folder}')
            
            if not all_tree_files:
                QMessageBox.warning(self, "No Tree Files", f"No tree_*.json files found in {folder}")
                self._finish_volume_load_profile('no_files')
                return

            # If specific tree(s) are selected in the tree list, load only their data
            selected_tree_ids = self.get_selected_tree_ids()
            if selected_tree_ids:
                selected_ids_str = {str(tid) for tid in selected_tree_ids}
                tree_files = []
                for tree_file in all_tree_files:
                    filename = os.path.basename(tree_file)
                    file_tree_id = filename.replace('tree_', '').replace('.json', '')
                    if file_tree_id in selected_ids_str:
                        tree_files.append(tree_file)

                if not tree_files:
                    QMessageBox.warning(
                        self, "No Data For Selected Tree(s)",
                        f"No volume data files found for selected tree(s): {sorted(selected_tree_ids)}\n\n"
                        f"Folder searched: {folder}")
                    self._finish_volume_load_profile('no_selected_tree_files')
                    return

                self.log_to_console(
                    f" Loading volume data only for selected tree(s): {sorted(selected_tree_ids)} "
                    f"({len(tree_files)} of {len(all_tree_files)} files in folder)")
            else:
                tree_files = all_tree_files
                self.log_to_console(
                    f"[File] No trees selected - found {len(tree_files)} tree files in folder. Loading all files...")

            # Clear existing volume data if any
            if self.accumulated_volume_sections:
                reply = QMessageBox.question(self, "Clear Existing Data",
                                           "Loading new volume data will clear existing calculations. Continue?",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    self._finish_volume_load_profile('cancelled')
                    return
                self.clear_accumulated_volume()
                self._profile_volume_load('clear_existing_volume')

            # Load data from all tree files
            self.accumulated_volume_sections = []
            loaded_trees = 0
            total_sections = 0

            self._set_progress(10, f"Reading {len(tree_files)} JSON files")

            # Read + parse all JSON files in parallel (I/O-bound). This is the
            # biggest win for folders with many tree files.
            from concurrent.futures import ThreadPoolExecutor

            def _read_tree_file(tree_file):
                with open(tree_file, 'r') as f:
                    return tree_file, json.load(f)

            parsed_files = []
            max_workers = min(32, max(4, (os.cpu_count() or 8) * 4))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, result in enumerate(executor.map(_read_tree_file, tree_files)):
                    parsed_files.append(result)
                    if i % max(1, len(tree_files) // 20) == 0:
                        self._set_progress(10 + int(30 * i / len(tree_files)),
                                           f"Reading JSON {i}/{len(tree_files)}")
            self._profile_volume_load('read_json_files', extra=f'{len(parsed_files)} files')

            self._set_progress(42, "Parsing sections")

            per_tree_logs = []
            for tree_file, tree_data in parsed_files:
                try:
                    # Extract tree_id from filename
                    filename = os.path.basename(tree_file)
                    tree_id = filename.replace('tree_', '').replace('.json', '')

                    # Validate data structure
                    if 'sections' not in tree_data:
                        per_tree_logs.append(f"[WARN] Skipping {filename}: no sections data")
                        continue

                    # Load sections for this tree
                    n_loaded = 0
                    for section_data in tree_data['sections']:
                        try:
                            # Handle both numeric and string section_ids
                            section_id = section_data['section_id']
                            if section_id == "global_extrapolated":
                                # Keep as string for extrapolated sections
                                pass
                            else:
                                # Convert numeric section_ids to int
                                section_id = int(section_id)
                            
                            section_info = {
                                'tree_id': tree_id,
                                'section_id': section_id,
                                'trunk_id': section_data.get('trunk_id'),
                                'branch_id': section_data.get('branch_id'),
                                'volume': section_data['volume'],
                                'point_color': section_data['point_color'],
                                'circle_color': section_data['circle_color'],
                                'points': np.array([]),  # Empty array since we don't save point data
                                'num_points': int(section_data.get('num_points', 0) or 0),
                                'tree_base_z': section_data.get('tree_base_z'),
                                'circle_data': section_data['circle_data']
                            }
                            self.accumulated_volume_sections.append(section_info)
                            total_sections += 1
                            n_loaded += 1
                        except (ValueError, TypeError) as e:
                            per_tree_logs.append(f"[WARN] Skipping section with invalid section_id '{section_data.get('section_id', 'unknown')}': {e}")
                            continue

                    loaded_trees += 1
                    per_tree_logs.append(f"[OK] Loaded tree {tree_id} with {n_loaded} sections")

                except Exception as e:
                    per_tree_logs.append(f"[WARN] Error loading {tree_file}: {e}")
                    continue
            self._profile_volume_load('parse_sections', extra=f'{total_sections} sections from {loaded_trees} trees')

            # Flush all per-tree messages in a single console update
            self.log_many_to_console(per_tree_logs)

            # Deduplicate loaded sections by (tree_id, section_id) — the JSON
            # may contain duplicates from older save logic.
            self._set_progress(46, "Deduplicating sections")
            _seen = set()
            _deduped = []
            for _s in self.accumulated_volume_sections:
                _key = (str(_s.get('tree_id', '')), str(_s.get('section_id', '')))
                if _key not in _seen:
                    _seen.add(_key)
                    _deduped.append(_s)
            _dup_count = len(self.accumulated_volume_sections) - len(_deduped)
            if _dup_count > 0:
                self.log_to_console(f"[Clean] Removed {_dup_count} duplicate section(s) from loaded data")
                self.accumulated_volume_sections = _deduped
                total_sections = len(self.accumulated_volume_sections)
            self._profile_volume_load('deduplicate_sections', extra=f'{total_sections} unique sections')

            # Track which trees have volume data for UI coloring.
            # Base this on ALL files present in the folder (not just the loaded
            # subset) so single-tree loads don't clear coloring for other trees.
            folder_tree_ids = set()
            for tree_file in all_tree_files:
                filename = os.path.basename(tree_file)
                folder_tree_ids.add(filename.replace('tree_', '').replace('.json', ''))
            self.trees_with_volume_data = folder_tree_ids
            
            # DEBUG: Print summary of loaded trees
            loaded_tree_ids = sorted(set(s['tree_id'] for s in self.accumulated_volume_sections))
            self.log_to_console(f" DEBUG: Loaded sections for {len(loaded_tree_ids)} tree(s): {loaded_tree_ids}")
            self.log_to_console(f" DEBUG: Folder contains volume data for {len(self.trees_with_volume_data)} trees: {sorted(self.trees_with_volume_data)}")
            
            # Refresh tree list colors to show which trees have volume data
            self._refresh_tree_list_colors()
            self._profile_volume_load('refresh_tree_list_colors')
            
            # Update UI
            self.clear_volume_button.setEnabled(True)
            self.save_volume_button.setEnabled(True)
            self.export_trunks_button.setEnabled(True)

            # Update info display
            total_sections = len(self.accumulated_volume_sections)
            total_volume = sum(s['volume'] for s in self.accumulated_volume_sections)
            total_trees = len(set(s['tree_id'] for s in self.accumulated_volume_sections))

            info_text = (f"Loaded Volume Calculation\n"
                        f"Trees: {total_trees}\n"
                        f"Sections: {total_sections}\n"
                        f"Total Volume: {total_volume:.4f} m³\n"
                        f"Source: {folder}")

            self.log_to_console(info_text)

            self.log_to_console(f"[File] Loaded volume data from {len(tree_files)} files in {folder}")
            
            # Restore visualizations for all loaded sections (batched for better performance)
            self.log_to_console(f"Color Restoring visualizations for {total_sections} sections...")
            if self.plotter is not None:
                self.plotter.suppress_rendering = True
            n_sections = len(self.accumulated_volume_sections)
            # Batch all section circles into a single mesh (one add_mesh call)
            # instead of one add_mesh per section, which is O(n²) and very slow
            # with hundreds of sections.
            individual_specs = []
            average_specs = []
            for i, section_info in enumerate(self.accumulated_volume_sections):
                circle_data = section_info.get('circle_data') or {}
                circles = circle_data.get('circles') or []
                if circles:
                    individual_specs.extend(circles)
                    avg_r = circle_data.get('avg_radius')
                    if avg_r and avg_r > 0:
                        average_specs.extend((center, avg_r) for center, _ in circles)
                if i % max(1, n_sections // 25) == 0:
                    self._set_progress(55 + int(40 * i / max(1, n_sections)),
                                       f"Restoring circles {i}/{n_sections}")

            if individual_specs:
                merged_individual = self._build_circles_mesh(individual_specs)
                self.plotter.add_mesh(
                    merged_individual,
                    color='cyan',
                    style='wireframe',
                    line_width=2,
                    opacity=0.9,
                    name='dbh_individual_circles_loaded',
                    label=f'Loaded Individual Circles ({len(individual_specs)} circles)',
                    pickable=False
                )
            if average_specs:
                merged_average = self._build_circles_mesh(average_specs)
                self.plotter.add_mesh(
                    merged_average,
                    color='red',
                    style='wireframe',
                    line_width=3,
                    opacity=0.9,
                    name='dbh_avg_circles_loaded',
                    label=f'Loaded Avg Circles ({len(average_specs)} circles)',
                    pickable=False
                )
            self._profile_volume_load('restore_section_circles', extra=f'{n_sections} sections')
            self._set_progress(96, "Building summaries")
            has_branch_mode = any(s.get('branch_id') is not None for s in self.accumulated_volume_sections)

            # Build trunk volume summary and log to console
            trunk_summary, trunk_summary_lines, trunk_total_volume = self._build_trunk_volume_summary_table()
            self._profile_volume_load('build_trunk_volume_summary')
            trunk_dbh_summary, trunk_dbh_lines, trunk_dbh_details = self._build_trunk_dbh_summary_table() if has_branch_mode else ({}, [], {})
            self._profile_volume_load(
                'build_trunk_dbh_summary',
                extra=f'{len(trunk_dbh_details)} trunks' if has_branch_mode else 'skipped'
            )

            if has_branch_mode and trunk_dbh_details:
                self._add_trunk_dbh_overlays(trunk_dbh_details)
                self._profile_volume_load('add_blue_dbh_overlays', extra=f'{len(trunk_dbh_details)} trunks')

            # Add section info labels showing trunk/branch/section for each loaded section
            self._add_section_info_labels()
            self._profile_volume_load('add_section_info_labels')
            if self.plotter is not None:
                self.plotter.suppress_rendering = False
                self._profile_volume_load('before_first_render')
                self.plotter.render()
                self._profile_volume_load('plotter_render')
            self._set_progress(97, "Finalizing")
            self.log_to_console(f"[OK] Visualization restoration complete")
            self._log_section_diameter_breakdown()
            self._profile_volume_load('log_section_diameter_breakdown')

            if trunk_summary:
                self.log_to_console(f"[Box] Total trunks used for volume: {len(trunk_summary)}")
                self.log_to_console("[Summary] Trunk volume summary:")
                for line in trunk_summary_lines:
                    self.log_to_console(line)
                    print(line)
                self.log_to_console(f"[Metrics] Combined trunk volume: {trunk_total_volume:.4f} m³")
                print(f"Combined trunk volume: {trunk_total_volume:.4f} m³")
            else:
                self.log_to_console("[Box] Total trunks used for volume: 0")
                self.log_to_console("[Summary] Trunk volume summary: none")
                print("Total trunks used for volume: 0")
                print("Trunk volume summary: none")

            if has_branch_mode:
                if trunk_dbh_summary:
                    dbh_h = getattr(self, 'dbh_reference_height', 1.3)
                    dbh_msg = f"[Dim] Trunk DBH summary ({dbh_h:.1f}m above base):"
                    self.log_to_console(dbh_msg)
                    for line in trunk_dbh_lines:
                        self.log_to_console(line)
                        print(line)
                else:
                    dbh_h = getattr(self, 'dbh_reference_height', 1.3)
                    dbh_msg = f"[Dim] Trunk DBH summary ({dbh_h:.1f}m above base): none"
                    self.log_to_console(dbh_msg)

            # Build volume info text overlay on the 3D plotter
            if self.plotter is not None:
                total_pts = sum(len(s.get('points', [])) for s in self.accumulated_volume_sections)
                mode_text = "Mode: Branch-wise" if has_branch_mode else "Mode: Standard"

                trunk_summary_overlay = ""
                if trunk_summary:
                    compact_rows = []
                    for tid in sorted(trunk_summary.keys()):
                        data = trunk_summary[tid]
                        compact_rows.append(f"T{tid}: {data['total_volume']:.4f} m³ / {data['num_branches']} branches")
                    trunk_summary_overlay = "\nTrunks Used: " + str(len(trunk_summary)) + "\n" + "\n".join(compact_rows)

                dbh_h = getattr(self, 'dbh_reference_height', 1.3)
                if has_branch_mode and trunk_dbh_summary:
                    dbh_lines = []
                    for key, dbh in sorted(trunk_dbh_summary.items()):
                        parts = str(key).rsplit('_', 1)
                        if len(parts) == 2:
                            tree_id, trunk_id = parts[0], parts[1]
                            dbh_lines.append(f"Tree{tree_id}/T{trunk_id}: {dbh:.1f} cm")
                        else:
                            dbh_lines.append(f"T{key}: {dbh:.1f} cm")
                    dbh_text = "\nDBH by trunk: " + "; ".join(dbh_lines)
                else:
                    dbh_text = "\nDBH: N/A"

                info_text = (f"Loaded Volume Calculation\n"
                            f"{mode_text}\n"
                            f"Sections: {total_sections}\n"
                            f"Total Points: {total_pts}\n"
                            f"Combined Trunk Volume: {total_volume:.4f} m³\n"
                            f"Trunks Used: {len(trunk_summary) if trunk_summary else 0}"
                            f"{trunk_summary_overlay}"
                            f"{dbh_text}\n"
                            f"Source: {os.path.basename(folder)}")

                try:
                    self.plotter.remove_actor('volume_info_text')
                except:
                    pass

                self.plotter.add_text(info_text, position='upper_left', font_size=10, color='#FFFFFF',
                                    name='volume_info_text')
                self.plotter.update()
                self._profile_volume_load('add_volume_info_text_and_update')

            self._set_progress(100, "Done")
            self.progress_bar.setVisible(False)
            self._finish_volume_load_profile('done')

            QMessageBox.information(self, "Load Successful",
                                  f"Volume data loaded from {len(tree_files)} files in {folder}\n"
                                  f"Trees: {total_trees}\n"
                                  f"Sections: {total_sections}\n"
                                  f"Total volume: {total_volume:.4f} m³")

        except Exception as e:
            self.progress_bar.setVisible(False)
            self._finish_volume_load_profile('error')
            QMessageBox.warning(self, "Load Error", f"Error loading volume data: {str(e)}")
            print(f"Load volume data error: {e}")


    @staticmethod
    def _build_circles_mesh(circle_specs, resolution=32):
        """Build a single merged PolyData containing many circles, vectorized.

        circle_specs: iterable of (center_xyz, radius).
        Much faster than creating pv.Circle per circle and merging pairwise.
        """
        circle_specs = list(circle_specs)
        if not circle_specs:
            return pv.PolyData()

        centers = np.asarray([c for c, _ in circle_specs], dtype=float)
        radii = np.asarray([r for _, r in circle_specs], dtype=float)
        n = len(circle_specs)

        # Unit-circle template (XY plane, z=0)
        theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Vertex positions: n circles × resolution points, scaled and translated
        pts = np.empty((n * resolution, 3), dtype=float)
        pts[:, 0] = np.repeat(centers[:, 0], resolution) + np.tile(cos_t, n) * np.repeat(radii, resolution)
        pts[:, 1] = np.repeat(centers[:, 1], resolution) + np.tile(sin_t, n) * np.repeat(radii, resolution)
        pts[:, 2] = np.repeat(centers[:, 2], resolution)

        # Line cells forming closed loops (VTK polyline format)
        lines = np.empty((n, resolution + 1), dtype=np.int64)
        lines[:, 0] = resolution
        idx = np.arange(n * resolution).reshape(n, resolution)
        lines[:, 1:] = np.roll(idx, shift=-1, axis=1)
        lines[:, 1] = idx[:, 0]  # close the loop back to the first point
        lines_flat = lines.reshape(-1)

        mesh = pv.PolyData()
        mesh.points = pts
        mesh.lines = lines_flat
        return mesh


    def check_volume_folder_for_trees(self):
        """Check the selected volume folder for tree_*.json files and update trees_with_volume_data set."""
        if not hasattr(self, 'volume_folder_path') or not isinstance(self.volume_folder_path, str) or not self.volume_folder_path or not os.path.exists(self.volume_folder_path):
            return

        try:
            import glob

            # Find all tree_*.json files
            tree_files = glob.glob(os.path.join(self.volume_folder_path, "tree_*.json"))
            
            if not tree_files:
                self.log_to_console(f" No volume files found in {self.volume_folder_path}")
                self.trees_with_volume_data = set()
                return

            # Extract tree IDs from filenames
            volume_tree_ids = set()
            for tree_file in tree_files:
                filename = os.path.basename(tree_file)
                tree_id = filename.replace('tree_', '').replace('.json', '')
                volume_tree_ids.add(tree_id)

            # Update the trees_with_volume_data set
            self.trees_with_volume_data = volume_tree_ids
            
            self.log_to_console(f"[Metrics] Updated volume data tracking for {len(volume_tree_ids)} trees: {sorted(volume_tree_ids)}")

        except Exception as e:
            self.log_to_console(f"[Error] Error checking volume folder: {str(e)}")
            self.trees_with_volume_data = set()


    def on_volume_folder_checkbox_changed(self, state):
        """Handle volume folder checkbox state change."""
        if state == 2:  # Checked
            # Update tree list colors based on volume folder content
            if hasattr(self, 'volume_folder_path') and self.volume_folder_path and isinstance(self.volume_folder_path, str):
                self.check_volume_folder_for_trees()
                self._refresh_tree_list_colors()
        else:  # Unchecked
            # Clear volume data coloring
            self.trees_with_volume_data = set()
            self._refresh_tree_list_colors()


