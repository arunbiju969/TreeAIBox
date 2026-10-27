"""Batch trunk volume analysis and metric exports. (TreeVisualizerGUI mixin)."""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import laspy
import pyvista as pv
from PyQt6.QtWidgets import (
    QApplication, QMessageBox, QFileDialog, QProgressBar, QLabel, QDialog
)
from PyQt6.QtCore import Qt
from forestry_suite.dialogs.batch_params import (
    BatchBranchParamsDialog,
    BatchSkipOptionsDialog,
    BatchTrunkParamsDialog,
)


class BatchMixin:
    """Batch trunk volume analysis and metric exports."""

    def batch_analyze_trunk_volumes(self):
        """
        Automated batch processing across all checked LAS files in the Layer Manager.
        For each checked LAS file, iterates through all tree IDs with trunk points,
        visualizes trunk, detects trunk/branches, calculates volume, and exports metrics.
        """
        # Determine target LAS files from layer manager (or active file fallback)
        checked_layers = [l for l in self.las_layers if l.get('checked', True)]
        if not checked_layers and self.current_las_path:
            norm_c = os.path.normpath(self.current_las_path)
            checked_layers = [{
                'path': norm_c,
                'name': os.path.basename(norm_c)
            }]

        if not checked_layers:
            QMessageBox.warning(self, "Warning", "No LAS files checked or loaded for batch analysis.")
            return

        try:
            import time

            batch_start_perf = time.perf_counter()
            batch_start_dt = datetime.now()
            self.log_to_console(f"[Time]️ Multi-LAS Batch analysis started for {len(checked_layers)} file(s) at {batch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            # --- Ask user for batch configuration dialogs (apply once for the entire batch run) ---
            skip_dialog = BatchSkipOptionsDialog(self, default_threshold=50)
            if skip_dialog.exec() != QDialog.Accepted:
                self.log_to_console("ℹ️ Batch cancelled by user (skip options)")
                return
            skip_opts = skip_dialog.get_options()
            self.log_to_console(f" Batch skip options: {skip_opts}")

            _batch_trunk_defaults = dict(getattr(self, 'trunk_detection_params', {}))
            _batch_trunk_defaults['dbh_reference_height'] = float(getattr(self, 'dbh_reference_height', 1.3))
            trunk_dialog = BatchTrunkParamsDialog(self, defaults=_batch_trunk_defaults)
            if trunk_dialog.exec() == QDialog.Accepted:
                new_trunk_params = trunk_dialog.get_params()
                try:
                    self.trunk_slice_height_input.setValue(int(new_trunk_params.get('slice_height_cm', 25)))
                    self.trunk_base_zone_input.setValue(float(new_trunk_params.get('base_zone_height_m', 0.0)))
                    self.trunk_eps_input.setValue(float(new_trunk_params.get('eps_m', 0.18)))
                    self.trunk_min_samples_input.setValue(int(new_trunk_params.get('min_samples', 8)))
                    self.trunk_merge_dist_input.setValue(float(new_trunk_params.get('merge_dist_m', 0.25)))
                    self.trunk_min_points_input.setValue(int(new_trunk_params.get('min_trunk_points', 120)))
                    self.trunk_min_span_input.setValue(float(new_trunk_params.get('min_vertical_span_m', 1.0)))
                    self.trunk_full_assign_input.setValue(float(new_trunk_params.get('full_height_assign_dist_m', 0.45)))
                    _batch_dbh_h = new_trunk_params.get('dbh_reference_height')
                    if _batch_dbh_h is not None:
                        if hasattr(self, 'dbh_height_input'):
                            self.dbh_height_input.setValue(float(_batch_dbh_h))
                        else:
                            self.dbh_reference_height = float(_batch_dbh_h)
                        self.log_to_console(
                            f"[Measure] Batch DBH reference height set to: {float(_batch_dbh_h):.2f}m"
                        )
                    self.trunk_detection_params.update(new_trunk_params)
                    self.log_to_console(" Trunk detection parameters updated for batch run")
                except Exception:
                    pass
            else:
                self.log_to_console("ℹ️ Batch cancelled by user (trunk parameters)")
                return

            branch_dialog = BatchBranchParamsDialog(self, defaults={
                'slice_height_cm': getattr(self, 'slice_height_input', None) and int(self.slice_height_input.value()) or 25,
                'branch_eps': getattr(self, 'branch_eps_input', None) and float(self.branch_eps_input.value()) or 0.12,
                'branch_min_samples': getattr(self, 'branch_min_samples_input', None) and int(self.branch_min_samples_input.value()) or 6,
                'branch_merge_dist': getattr(self, 'branch_merge_dist_input', None) and float(self.branch_merge_dist_input.value()) or 0.25,
            })
            if branch_dialog.exec() == QDialog.Accepted:
                new_branch_params = branch_dialog.get_params()
                try:
                    self.slice_height_input.setValue(int(new_branch_params.get('slice_height_cm', 25)))
                    self.branch_eps_input.setValue(float(new_branch_params.get('branch_eps', 0.12)))
                    self.branch_min_samples_input.setValue(int(new_branch_params.get('branch_min_samples', 6)))
                    self.branch_merge_dist_input.setValue(float(new_branch_params.get('branch_merge_dist', 0.25)))
                    self.log_to_console(" Branch detection parameters updated for batch run")
                except Exception:
                    pass
            else:
                self.log_to_console("ℹ️ Batch cancelled by user (branch parameters)")
                return

            total_trees_processed = 0
            total_trees_found = 0
            all_skipped_trees = []
            exported_csv_files = []

            for file_idx, layer in enumerate(checked_layers, 1):
                file_path = layer['path']
                file_name = layer['name']

                self.log_to_console(f"\n{'='*60}")
                self.log_to_console(f" [File {file_idx}/{len(checked_layers)}] Starting batch analysis for: {file_name}")
                self.log_to_console(f"{'='*60}")

                # Load/activate the LAS file
                try:
                    self.activate_las_layer(file_path, silent=True)
                except Exception as load_err:
                    self.log_to_console(f"[Error] Failed to load file {file_name}: {load_err}")
                    continue

                if self.las_data is None or 'stemcls' not in self.las_data.point_format.dimension_names:
                    self.log_to_console(f"[WARN] Skipping file {file_name}: no valid LAS data or 'stemcls' field missing.")
                    continue

                all_forest_ids = sorted(self.unique_itc_values.tolist())
                itc_values = np.array(self.las_data['itc'])
                stemcls_values = np.array(self.las_data['stemcls'])
                
                valid_tree_ids = []
                for tree_id in all_forest_ids:
                    mask = (itc_values == tree_id) & (stemcls_values != 1)
                    if np.any(mask):
                        valid_tree_ids.append(tree_id)

                self.log_to_console(f"[Tree] [{file_name}] Trees with trunk points: {len(valid_tree_ids)} / {len(all_forest_ids)}")
                total_trees_found += len(valid_tree_ids)

                if not valid_tree_ids:
                    self.log_to_console(f"ℹ️ [{file_name}] No trees with trunk points found.")
                    continue

                self.accumulated_volume_sections = []
                folder = os.path.join(os.path.dirname(file_path), 'volume', os.path.splitext(file_name)[0])
                os.makedirs(folder, exist_ok=True)
                self.volume_folder_path = folder
                if hasattr(self, 'volume_folder_label'):
                    self.volume_folder_label.setText(f"Volume Folder: {os.path.basename(folder)}")

                file_processed_count = 0
                skipped_trees = []

                for idx, tree_id in enumerate(valid_tree_ids, 1):
                    self.log_to_console(f"\n[Point] [{file_name} | {idx}/{len(valid_tree_ids)}] Processing tree ID={tree_id}")

                    try:
                        self._visualize_trunk_single_tree(tree_id)
                        QApplication.processEvents()

                        try:
                            self.detect_trunks_in_slices()
                            QApplication.processEvents()
                        except Exception as dt_e:
                            self.log_to_console(f"[WARN] Trunk detection failed for tree {tree_id}: {dt_e}")

                        num_trunks = 0
                        if hasattr(self, 'trunk_assignment') and self.trunk_assignment is not None:
                            unique_trunks = np.unique(self.trunk_assignment)
                            unique_trunks = unique_trunks[unique_trunks >= 0]
                            num_trunks = len(unique_trunks)

                        if num_trunks == 0 and skip_opts.get('skip_no_trunk', True):
                            self.log_to_console(f"️ Skipping tree {tree_id}: no trunks detected")
                            skipped_trees.append((tree_id, 'no_trunks'))
                            continue

                        if skip_opts.get('skip_small_trunk', True) and num_trunks > 0:
                            trunk_sizes = [int(np.sum(self.trunk_assignment == t)) for t in unique_trunks]
                            max_size = max(trunk_sizes) if trunk_sizes else 0
                            if max_size < int(skip_opts.get('small_trunk_threshold', 50)):
                                self.log_to_console(f"️ Skipping tree {tree_id}: largest trunk {max_size} pts < threshold")
                                skipped_trees.append((tree_id, 'small_trunk'))
                                continue

                        try:
                            if num_trunks > 0:
                                self.detect_branch_clusters_in_slices()
                                QApplication.processEvents()
                        except Exception as db_e:
                            self.log_to_console(f"[WARN] Branch detection failed for tree {tree_id}: {db_e}")

                        self._select_all_points_silent()
                        QApplication.processEvents()
                        self._calculate_volume_silent()
                        QApplication.processEvents()

                        self._save_volume_data_silent()
                        QApplication.processEvents()

                        try:
                            self.export_trunk_metrics()
                        except Exception as export_e:
                            self.log_to_console(f"[WARN] Export failed for tree {tree_id}: {export_e}")
                            raise

                        self.clear_accumulated_volume()

                        file_processed_count += 1
                        total_trees_processed += 1
                        self.log_to_console(f"[OK] [{file_name}] Successfully processed tree {tree_id}")

                    except Exception as e:
                        self.log_to_console(f"[WARN] [{file_name}] Skipped tree {tree_id}: {str(e)}")
                        skipped_trees.append((tree_id, str(e)))

                all_skipped_trees.extend([(file_name, tid, reason) for tid, reason in skipped_trees])

                # Save export file path for master merge
                export_file = os.path.join(os.path.dirname(file_path), 'export', f"{os.path.splitext(file_name)[0]}_trunk_metrics.csv")
                if os.path.exists(export_file):
                    exported_csv_files.append(export_file)

            # Master Multi-LAS Export Merger
            if len(exported_csv_files) > 1:
                try:
                    self._create_combined_multi_las_export(exported_csv_files)
                except Exception as merge_err:
                    self.log_to_console(f"[WARN] Failed to generate combined multi-LAS export CSV: {merge_err}")

            # Final summary
            batch_end_dt = datetime.now()
            elapsed_seconds = max(time.perf_counter() - batch_start_perf, 1e-9)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
            self.log_to_console(
                f"\n Multi-LAS Batch analysis completed! Total files: {len(checked_layers)} | Total trees processed: {total_trees_processed} | Duration: {elapsed_str}"
            )

            self._display_batch_completion_summary(
                total_trees_processed,
                total_trees_found,
                [(tid, reason) for _, tid, reason in all_skipped_trees],
                batch_start_dt,
                batch_end_dt,
                elapsed_seconds,
            )

        except Exception as e:
            self.log_to_console(f"[Error] Batch analysis error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Batch analysis failed: {str(e)}")


    def _create_combined_multi_las_export(self, csv_files):
        """Combine metrics from multiple LAS export CSV files into a master CSV."""
        import csv
        combined_rows = []
        fieldnames = None

        for csv_path in csv_files:
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    raw_fields = list(reader.fieldnames or [])
                    if 'las_file' not in raw_fields:
                        fieldnames = ['las_file'] + raw_fields
                    else:
                        fieldnames = raw_fields
                
                las_name = os.path.basename(csv_path).replace('_trunk_metrics.csv', '')
                for row in reader:
                    row['las_file'] = las_name
                    combined_rows.append(row)

        if combined_rows and fieldnames:
            first_file_dir = os.path.dirname(csv_files[0])
            master_csv = os.path.join(first_file_dir, "combined_multi_las_trunk_metrics.csv")
            with open(master_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combined_rows)
            self.log_to_console(f"[Metrics] Generated combined multi-LAS export: {master_csv}")



    def _visualize_trunk_single_tree(self, tree_id):
        """Visualize trunk for a single tree ID in batch mode."""
        if self.las_data is None:
            raise RuntimeError("No LAS data available")

        if not self.plotter:
            raise RuntimeError("3D viewer not initialized")

        # Clear previous visualization
        self.plotter.clear()
        self._restore_gdb_layers()

        try:
            itc_values = np.array(self.las_data['itc'])
            stemcls_values = np.array(self.las_data['stemcls'])
            
            # Get trunk points for this single tree
            mask = (itc_values == tree_id) & (stemcls_values != 1)

            # Apply tree-point filter if enabled
            tree_mask = self.get_tree_point_mask()
            if tree_mask is not None:
                mask = mask & tree_mask

            if not np.any(mask):
                raise ValueError(f"No trunk points for tree {tree_id}")

            points = np.column_stack([
                np.array(self.las_data.x)[mask],
                np.array(self.las_data.y)[mask],
                np.array(self.las_data.z)[mask]
            ])

            # Store for later use
            self.current_tree_points = points
            self.current_tree_id = tree_id
            self.current_mask = mask
            self.current_tree_ids = np.full(len(points), tree_id)
            self.current_color_values = None
            self.current_color_field = "Default (green)"

            # Compute full tree height (using all points for this tree ID, including canopy) for export.
            full_mask = (itc_values == tree_id)
            tree_mask_full = self.get_tree_point_mask()
            if tree_mask_full is not None:
                full_mask = full_mask & tree_mask_full

            if np.any(full_mask):
                full_zs = np.array(self.las_data.z)[full_mask]
                self.current_tree_height = float(np.max(full_zs) - np.min(full_zs))
                self.current_tree_ground_z = float(np.min(full_zs))
            else:
                self.current_tree_height = float(np.max(points[:, 2]) - np.min(points[:, 2]))
                self.current_tree_ground_z = float(np.min(points[:, 2]))

            # Visualize with green color
            self.plotter.add_points(points, color='green', point_size=3, name=f'trunk_tree_{tree_id}')
            self.plotter.reset_camera()

        except Exception as e:
            raise RuntimeError(f"Failed to visualize trunk for tree {tree_id}: {str(e)}")


    def _select_all_points_silent(self):
        """Select all points without UI messages (batch mode)."""
        if self.current_tree_points is None or len(self.current_tree_points) == 0:
            raise ValueError("No tree points available for selection")

        self.selected_point_indices = np.arange(len(self.current_tree_points))
        self.selected_polygon = None
        self.visualize_selected_points()  # Update visualization with highlighted selection


    def _calculate_volume_silent(self):
        """Calculate volume without user dialogs (batch mode)."""
        if self.selected_point_indices is None or len(self.selected_point_indices) == 0:
            raise ValueError("No points selected for volume calculation")

        if not hasattr(self, 'current_tree_points') or self.current_tree_points is None:
            raise ValueError("No tree points available")

        volume_groups, is_branch_mode = self._get_volume_groups_from_selection()
        if not volume_groups:
            raise ValueError("No valid point groups found for volume calculation")
        
        # Determine tree_id
        if hasattr(self, 'current_tree_ids') and self.current_tree_ids is not None:
            selected_tree_ids = self.current_tree_ids[self.selected_point_indices]
            unique_ids, counts = np.unique(selected_tree_ids, return_counts=True)
            section_tree_id = unique_ids[np.argmax(counts)]
        else:
            section_tree_id = self.current_tree_id
        
        # Find next section ID
        existing_section_ids = [int(s['section_id']) for s in self.accumulated_volume_sections 
                               if s['tree_id'] == section_tree_id and str(s['section_id']).isdigit()]
        section_id = max(existing_section_ids) + 1 if existing_section_ids else 1
        
        # Define colors
        section_colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'purple', 'brown']
        point_color = section_colors[(section_id - 1) % len(section_colors)]
        circle_color = section_colors[(section_id - 1) % len(section_colors)]
        avg_color = 'red'

        # Support trunk-scoped processing when available
        if hasattr(self, 'volume_groups_by_trunk') and self.volume_groups_by_trunk:
            # Initialize trunk summary container
            self.trunk_volume_summary = {}

            # Suppress rendering during batch additions for performance
            if self.plotter is not None:
                self.plotter.suppress_rendering = True

            for trunk_id, branch_list in sorted(self.volume_groups_by_trunk.items(), key=lambda x: x[0]):
                trunk_total_vol = 0.0
                trunk_point_count = 0
                trunk_branch_details = []

                for branch_info in branch_list:
                    group_branch_id, _, group_points = branch_info
                    point_color = section_colors[(section_id - 1) % len(section_colors)]
                    circle_color = section_colors[(section_id - 1) % len(section_colors)]
                    branch_suffix = f" | B{group_branch_id}" if group_branch_id is not None else ""

                    # Add selected points
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

                    # Add circle fitting
                    measurement_label = "Diameter" if is_branch_mode else "DBH"
                    section_data = self._add_dbh_circle_fitting_for_section(
                        group_points,
                        section_id,
                        circle_color,
                        avg_color,
                        measurement_label=measurement_label
                    )

                    # Calculate volume
                    section_volume = self._calculate_section_volume(section_data)

                    # Store section (annotate with trunk_id)
                    section_record = {
                        'tree_id': section_tree_id,
                        'section_id': section_id,
                        'trunk_id': int(trunk_id),
                        'branch_id': int(group_branch_id) if group_branch_id is not None else None,
                        'points': group_points.copy(),
                        'point_color': point_color,
                        'circle_color': circle_color,
                        'volume': section_volume,
                        'circle_data': section_data
                    }
                    self.accumulated_volume_sections.append(section_record)

                    # Update trunk totals and details
                    trunk_total_vol += float(section_volume)
                    trunk_point_count += int(len(group_points))
                    trunk_branch_details.append({
                        'branch_id': int(group_branch_id) if group_branch_id is not None else None,
                        'volume': float(section_volume),
                        'num_points': int(len(group_points)),
                        'min_z': float(section_data.get('min_z', 0.0)),
                        'max_z': float(section_data.get('max_z', 0.0))
                    })

                    self.log_to_console(
                        f"[Measure] Calculated volume for tree {section_tree_id}, trunk T{trunk_id} branch B{group_branch_id}: {section_volume:.4f} m³ (height: {section_data.get('height_range', 0.0):.2f}m, Z {section_data.get('min_z', 0.0):.2f}-{section_data.get('max_z', 0.0):.2f}m)"
                    )

                    section_id += 1

                # Save trunk summary
                self.trunk_volume_summary[int(trunk_id)] = {
                    'tree_id': section_tree_id,
                    'trunk_id': int(trunk_id),
                    'total_volume': float(trunk_total_vol),
                    'num_branches': len(trunk_branch_details),
                    'total_points': int(trunk_point_count),
                    'branches': trunk_branch_details
                }

                self.log_to_console(f" Trunk T{trunk_id} total volume: {trunk_total_vol:.4f} m³ ({len(trunk_branch_details)} branches, {trunk_point_count} pts)")

            # Re-enable rendering after batch additions
            if self.plotter is not None:
                self.plotter.suppress_rendering = False
                self.plotter.render()

            # Add section info labels showing trunk/branch/section for each section
            self._add_section_info_labels()
            self._log_section_diameter_breakdown()

            # Also log overall totals
            overall_total = sum(t['total_volume'] for t in self.trunk_volume_summary.values())
            self.log_to_console(f"[Metrics] Overall tree total (sum of trunks): {overall_total:.4f} m³")

        else:
            # Fallback to original behavior (global/grouped branches)
            # Suppress rendering during batch additions for performance
            if self.plotter is not None:
                self.plotter.suppress_rendering = True

            for group_branch_id, _, group_points in volume_groups:
                point_color = section_colors[(section_id - 1) % len(section_colors)]
                circle_color = section_colors[(section_id - 1) % len(section_colors)]
                branch_suffix = f" | B{group_branch_id}" if group_branch_id is not None else ""

                # Add selected points
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

                # Add circle fitting
                measurement_label = "Diameter" if is_branch_mode else "DBH"
                section_data = self._add_dbh_circle_fitting_for_section(
                    group_points,
                    section_id,
                    circle_color,
                    avg_color,
                    measurement_label=measurement_label
                )

                # Calculate volume
                section_volume = self._calculate_section_volume(section_data)

                # Store section
                section_info = {
                    'tree_id': section_tree_id,
                    'section_id': section_id,
                    'branch_id': int(group_branch_id) if group_branch_id is not None else None,
                    'points': group_points.copy(),
                    'point_color': point_color,
                    'circle_color': circle_color,
                    'volume': section_volume,
                    'circle_data': section_data
                }
                self.accumulated_volume_sections.append(section_info)

                if group_branch_id is not None:
                    self.log_to_console(
                        f"[Measure] Calculated volume for tree {section_tree_id}, branch B{group_branch_id}: {section_volume:.4f} m³ (height: {section_data.get('height_range', 0.0):.2f}m, Z {section_data.get('min_z', 0.0):.2f}-{section_data.get('max_z', 0.0):.2f}m)"
                    )
                else:
                    self.log_to_console(f"[Measure] Calculated volume for tree {section_tree_id}: {section_volume:.4f} m³ (height: {section_data.get('height_range', 0.0):.2f}m, Z {section_data.get('min_z', 0.0):.2f}-{section_data.get('max_z', 0.0):.2f}m)")

                section_id += 1

            # Re-enable rendering after batch additions
            if self.plotter is not None:
                self.plotter.suppress_rendering = False
                self.plotter.render()

            # Add section info labels showing trunk/branch/section for each section
            self._add_section_info_labels()
            self._log_section_diameter_breakdown()

            # Perform global ground proximity check (allow in branch mode too)
            # This enables using globally fitted/extrapolated circles as DBH references
            try:
                self._check_global_ground_proximity_and_extrapolate()
            except Exception as e:
                self.log_to_console(f"[WARN] Error during global extrapolation: {e}")

            # Log extrapolation notification for any sections that were extrapolated
            for sec in self.accumulated_volume_sections:
                if sec.get('extrapolated') and sec.get('section_id') != "global_extrapolated":
                    sec_id = sec['section_id']
                    orig_hr = sec.get('original_height_range', 0.0)
                    extr_hr = sec.get('extrapolated_height_range', 0.0)
                    orig_min = sec.get('original_min_z', 0.0)
                    extr_min = sec.get('extrapolated_min_z', 0.0)
                    n_ext = sec.get('num_extrapolated_circles', 0)
                    ground = sec.get('ground_level_used', 0.0)
                    self.log_to_console(
                        f"[WARN] Section {sec_id}: EXTRAPOLATED to ground — original height: {orig_hr:.2f}m (Z {orig_min:.2f}m), "
                        f"extrapolated height: {extr_hr:.2f}m (Z {extr_min:.2f}m), "
                        f"{n_ext} circles added, ground level: {ground:.2f}m"
                    )
                elif sec.get('extrapolated') and sec.get('section_id') == "global_extrapolated":
                    extr_hr = sec.get('extrapolated_height_range', 0.0)
                    extr_min = sec.get('extrapolated_min_z', 0.0)
                    n_ext = sec.get('num_extrapolated_circles', 0)
                    ground = sec.get('ground_level_used', 0.0)
                    self.log_to_console(
                        f"[WARN] Global extrapolation section: height {extr_hr:.2f}m (Z {extr_min:.2f}m), "
                        f"{n_ext} circles added, ground level: {ground:.2f}m"
                    )

            self.plotter.update()


    def _save_volume_data_silent(self):
        """Save volume data without prompts (batch mode - reuses pre-set folder)."""
        if not self.accumulated_volume_sections:
            self.log_to_console("[WARN] No volume data to save")
            return

        if not self.volume_folder_path:
            raise RuntimeError("Volume folder path not set")

        try:
            import json
            
            # Group sections by tree_id
            trees_data = {}
            for section in self.accumulated_volume_sections:
                section_id = section['section_id']
                if section_id != "global_extrapolated":
                    try:
                        int(section_id)
                    except (ValueError, TypeError):
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

            # Calculate totals
            for tree_id, tree_data in trees_data.items():
                tree_data['total_sections'] = len(tree_data['sections'])
                tree_data['total_volume'] = sum(s['volume'] for s in tree_data['sections'])
                tree_data['total_points'] = sum(s['num_points'] for s in tree_data['sections'])

            # Save files
            saved_count = 0
            for tree_id, tree_data in trees_data.items():
                filename = os.path.join(self.volume_folder_path, f"tree_{tree_id}.json")
                
                # Check for existing file and merge
                existing_data = None
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r') as f:
                            existing_data = json.load(f)
                    except:
                        pass
                
                if existing_data and 'sections' in existing_data:
                    existing_data['sections'].extend(tree_data['sections'])
                    existing_data['total_sections'] = len(existing_data['sections'])
                    existing_data['total_volume'] = sum(s['volume'] for s in existing_data['sections'])
                    existing_data['total_points'] = sum(s['num_points'] for s in existing_data['sections'])
                    save_data = existing_data
                else:
                    save_data = tree_data
                
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                saved_count += 1
                self.log_to_console(f"[Save] Saved tree {tree_id}: {save_data['total_volume']:.4f} m³ → {filename}")

                # Also write branch and trunk CSV summaries for this tree
                try:
                    import csv

                    branch_csv = os.path.join(self.volume_folder_path, f"tree_{tree_id}_branchVolumes.csv")
                    trunk_csv = os.path.join(self.volume_folder_path, f"tree_{tree_id}_trunkVolumes.csv")

                    # Prepare branch rows
                    branch_rows = []
                    for sec in save_data.get('sections', []):
                        circ = sec.get('circle_data', {}) or {}
                        branch_rows.append({
                            'tree_id': tree_id,
                            'section_id': sec.get('section_id'),
                            'trunk_id': sec.get('trunk_id', None) if 'trunk_id' in sec else None,
                            'branch_id': sec.get('branch_id'),
                            'volume_m3': sec.get('volume', 0.0),
                            'num_points': sec.get('num_points', 0),
                            'min_z': circ.get('min_z', sec.get('original_min_z')),
                            'max_z': circ.get('max_z', sec.get('extrapolated_min_z')),
                            'height_m': circ.get('height_range', None)
                        })

                    # Write branch CSV
                    if branch_rows:
                        with open(branch_csv, 'w', newline='') as bf:
                            writer = csv.DictWriter(bf, fieldnames=list(branch_rows[0].keys()))
                            writer.writeheader()
                            for r in branch_rows:
                                writer.writerow(r)
                        self.log_to_console(f"[Save] Branch CSV saved: {branch_csv}")

                    # Prepare trunk aggregation
                    trunk_totals = {}
                    for r in branch_rows:
                        trunk_key = r.get('trunk_id', None)
                        if trunk_key is None:
                            trunk_key = -1
                        trunk = trunk_totals.setdefault(trunk_key, {'total_volume': 0.0, 'num_branches': 0, 'total_points': 0})
                        trunk['total_volume'] += float(r.get('volume_m3', 0.0) or 0.0)
                        trunk['num_branches'] += 1
                        trunk['total_points'] += int(r.get('num_points', 0) or 0)

                    # Write trunk CSV
                    trunk_rows = []
                    for t_id, vals in trunk_totals.items():
                        trunk_rows.append({
                            'tree_id': tree_id,
                            'trunk_id': t_id,
                            'total_volume_m3': vals['total_volume'],
                            'num_branches': vals['num_branches'],
                            'total_points': vals['total_points']
                        })

                    if trunk_rows:
                        with open(trunk_csv, 'w', newline='') as tf:
                            writer = csv.DictWriter(tf, fieldnames=list(trunk_rows[0].keys()))
                            writer.writeheader()
                            for r in trunk_rows:
                                writer.writerow(r)
                        self.log_to_console(f"[Save] Trunk CSV saved: {trunk_csv}")

                    # Add multi_trunk flag to saved JSON
                    multi_trunk = len([t for t in trunk_totals.keys() if t is not None and t != -1]) > 1
                    save_data['multi_trunk'] = multi_trunk
                    # Rewrite JSON with the flag
                    with open(filename, 'w') as f:
                        json.dump(save_data, f, indent=2)

                except Exception as csv_e:
                    self.log_to_console(f"[WARN] Failed to write branch/trunk CSVs for tree {tree_id}: {csv_e}")

            self.log_to_console(f"[OK] All volume data saved to {self.volume_folder_path}")

        except Exception as e:
            self.log_to_console(f"[Error] Error saving volume data: {str(e)}")
            raise



    def _save_batch_results(self, export_folder, volume_folder, batch_meta=None):
        """Save accumulated batch results as a single JSON and aggregated CSVs.

        export_folder: path to write CSVs (export/batch/<las_basename>)
        volume_folder: path to write JSON (volume/batch/<las_basename>)
        batch_meta: optional dict with metadata about the batch run
        """
        if not self.accumulated_volume_sections:
            self.log_to_console("[WARN] No accumulated volume sections to save for batch")
            return

        try:
            import json
            import csv
            import tempfile

            # Determine base name for files
            if hasattr(self, 'current_las_path') and self.current_las_path:
                las_basename = os.path.splitext(os.path.basename(self.current_las_path))[0]
            else:
                las_basename = 'batch'

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Build trees_data dictionary (exclude raw point arrays)
            trees_data = {}
            for sec in self.accumulated_volume_sections:
                section_id = sec.get('section_id')
                if section_id != 'global_extrapolated':
                    try:
                        int(section_id)
                    except (ValueError, TypeError):
                        continue

                tree_id = sec.get('tree_id')
                if tree_id not in trees_data:
                    trees_data[tree_id] = {'sections': [], 'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now())}

                circ = sec.get('circle_data') or {}
                sec_entry = {
                    'section_id': sec.get('section_id'),
                    'trunk_id': sec.get('trunk_id', None),
                    'branch_id': sec.get('branch_id', None),
                    'volume': float(sec.get('volume', 0.0) or 0.0),
                    'num_points': int(len(sec.get('points') or [])),
                    'point_color': sec.get('point_color'),
                    'circle_color': sec.get('circle_color'),
                    'circle_data': circ,
                    'extrapolated': sec.get('extrapolated', False),
                    'original_height_range': sec.get('original_height_range'),
                    'extrapolated_height_range': sec.get('extrapolated_height_range'),
                    'original_min_z': sec.get('original_min_z'),
                    'extrapolated_min_z': sec.get('extrapolated_min_z'),
                    'num_extrapolated_circles': sec.get('num_extrapolated_circles'),
                    'ground_level_used': sec.get('ground_level_used')
                }
                trees_data[tree_id]['sections'].append(sec_entry)

            # Compute per-tree totals
            total_sections = 0
            total_volume = 0.0
            for tree_id, td in trees_data.items():
                td['total_sections'] = len(td['sections'])
                td['total_volume'] = sum(s['volume'] for s in td['sections'])
                td['total_points'] = sum(s['num_points'] for s in td['sections'])
                total_sections += td['total_sections']
                total_volume += td['total_volume']

            # Build batch JSON
            batch_id = ts
            batch_obj = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'source_las': getattr(self, 'current_las_path', None),
                'batch_meta': batch_meta or {},
                'total_sections': total_sections,
                'total_volume': total_volume,
                'trees': trees_data
            }

            # Atomic write JSON
            json_name = os.path.join(volume_folder, f"{las_basename}_batch_{ts}.json")
            fd, tmp_json = tempfile.mkstemp(suffix='.json', prefix=f"{las_basename}_batch_{ts}_", dir=volume_folder)
            try:
                with os.fdopen(fd, 'w') as tf:
                    json.dump(batch_obj, tf, indent=2)
                os.replace(tmp_json, json_name)
            finally:
                if os.path.exists(tmp_json):
                    try:
                        os.remove(tmp_json)
                    except Exception:
                        pass

            # Prepare aggregated CSV rows
            branch_rows = []
            for tree_id, td in trees_data.items():
                for s in td['sections']:
                    circ = s.get('circle_data') or {}
                    branch_rows.append({
                        'tree_id': tree_id,
                        'section_id': s.get('section_id'),
                        'trunk_id': s.get('trunk_id'),
                        'branch_id': s.get('branch_id'),
                        'volume_m3': s.get('volume', 0.0),
                        'num_points': s.get('num_points', 0),
                        'min_z': circ.get('min_z', s.get('original_min_z')),
                        'max_z': circ.get('max_z', s.get('extrapolated_min_z')),
                        'height_m': circ.get('height_range', None)
                    })

            # Write branch CSV
            branch_csv = os.path.join(export_folder, f"{las_basename}_batch_{ts}_branches.csv")
            if branch_rows:
                tmp_branch = branch_csv + '.tmp'
                with open(tmp_branch, 'w', newline='') as bf:
                    writer = csv.DictWriter(bf, fieldnames=list(branch_rows[0].keys()))
                    writer.writeheader()
                    for r in branch_rows:
                        writer.writerow(r)
                os.replace(tmp_branch, branch_csv)

            # Aggregate trunk rows
            trunk_totals = {}
            for r in branch_rows:
                trunk_key = (r.get('tree_id'), r.get('trunk_id'))
                if trunk_key not in trunk_totals:
                    trunk_totals[trunk_key] = {'total_volume': 0.0, 'num_branches': 0, 'total_points': 0}
                trunk_totals[trunk_key]['total_volume'] += float(r.get('volume_m3') or 0.0)
                trunk_totals[trunk_key]['num_branches'] += 1
                trunk_totals[trunk_key]['total_points'] += int(r.get('num_points') or 0)

            trunk_rows = []
            for (tree_id, trunk_id), vals in trunk_totals.items():
                trunk_rows.append({
                    'tree_id': tree_id,
                    'trunk_id': trunk_id,
                    'total_volume_m3': vals['total_volume'],
                    'num_branches': vals['num_branches'],
                    'total_points': vals['total_points']
                })

            trunk_csv = os.path.join(export_folder, f"{las_basename}_batch_{ts}_trunks.csv")
            if trunk_rows:
                tmp_trunk = trunk_csv + '.tmp'
                with open(tmp_trunk, 'w', newline='') as tf:
                    writer = csv.DictWriter(tf, fieldnames=list(trunk_rows[0].keys()))
                    writer.writeheader()
                    for r in trunk_rows:
                        writer.writerow(r)
                os.replace(tmp_trunk, trunk_csv)

            self.log_to_console(f"[Save] Batch JSON saved: {json_name}")
            if branch_rows:
                self.log_to_console(f"[Save] Batch branch CSV saved: {branch_csv}")
            if trunk_rows:
                self.log_to_console(f"[Save] Batch trunk CSV saved: {trunk_csv}")

            self.log_to_console(f"[OK] Batch results saved to {export_folder} and {volume_folder}")

        except Exception as e:
            self.log_to_console(f"[Error] Error saving batch results: {e}")
            raise


    def export_trunk_metrics(self):
        """Export one CSV row per trunk with location, total volume, and DBH to export folder."""
        if not self.accumulated_volume_sections:
            self.log_to_console("[WARN] No volume calculation data to export.")
            return

        try:
            trunk_summary, _, _ = self._build_trunk_volume_summary_table()
            trunk_dbh_summary, _, trunk_dbh_details = self._build_trunk_dbh_summary_table()

            if not trunk_summary:
                QMessageBox.warning(
                    self,
                    "No Trunk Data",
                    "No trunk-level data found. Run trunk/branch detection and volume calculation first."
                )
                return

            # Determine export directory and filename based on loaded point cloud
            if hasattr(self, 'current_las_path') and self.current_las_path:
                las_dir = os.path.dirname(self.current_las_path)
                las_basename = os.path.splitext(os.path.basename(self.current_las_path))[0]
                export_dir = os.path.join(las_dir, 'export')
            else:
                export_dir = os.path.join(os.getcwd(), 'export')
                las_basename = 'trunk_metrics'

            # Create export directory if it doesn't exist
            os.makedirs(export_dir, exist_ok=True)

            save_path = os.path.join(export_dir, f"{las_basename}_trunk_metrics.csv")

            rows = []
            # Build a lookup of DBH records by (tree_id, trunk_id) composite key.
            # These records are exactly what the blue trunk-DBH circles are drawn from,
            # so the export stays consistent with the visualized blue circles.
            dbh_by_key = {}
            for composite_key, dbh_rec in (trunk_dbh_details or {}).items():
                try:
                    key_parts = str(composite_key).rsplit('_', 1)
                    dbh_by_key[(key_parts[0], int(key_parts[1]))] = dbh_rec
                except (ValueError, TypeError, IndexError):
                    continue

            dbh_ref_height = float(getattr(self, 'dbh_reference_height', 1.3))

            for trunk_id in sorted(trunk_summary.keys(), key=lambda x: (isinstance(x, str), x)):
                # Skip non-numeric trunk IDs (e.g., 'Multiple Trunks (26)')
                try:
                    trunk_id_int = int(trunk_id)
                except (ValueError, TypeError):
                    self.log_to_console(f"[WARN] Skipping non-numeric trunk ID: {trunk_id}")
                    continue

                summary_rec = trunk_summary[trunk_id]

                # Collect associated tree IDs from accumulated sections for this trunk
                tree_ids = []
                import re
                for sec in self.accumulated_volume_sections:
                    if sec.get('trunk_id', None) == trunk_id and sec.get('tree_id', None) is not None:
                        try:
                            sec_tree_id = str(sec.get('tree_id'))
                            found_ids = [int(x) for x in re.findall(r'\d+', sec_tree_id)]
                            if found_ids:
                                tree_ids.extend(found_ids)
                            else:
                                tree_ids.append(int(sec_tree_id))
                        except (ValueError, TypeError):
                            pass
                tree_ids = sorted(set(tree_ids))

                # Fallbacks: if no tree_ids found in sections, use the tree_id stored in trunk_volume_summary
                if not tree_ids:
                    try:
                        tinfo = getattr(self, 'trunk_volume_summary', {}).get(int(trunk_id))
                        if tinfo is not None:
                            tid = tinfo.get('tree_id')
                            if tid is not None:
                                found_ids = [int(x) for x in re.findall(r'\d+', str(tid))]
                                if found_ids:
                                    tree_ids = found_ids
                                else:
                                    tree_ids = [int(tid)]
                    except Exception:
                        pass

                # Final fallback: use current_tree_id if available
                if not tree_ids and getattr(self, 'current_tree_id', None) is not None:
                    try:
                        curr_id_str = str(self.current_tree_id)
                        found_ids = [int(x) for x in re.findall(r'\d+', curr_id_str)]
                        if found_ids:
                            tree_ids = found_ids
                        else:
                            tree_ids = [int(self.current_tree_id)]
                    except Exception:
                        pass

                # Absolute fallback: store trunk_volume_summary tree_id or current_tree_id directly in the row if still empty
                resolved_tree_ids = ';'.join(str(t) for t in tree_ids) if tree_ids else ''
                if not resolved_tree_ids:
                    try:
                        tinfo = getattr(self, 'trunk_volume_summary', {}).get(int(trunk_id))
                        if tinfo is not None and tinfo.get('tree_id') is not None:
                            tid = tinfo.get('tree_id')
                            found_ids = [int(x) for x in re.findall(r'\d+', str(tid))]
                            if found_ids:
                                resolved_tree_ids = ';'.join(str(t) for t in found_ids)
                            else:
                                resolved_tree_ids = str(int(tid))
                    except Exception:
                        pass
                if not resolved_tree_ids and getattr(self, 'current_tree_id', None) is not None:
                    try:
                        curr_id_str = str(self.current_tree_id)
                        found_ids = [int(x) for x in re.findall(r'\d+', curr_id_str)]
                        if found_ids:
                            resolved_tree_ids = ';'.join(str(t) for t in found_ids)
                        else:
                            resolved_tree_ids = str(int(self.current_tree_id))
                    except Exception:
                        pass

                # Find the DBH record (blue circle) for this trunk. Trunk IDs can
                # repeat across trees, so prefer matching by (tree_id, trunk_id).
                dbh_rec = None
                primary_tree = tree_ids[0] if tree_ids else None
                if primary_tree is not None:
                    dbh_rec = dbh_by_key.get((str(primary_tree), trunk_id_int))
                if dbh_rec is None:
                    for (_tree_key, _trunk_key), _rec in dbh_by_key.items():
                        if _trunk_key == trunk_id_int:
                            dbh_rec = _rec
                            if primary_tree is None:
                                primary_tree = _tree_key
                            break

                dbh_cm = dbh_rec.get('dbh_cm') if dbh_rec else None
                dbh_section = dbh_rec.get('section_id', '') if dbh_rec else ''
                dbh_branch = dbh_rec.get('branch_id') if dbh_rec else None
                # DBH method: 'dbh_height' when a section contains the DBH
                # reference height; 'closest_section_fallback' when no trunk
                # points existed at that height and the closest section was used.
                if dbh_rec is None:
                    dbh_method = 'no_dbh_circle'
                elif dbh_rec.get('priority', 1) == 0:
                    dbh_method = 'dbh_height'
                else:
                    dbh_method = 'closest_section_fallback'

                # Location = centroid of the blue DBH circle when available.
                center_xyz = dbh_rec.get('center_xyz') if dbh_rec else None
                location_source = 'dbh_circle' if center_xyz is not None else 'section_avg_center'

                if center_xyz is None:
                    # Fallback location from section average centers for this trunk.
                    centers = []
                    z_refs = []
                    for sec in self.accumulated_volume_sections:
                        if sec.get('trunk_id', None) != trunk_id:
                            continue
                        circle_data = sec.get('circle_data') or {}
                        avg_center = circle_data.get('avg_center')
                        if avg_center is not None and len(avg_center) >= 2:
                            centers.append([float(avg_center[0]), float(avg_center[1])])
                            z_refs.append(float(circle_data.get('min_z', 0.0)))

                    if centers:
                        centers_arr = np.array(centers, dtype=float)
                        center_xyz = [
                            float(np.mean(centers_arr[:, 0])),
                            float(np.mean(centers_arr[:, 1])),
                            float(np.mean(z_refs)) if z_refs else 0.0,
                        ]
                    else:
                        center_xyz = [None, None, None]

                # Tree height from the currently visualized tree (ground to top).
                tree_height = getattr(self, 'current_tree_height', None)

                try:
                    rows.append({
                        'tree_ids': resolved_tree_ids,
                        'trunk_id': trunk_id_int,
                        'branch_id': dbh_branch,
                        'section': dbh_section,
                        'dbh_method': dbh_method,
                        'dbh_ref_height_m': dbh_ref_height,
                        'tree_height_m': float(tree_height) if tree_height is not None else None,
                        'location_x_m': center_xyz[0],
                        'location_y_m': center_xyz[1],
                        'location_z_m': center_xyz[2],
                        'location_source': location_source,
                        'total_volume_m3': float(summary_rec.get('total_volume', 0.0)),
                        'dbh_cm': float(dbh_cm) if dbh_cm is not None else None,
                        'num_branches': int(summary_rec.get('num_branches', 0)),
                        'total_points': int(summary_rec.get('total_points', 0)),
                    })
                except (ValueError, TypeError) as row_err:
                    self.log_to_console(f"[WARN] Skipping trunk {trunk_id} due to conversion error: {row_err}")
                    continue

            # Also include any trunk that has a blue DBH circle but no volume
            # summary row, so the report is complete relative to the circles shown.
            for (dbh_tree_key, dbh_trunk_key), dbh_rec in dbh_by_key.items():
                if dbh_trunk_key in trunk_summary:
                    continue
                dbh_cm = dbh_rec.get('dbh_cm')
                center_xyz = dbh_rec.get('center_xyz')
                dbh_branch = dbh_rec.get('branch_id')
                dbh_method = 'dbh_height' if dbh_rec.get('priority', 1) == 0 else 'closest_section_fallback'
                tree_height = getattr(self, 'current_tree_height', None)
                try:
                    rows.append({
                        'tree_ids': dbh_tree_key,
                        'trunk_id': dbh_trunk_key,
                        'branch_id': dbh_branch,
                        'section': dbh_section,
                        'dbh_method': dbh_method,
                        'dbh_ref_height_m': dbh_ref_height,
                        'tree_height_m': float(tree_height) if tree_height is not None else None,
                        'location_x_m': float(center_xyz[0]) if center_xyz and len(center_xyz) >= 2 else None,
                        'location_y_m': float(center_xyz[1]) if center_xyz and len(center_xyz) >= 2 else None,
                        'location_z_m': float(center_xyz[2]) if center_xyz and len(center_xyz) >= 3 else None,
                        'location_source': 'dbh_circle',
                        'total_volume_m3': 0.0,
                        'dbh_cm': float(dbh_cm) if dbh_cm is not None else None,
                        'num_branches': 0,
                        'total_points': 0,
                    })
                except (ValueError, TypeError) as row_err:
                    self.log_to_console(f"[WARN] Skipping DBH-only trunk {dbh_trunk_key} due to conversion error: {row_err}")
                    continue

            import csv

            # If file exists, read existing rows so we can preserve prior exports and backfill blank tree IDs.
            existing_rows = []
            if os.path.exists(save_path):
                try:
                    with open(save_path, 'r', newline='') as rf:
                        reader = csv.DictReader(rf)
                        for r in reader:
                            existing_rows.append(r)
                    self.log_to_console(f"ℹ️ Found existing export file; {len(existing_rows)} existing row(s) loaded for merge/dedup.")
                except Exception as e:
                    self.log_to_console(f"[WARN] Could not read existing export file: {e}")

            # Merge by (trunk_id, tree_id) composite key to keep trunks from different trees separate.
            merged_by_trunk = {}

            def _split_tree_ids(value):
                return [s.strip() for s in str(value or '').split(';') if s.strip()]

            def _get_merge_key(trunk_id, tree_ids_str):
                """Get a unique merge key based on trunk_id and primary tree_id."""
                tree_ids = _split_tree_ids(tree_ids_str)
                primary_tree_id = tree_ids[0] if tree_ids else None
                return (trunk_id, primary_tree_id)

            for r in existing_rows:
                try:
                    trunk_int = int(r.get('trunk_id'))
                    tree_ids_str = r.get('tree_ids', '')
                    key = _get_merge_key(trunk_int, tree_ids_str)
                except Exception:
                    continue
                merged_by_trunk[key] = dict(r)

            for r in rows:
                try:
                    trunk_int = int(r.get('trunk_id'))
                    tree_ids_str = r.get('tree_ids', '')
                    key = _get_merge_key(trunk_int, tree_ids_str)
                except Exception:
                    continue

                new_tree_ids = _split_tree_ids(tree_ids_str)
                existing = merged_by_trunk.get(key)
                if existing is None:
                    merged_by_trunk[key] = dict(r)
                    continue

                # Row already exists for this trunk+tree combo; update with new data
                old_tree_ids = _split_tree_ids(existing.get('tree_ids', ''))
                combined_tree_ids = sorted(set(old_tree_ids + new_tree_ids), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
                existing['tree_ids'] = ';'.join(combined_tree_ids)

                for key_name, value in r.items():
                    if key_name == 'tree_ids':
                        continue
                    if value not in (None, ''):
                        existing[key_name] = value

            rows_to_write = [merged_by_trunk[k] for k in sorted(merged_by_trunk.keys(), key=lambda x: (x[0], int(x[1]) if x[1] and str(x[1]).isdigit() else x[1]))]

            with open(save_path, 'w', newline='') as f:
                fieldnames = [ 'tree_height_m',
                    'tree_ids', 'trunk_id', 'branch_id',
                    'section', 'dbh_method', 'dbh_ref_height_m',
                    'location_x_m', 'location_y_m', 'location_z_m', 'location_source',
                    'total_volume_m3', 'dbh_cm',
                    'num_branches', 'total_points'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows_to_write:
                    writer.writerow(row)

            self.log_to_console(f"[Save] Wrote {len(rows_to_write)} merged trunk row(s) to: {save_path}")

        except Exception as e:
            self.log_to_console(f"[Error] Error exporting trunk metrics: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export trunk metrics:\n{str(e)}")


    def _display_batch_completion_summary(self, processed_count, total_count, skipped_trees, batch_start_dt, batch_end_dt, elapsed_seconds):
        """Display completion summary in console."""
        import time

        elapsed_seconds = max(elapsed_seconds, 1e-9)
        elapsed_minutes = elapsed_seconds / 60.0
        average_trees_per_min = processed_count / elapsed_minutes if elapsed_minutes > 0 else 0.0
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))

        summary = f"\n\n{'='*60}\n[Metrics] BATCH ANALYSIS COMPLETE\n{'='*60}\n"
        summary += f"Successfully processed: {processed_count}/{total_count} trees\n"
        summary += f"Skipped: {len(skipped_trees)} trees\n"
        summary += f"Time elapsed: {elapsed_str}\n"
        summary += f"Average: {average_trees_per_min:.1f} trees/min\n"
        summary += f"{'='*60}\n"
        
        self.log_to_console(summary)

    # ======================== PHASE 1: TRUNK DETECTION ========================


