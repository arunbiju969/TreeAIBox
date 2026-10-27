"""Stem component extraction and visualization (connected components)."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def extract_components(points: np.ndarray, branching_height: float, assign_radius: float = 0.5,
                      vertical_window: float = 0.2, min_vertical_overlap: float = 0.1, use_nearest_neighbor: bool = False):
    """
    Extract tree components: trunk (below branching height) and branches (above, assigned to nearest cluster).

    Args:
        points: All stem points
        branching_height: Height where branching occurs (from base)
        assign_radius: Maximum horizontal distance for assigning points to branch centroids
        vertical_window: Vertical distance window for connectivity checks
        min_vertical_overlap: Minimum vertical overlap required for connectivity
    """
    min_z = np.min(points[:, 2])
    branching_z = min_z + branching_height

    # Separate points by height
    trunk_mask = points[:, 2] <= branching_z
    trunk_points = points[trunk_mask]
    branch_candidate_points = points[~trunk_mask]

    print(f"Branching height: {branching_height:.2f}m (Z = {branching_z:.2f}m)")
    print(f"Total points: {len(points)}")
    print(f"Trunk points: {len(trunk_points)}")
    print(f"Branch candidate points: {len(branch_candidate_points)}")

    # If no branch points, return just trunk
    if len(branch_candidate_points) == 0:
        return {
            'trunk': trunk_points,
            'branches': [],
            'unassigned': np.empty((0, 3))
        }

    # Use middle slice clustering for branch identification (same as visualization)
    # Calculate middle height of the crown
    if len(branch_candidate_points) > 0:
        crown_min_z = np.min(branch_candidate_points[:, 2])
        crown_max_z = np.max(branch_candidate_points[:, 2])
        middle_z = (crown_min_z + crown_max_z) / 2

        # Extract points near middle slice for clustering (within 0.1m tolerance)
        slice_tolerance = 0.1
        slice_mask = (branch_candidate_points[:, 2] >= middle_z - slice_tolerance) & (branch_candidate_points[:, 2] <= middle_z + slice_tolerance)
        slice_points = branch_candidate_points[slice_mask]

        centroids = []
        if len(slice_points) >= 3:
            # Apply DBSCAN clustering to horizontal slice (same as visualization)
            clustering = DBSCAN(eps=0.2, min_samples=3).fit(slice_points[:, :2])
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            print(f"Found {n_clusters} branch clusters at middle slice (Z = {middle_z:.2f}m)")

            # Get centroids for each cluster
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                cluster_points = slice_points[cluster_mask]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)
        else:
            print("Warning: Insufficient points at middle slice for clustering")
            centroids = []

    if len(centroids) == 0:
        return {
            'trunk': trunk_points,
            'branches': [branch_candidate_points],
            'unassigned': np.empty((0, 3))
        }

    # Use connectivity-based approach (same as visualization)
    # Build connectivity graph for all branch candidate points
    horizontal_radius = 0.2  # 20cm horizontal connectivity radius
    vertical_radius = 0.5    # 50cm vertical connectivity radius

    print(f"Building connectivity graph with {len(branch_candidate_points)} branch candidate points...")
    print(f"Connectivity: horizontal={horizontal_radius}m, vertical={vertical_radius}m")
    n_points = len(branch_candidate_points)

    # Create adjacency matrix for 3D connectivity (horizontal + vertical)
    # Points are connected if within BOTH horizontal AND vertical radius
    rows = []
    cols = []

    for i in range(n_points):
        for j in range(i+1, n_points):
            # Check horizontal distance
            dist_xy = np.sqrt((branch_candidate_points[i, 0] - branch_candidate_points[j, 0])**2 +
                            (branch_candidate_points[i, 1] - branch_candidate_points[j, 1])**2)
            # Check vertical distance
            dist_z = abs(branch_candidate_points[i, 2] - branch_candidate_points[j, 2])
            
            if dist_xy <= horizontal_radius and dist_z <= vertical_radius:
                rows.extend([i, j])
                cols.extend([j, i])

    branch_arrays = []
    unassigned_points = []

    if len(rows) > 0:
        # Create sparse adjacency matrix
        adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_points, n_points))

        # Find connected components
        n_components, connectivity_labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)

        print(f"Found {n_components} connected components in crown (3D connectivity)")

        # For each cluster from middle slice, find connected components containing cluster points
        for cluster_id in range(len(centroids)):
            cluster_mask = labels == cluster_id
            cluster_points = slice_points[cluster_mask]
            
            if len(cluster_points) > 0:
                # Find indices of cluster points in the full branch_candidate_points array
                cluster_indices = []
                for cluster_point in cluster_points:
                    # Find this point in the full branch candidate array
                    diffs = branch_candidate_points - cluster_point
                    distances = np.linalg.norm(diffs, axis=1)
                    idx = np.argmin(distances)
                    if distances[idx] < 0.01:  # Very close match
                        cluster_indices.append(idx)
                
                if len(cluster_indices) > 0:
                    # Get the component labels for these cluster points
                    cluster_component_labels = set(connectivity_labels[idx] for idx in cluster_indices)
                    
                    # Collect all points in these connected components
                    branch_points = []
                    for comp_label in cluster_component_labels:
                        component_mask = connectivity_labels == comp_label
                        component_points = branch_candidate_points[component_mask]
                        branch_points.extend(component_points)
                    
                    if len(branch_points) > 0:
                        branch_arrays.append(np.array(branch_points))

        # Find unassigned points (not in any branch's connected components)
        assigned_indices = set()
        for branch in branch_arrays:
            for branch_point in branch:
                # Find index of this point in branch_candidate_points
                diffs = branch_candidate_points - branch_point
                distances = np.linalg.norm(diffs, axis=1)
                idx = np.argmin(distances)
                if distances[idx] < 0.01:
                    assigned_indices.add(idx)
        
        unassigned_indices = [i for i in range(len(branch_candidate_points)) if i not in assigned_indices]
        if len(unassigned_indices) > 0:
            unassigned_points = branch_candidate_points[unassigned_indices]
        else:
            unassigned_points = np.empty((0, 3))
    else:
        # No connectivity found, assign all to first branch
        branch_arrays = [branch_candidate_points]
        unassigned_points = np.empty((0, 3))

    unassigned_array = np.array(unassigned_points) if len(unassigned_points) > 0 else np.empty((0, 3))

    print(f"Assigned to {len(branch_arrays)} branches")
    print(f"Unassigned points: {len(unassigned_array)}")
    print(f"Connectivity parameters: horizontal={horizontal_radius}m, vertical={vertical_radius}m")

    return {
        'trunk': trunk_points,
        'branches': branch_arrays,
        'unassigned': unassigned_array,
        'centroids': centroids
    }


def visualize_components(components: dict, plotter, branching_height: float, centroids: list, all_points: np.ndarray, viz_mode: str = 'components'):
    """
    Visualize tree components based on the specified mode.

    Args:
        components: Dictionary with trunk, branches, and unassigned points
        plotter: PyVista plotter instance
        branching_height: Height where branching occurs
        centroids: List of branch centroids
        all_points: All original points
        viz_mode: Visualization mode ('components', 'sequence', etc.)
    """
    if viz_mode == 'components':
        # Visualize each component with different colors
        trunk_points = components['trunk']
        branches = components['branches']
        unassigned_points = components['unassigned']

        # Clear existing plot
        plotter.clear()

        # Visualize trunk
        if len(trunk_points) > 0:
            trunk_cloud = pv.PolyData(trunk_points)
            plotter.add_mesh(trunk_cloud, color='brown', point_size=4, opacity=0.9,
                           label=f'Trunk ({len(trunk_points)} pts)', render_points_as_spheres=False)

        # Visualize branches with different colors
        branch_colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, branch_points in enumerate(branches):
            if len(branch_points) > 0:
                color = branch_colors[i % len(branch_colors)]
                branch_cloud = pv.PolyData(branch_points)
                plotter.add_mesh(branch_cloud, color=color, point_size=4, opacity=0.8,
                               label=f'Branch {i+1} ({len(branch_points)} pts)', render_points_as_spheres=False)

        # Visualize unassigned points
        if len(unassigned_points) > 0:
            unassigned_cloud = pv.PolyData(unassigned_points)
            plotter.add_mesh(unassigned_cloud, color='gray', point_size=3, opacity=0.6,
                           label=f'Unassigned ({len(unassigned_points)} pts)', render_points_as_spheres=False)

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
            plotter.add_mesh(reference_plane, color='red', opacity=0.3,
                           label=f'Branching Height: {branching_height:.1f}m')

        # Add bounding box
        if len(all_points) > 0:
            bounds = [
                np.min(all_points[:, 0]), np.max(all_points[:, 0]),
                np.min(all_points[:, 1]), np.max(all_points[:, 1]),
                np.min(all_points[:, 2]), np.max(all_points[:, 2])
            ]
            bounding_box = pv.Cube(bounds=bounds)
            plotter.add_mesh(bounding_box, color='black', style='wireframe', line_width=2,
                           label='Bounding Box')

        # Calculate dimensions
        width = bounds[1] - bounds[0]
        length = bounds[3] - bounds[2]
        height = bounds[5] - bounds[4]

        info_text = f"Tree Component Analysis\n"
        info_text += f"Total stem points: {len(all_points)}\n"
        info_text += f"Trunk points: {len(trunk_points)}\n"
        for i, branch in enumerate(branches):
            info_text += f"Branch {i+1}: {len(branch)} points\n"
        info_text += f"Unassigned: {len(unassigned_points)} points\n"
        info_text += f"Dimensions: {width:.2f}m W × {length:.2f}m L × {height:.2f}m H\n"
        info_text += f"Branching height: {branching_height:.2f}m"

        plotter.add_text(info_text, position='upper_left', font_size=10, color='black')
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 30
        plotter.update()

    else:
        print(f"Visualization mode '{viz_mode}' not implemented. Use 'components' for component visualization.")


