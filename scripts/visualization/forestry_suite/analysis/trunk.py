"""
Stem and trunk trajectory detection algorithms.

Provides pure numeric algorithms for:
- Z-coordinate slice generation
- Horizontal 2D slice clustering via DBSCAN
- Vertical trajectory tracing of tree stems across slices
- Point-to-trunk assignment based on spatial proximity
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


def build_z_slices(points: np.ndarray, slice_height_m: float) -> Tuple[np.ndarray, float, float, np.ndarray, int]:
    """
    Build z-slice boundaries for a given vertical slice height.

    Args:
        points: (N, 3) array of 3D point coordinates.
        slice_height_m: Thickness of each horizontal slice in meters.

    Returns:
        Tuple of:
            - z_coords: (N,) array of Z coordinates
            - min_z: Lowest Z value
            - max_z: Highest Z value
            - slice_boundaries: Array of slice edge coordinates
            - num_slices: Total count of slices
    """
    z_coords = points[:, 2]
    min_z = float(np.min(z_coords))
    max_z = float(np.max(z_coords))
    slice_boundaries = np.arange(min_z, max_z + slice_height_m, slice_height_m)
    num_slices = max(1, len(slice_boundaries) - 1)
    return z_coords, min_z, max_z, slice_boundaries, num_slices


def cluster_slice_xy(slice_points_xy: np.ndarray, eps_m: float, min_samples: int) -> Tuple[np.ndarray, int, int]:
    """
    Run DBSCAN clustering on 2D horizontal coordinates of a single slice.

    Args:
        slice_points_xy: (M, 2) array of X, Y coordinates in the slice.
        eps_m: DBSCAN neighborhood search radius in meters.
        min_samples: Minimum cluster core samples.

    Returns:
        Tuple of:
            - labels: (M,) cluster assignments (-1 = noise)
            - num_clusters: Total distinct clusters identified
            - num_noise: Count of noise points
    """
    if len(slice_points_xy) < min_samples:
        return np.full(len(slice_points_xy), -1, dtype=int), 0, len(slice_points_xy)

    dbscan = DBSCAN(eps=eps_m, min_samples=min_samples)
    labels = dbscan.fit_predict(slice_points_xy)
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    num_noise = int(np.sum(labels == -1))
    return labels, num_clusters, num_noise


def extract_slice_data(
    points: np.ndarray,
    slice_height_m: float = 0.25,
    eps_m: float = 0.18,
    min_samples: int = 8,
) -> List[Dict[str, Any]]:
    """
    Slice 3D points vertically and cluster each slice in 2D.

    Args:
        points: (N, 3) coordinates.
        slice_height_m: Thickness of each slice in meters.
        eps_m: DBSCAN search radius.
        min_samples: Minimum cluster core points.

    Returns:
        List of slice dictionaries with keys:
            'slice_idx', 'z_start', 'z_end', 'points', 'labels',
            'centroids', 'num_clusters', 'num_noise'
    """
    z_coords, min_z, max_z, boundaries, num_slices = build_z_slices(points, slice_height_m)
    slices: List[Dict[str, Any]] = []

    for i in range(num_slices):
        z_start = boundaries[i]
        z_end = boundaries[i + 1]
        mask = (z_coords >= z_start) & (z_coords < z_end)
        slice_pts = points[mask]

        if len(slice_pts) == 0:
            slices.append({
                'slice_idx': i,
                'z_start': z_start,
                'z_end': z_end,
                'points': np.empty((0, 3)),
                'labels': np.empty(0, dtype=int),
                'centroids': [],
                'num_clusters': 0,
                'num_noise': 0,
            })
            continue

        labels, num_clusters, num_noise = cluster_slice_xy(slice_pts[:, :2], eps_m, min_samples)

        centroids = []
        for cid in range(num_clusters):
            c_mask = labels == cid
            c_pts = slice_pts[c_mask]
            if len(c_pts) > 0:
                c_xyz = np.mean(c_pts, axis=0)
                centroids.append({
                    'cluster_id': cid,
                    'centroid': c_xyz,
                    'point_count': len(c_pts),
                })

        slices.append({
            'slice_idx': i,
            'z_start': z_start,
            'z_end': z_end,
            'points': slice_pts,
            'labels': labels,
            'centroids': centroids,
            'num_clusters': num_clusters,
            'num_noise': num_noise,
        })

    return slices


def assign_points_to_nearest_trajectory(
    points: np.ndarray,
    trunk_models: Dict[int, Dict[str, Any]],
    max_distance_m: float = 0.45,
) -> np.ndarray:
    """
    Assign points to the closest trunk model based on 2D horizontal distance to trajectory centroids.

    Args:
        points: (N, 3) point coordinates.
        trunk_models: Dict mapping trunk_id -> dict with 'trajectory' (list of centroids).
        max_distance_m: Maximum allowable distance from trajectory to claim a point.

    Returns:
        Array of length N with assigned trunk IDs (-1 = unassigned / noise).
    """
    n_points = len(points)
    assignments = np.full(n_points, -1, dtype=np.int32)

    if not trunk_models or n_points == 0:
        return assignments

    # Collect all trajectory centroids with their trunk ID
    centroid_coords = []
    centroid_trunk_ids = []

    for tid, model in trunk_models.items():
        traj = model.get('trajectory', [])
        for c in traj:
            centroid_coords.append(c)
            centroid_trunk_ids.append(tid)

    if not centroid_coords:
        return assignments

    centroid_array = np.asarray(centroid_coords)
    trunk_id_array = np.asarray(centroid_trunk_ids, dtype=np.int32)

    # Use KDTree for spatial query
    kdtree = cKDTree(centroid_array)
    distances, indices = kdtree.query(points, k=1)

    valid_mask = distances <= max_distance_m
    assignments[valid_mask] = trunk_id_array[indices[valid_mask]]

    return assignments
