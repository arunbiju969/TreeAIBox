"""
Mathematical volume calculation and stem geometry estimation.

Provides pure numeric algorithms for:
- Circle fitting across horizontal height intervals
- Smalian, Huber, and frustum log volume calculation
- DBH (Diameter at Breast Height) estimation at customizable reference heights
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def fit_section_circles(
    points: np.ndarray,
    step: float = 0.1,
    thickness: float = 0.05,
    min_points: int = 3,
    min_radius: float = 0.03,
    max_radius: float = 0.50,
) -> Dict[str, Any]:
    """
    Fit bounding / centroid circles at discrete height intervals along stem points.

    Args:
        points: (N, 3) array of stem 3D coordinates.
        step: Height interval between slices in meters (default 0.1m / 10cm).
        thickness: Vertical window thickness around each height level in meters (default 0.05m / 5cm).
        min_points: Minimum points in a slice required to fit a circle.
        min_radius: Minimum allowable radius in meters (default 3cm).
        max_radius: Maximum allowable radius in meters (default 50cm).

    Returns:
        Dictionary with:
            - 'centers': List of [x, y, z] circle centers
            - 'radii': List of circle radii in meters
            - 'heights': List of relative heights from min_z
            - 'avg_radius': Mean radius across all valid circles (meters)
            - 'avg_center': [mean_x, mean_y, 0]
            - 'height_range': max_z - min_z (meters)
            - 'min_z': Lowest Z coordinate in points
            - 'max_z': Highest Z coordinate in points
    """
    if points is None or len(points) == 0:
        return {
            'centers': [],
            'radii': [],
            'heights': [],
            'avg_radius': 0.0,
            'avg_center': [0.0, 0.0, 0.0],
            'height_range': 0.0,
            'min_z': 0.0,
            'max_z': 0.0,
        }

    min_z = float(np.min(points[:, 2]))
    max_z = float(np.max(points[:, 2]))
    height_range = max_z - min_z

    circle_centers: List[List[float]] = []
    circle_radii: List[float] = []
    circle_heights: List[float] = []

    current_z = min_z
    while current_z <= max_z:
        mask = (points[:, 2] >= current_z - thickness / 2.0) & (points[:, 2] <= current_z + thickness / 2.0)
        slice_pts = points[mask]

        if len(slice_pts) >= min_points:
            cluster_xy = slice_pts[:, :2]
            centroid = np.mean(cluster_xy, axis=0)
            distances = np.linalg.norm(cluster_xy - centroid, axis=1)
            radius = float(np.max(distances))

            if min_radius <= radius <= max_radius:
                circle_centers.append([float(centroid[0]), float(centroid[1]), float(current_z)])
                circle_radii.append(radius)
                circle_heights.append(float(current_z - min_z))

        current_z += step

    avg_radius = float(np.mean(circle_radii)) if circle_radii else 0.0
    avg_center = (
        [float(np.mean([c[0] for c in circle_centers])),
         float(np.mean([c[1] for c in circle_centers])),
         0.0]
        if circle_centers else [0.0, 0.0, 0.0]
    )

    return {
        'centers': circle_centers,
        'radii': circle_radii,
        'heights': circle_heights,
        'avg_radius': avg_radius,
        'avg_center': avg_center,
        'height_range': height_range,
        'min_z': min_z,
        'max_z': max_z,
    }


def calculate_cylinder_volume(avg_radius: float, height: float) -> float:
    """
    Calculate cylinder volume: V = pi * r^2 * h.

    Args:
        avg_radius: Radius in meters.
        height: Length/height in meters.

    Returns:
        Volume in cubic meters (m³).
    """
    if avg_radius <= 0.0 or height <= 0.0:
        return 0.0
    return float(np.pi * (avg_radius ** 2) * height)


def calculate_smalian_volume(radii: List[float], heights: List[float]) -> float:
    """
    Calculate tree volume using Smalian's formula across consecutive slices:
    V_i = (pi / 2) * (r_i^2 + r_{i+1}^2) * delta_h_i

    Args:
        radii: List of circle radii in meters (sorted by height).
        heights: List of circle heights in meters.

    Returns:
        Total volume in cubic meters (m³).
    """
    if len(radii) < 2 or len(radii) != len(heights):
        if len(radii) == 1 and heights:
            return 0.0
        return 0.0

    total_volume = 0.0
    for i in range(len(radii) - 1):
        dh = abs(heights[i + 1] - heights[i])
        if dh > 0:
            area_1 = np.pi * (radii[i] ** 2)
            area_2 = np.pi * (radii[i + 1] ** 2)
            total_volume += 0.5 * (area_1 + area_2) * dh

    return float(total_volume)


def calculate_frustum_volume(radii: List[float], heights: List[float]) -> float:
    """
    Calculate tree volume using the truncated cone (frustum) formula:
    V_i = (pi * delta_h / 3) * (r_i^2 + r_i * r_{i+1} + r_{i+1}^2)

    Args:
        radii: List of circle radii in meters.
        heights: List of circle heights in meters.

    Returns:
        Total volume in cubic meters (m³).
    """
    if len(radii) < 2 or len(radii) != len(heights):
        return 0.0

    total_volume = 0.0
    for i in range(len(radii) - 1):
        dh = abs(heights[i + 1] - heights[i])
        r1, r2 = radii[i], radii[i + 1]
        total_volume += (np.pi * dh / 3.0) * (r1**2 + r1 * r2 + r2**2)

    return float(total_volume)


def estimate_dbh_at_height(
    centers: List[List[float]],
    radii: List[float],
    min_z: float,
    ref_height: float = 1.3,
    tolerance: float = 0.15,
) -> Tuple[Optional[float], Optional[List[float]]]:
    """
    Extract or interpolate Diameter at Breast Height (DBH) and circle center at ref_height above min_z.

    Args:
        centers: List of [x, y, z] circle centers.
        radii: List of circle radii.
        min_z: Base elevation of the tree stem.
        ref_height: Height above base in meters (default 1.3m).
        tolerance: Maximum vertical distance in meters to accept a fitted circle directly.

    Returns:
        Tuple of (diameter_meters: Optional[float], center_xyz: Optional[List[float]]).
    """
    if not centers or not radii:
        return None, None

    target_z = min_z + ref_height
    closest_idx = None
    min_dist = float('inf')

    for i, center in enumerate(centers):
        dist = abs(center[2] - target_z)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    if closest_idx is not None and min_dist <= tolerance:
        return float(radii[closest_idx] * 2.0), centers[closest_idx]

    # If within overall range, interpolate linearly between the closest two
    z_vals = [c[2] for c in centers]
    if min(z_vals) <= target_z <= max(z_vals):
        sorted_pairs = sorted(zip(z_vals, radii, centers), key=lambda x: x[0])
        for i in range(len(sorted_pairs) - 1):
            z1, r1, c1 = sorted_pairs[i]
            z2, r2, c2 = sorted_pairs[i + 1]
            if z1 <= target_z <= z2:
                denom = z2 - z1 if z2 != z1 else 1e-6
                t = (target_z - z1) / denom
                interp_r = r1 + t * (r2 - r1)
                interp_c = [
                    c1[0] + t * (c2[0] - c1[0]),
                    c1[1] + t * (c2[1] - c1[1]),
                    target_z,
                ]
                return float(interp_r * 2.0), interp_c

    # Fallback to closest available circle if any exists
    if closest_idx is not None:
        return float(radii[closest_idx] * 2.0), centers[closest_idx]

    return None, None
