"""
Analysis and mathematical modeling modules for the Forestry AI Suite.
"""

try:
    from forestry_suite.analysis.components import extract_components, visualize_components
    from forestry_suite.analysis.volume import (
        fit_section_circles,
        calculate_cylinder_volume,
        calculate_smalian_volume,
        calculate_frustum_volume,
        estimate_dbh_at_height,
    )
    from forestry_suite.analysis.trunk import (
        build_z_slices,
        cluster_slice_xy,
        extract_slice_data,
        assign_points_to_nearest_trajectory,
    )
except ImportError:
    from .components import extract_components, visualize_components
    from .volume import (
        fit_section_circles,
        calculate_cylinder_volume,
        calculate_smalian_volume,
        calculate_frustum_volume,
        estimate_dbh_at_height,
    )
    from .trunk import (
        build_z_slices,
        cluster_slice_xy,
        extract_slice_data,
        assign_points_to_nearest_trajectory,
    )

__all__ = [
    "extract_components",
    "visualize_components",
    "fit_section_circles",
    "calculate_cylinder_volume",
    "calculate_smalian_volume",
    "calculate_frustum_volume",
    "estimate_dbh_at_height",
    "build_z_slices",
    "cluster_slice_xy",
    "extract_slice_data",
    "assign_points_to_nearest_trajectory",
]
