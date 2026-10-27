"""
LAS and Point Cloud Utility Functions for Forestry AI Applications.

Provides unified tree ID field discovery, dimension normalization,
and point classification filtering across TreeAIBox, TreeLearn, and ForestFormer.
"""

from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import laspy


# Known candidate names for tree ID dimensions in forestry LAS files
TREE_ID_EXACT_CANDIDATES = [
    'itc',
    'treeID',
    'TreeID',
    'tree_id',
    'treeId',
    'InstanceID',
    'instance_id',
    'ClusterID',
    'cluster_id',
    'cluster',
    'treeoff',
    'pred_itc',
    'pred_treeitc',
    'itc_id',
]

TREE_ID_LOWERCASE_CANDIDATES = [
    'itc',
    'treeid',
    'tree_id',
    'instanceid',
    'instance_id',
    'clusterid',
    'cluster_id',
    'cluster',
    'treeoff',
    'pred_itc',
    'pred_treeitc',
    'itc_id',
]


def find_tree_id_dimension(dimensions: Iterable[str]) -> Optional[str]:
    """
    Identify the scalar field containing tree instance IDs from LAS dimension names.

    Prioritizes exact matches, then falls back to case-insensitive matching.

    Args:
        dimensions: Iterable of dimension name strings (e.g., las.point_format.dimension_names).

    Returns:
        The matched dimension name as it appears in the LAS file, or None if not found.
    """
    dim_list = list(dimensions)

    # 1. Exact match check
    for cand in TREE_ID_EXACT_CANDIDATES:
        if cand in dim_list:
            return cand

    # 2. Case-insensitive fallback
    lower_map = {d.lower(): d for d in dim_list}
    for cand in TREE_ID_LOWERCASE_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand]

    return None


def get_tree_id_array(las_data) -> Optional[np.ndarray]:
    """
    Extract the tree ID array from a laspy LAS data object.

    Args:
        las_data: A laspy LasData object.

    Returns:
        Numpy array of tree IDs (dtype int32), or None if no tree ID dimension exists.
    """
    if las_data is None:
        return None

    dim_names = list(las_data.point_format.dimension_names)
    dim = find_tree_id_dimension(dim_names)
    if dim is None:
        return None

    return np.asarray(las_data[dim], dtype=np.int32)


def ensure_itc_dimension(las_data, target_dim: str = 'itc') -> Tuple[bool, Optional[str]]:
    """
    Ensure the target tree ID dimension (default 'itc') exists on a LasData object.

    If missing, it searches for a compatible tree ID field and clones it to target_dim.

    Args:
        las_data: A laspy LasData object.
        target_dim: Dimension name to guarantee (default 'itc').

    Returns:
        Tuple of (success: bool, source_dimension_used: Optional[str]).
    """
    if las_data is None:
        return False, None

    dim_names = list(las_data.point_format.dimension_names)
    if target_dim in dim_names:
        return True, target_dim

    source_dim = find_tree_id_dimension(dim_names)
    if source_dim is None:
        return False, None

    try:
        source_array = np.asarray(las_data[source_dim])
        dtype = source_array.dtype
        # If float, convert to int32 for discrete tree IDs
        if np.issubdtype(dtype, np.floating):
            source_array = source_array.astype(np.int32)
            dtype = np.int32

        las_data.add_extra_dim(laspy.ExtraBytesParams(name=target_dim, type=dtype))
        las_data[target_dim] = source_array
        return True, source_dim
    except Exception as e:
        print(f"[WARN] Failed to map dimension '{source_dim}' to '{target_dim}': {e}")
        return False, source_dim


def detect_tree_filter_field(dimensions: Iterable[str]) -> Tuple[Optional[str], int]:
    """
    Detect the most appropriate scalar field and threshold value for filtering tree points.

    Priority:
        1. 'treefilter' (TreeAIBox: tree points = 2)
        2. 'semantic_pred' (Semantic Segmentation: 0 = ground, 1 = trunk, 2 = foliage)
        3. 'stemcls' (TreeLearn: stem points = 2)

    Args:
        dimensions: Iterable of dimension names.

    Returns:
        Tuple of (field_name: Optional[str], target_value: int).
    """
    dim_list = list(dimensions)
    if 'treefilter' in dim_list:
        return 'treefilter', 2
    if 'semantic_pred' in dim_list:
        return 'semantic_pred', 2
    if 'stemcls' in dim_list:
        return 'stemcls', 2
    return None, 2


def get_tree_point_mask(las_data,
                        filter_field: Optional[str] = None,
                        filter_value: int = 2) -> Optional[np.ndarray]:
    """
    Compute a boolean mask over all points in a LasData object marking tree points.

    Args:
        las_data: LasData object.
        filter_field: Scalar field name to filter on. If None, auto-detected.
        filter_value: Target scalar value representing trees (default 2).

    Returns:
        Boolean numpy array of shape (N,) or None if no filter applies.
    """
    if las_data is None:
        return None

    dims = list(las_data.point_format.dimension_names)
    if filter_field is None:
        filter_field, filter_value = detect_tree_filter_field(dims)

    if not filter_field or filter_field not in dims:
        return None

    vals = np.asarray(las_data[filter_field])
    if filter_field == 'semantic_pred' and int(filter_value) == 2:
        return (vals == 1) | (vals == 2)
    return vals == int(filter_value)


def get_unique_tree_ids(tree_ids: np.ndarray, exclude_unassigned: bool = True) -> List[int]:
    """
    Return sorted list of unique integer tree IDs.

    Args:
        tree_ids: Array of tree ID integers.
        exclude_unassigned: If True, filters out values <= 0 (unassigned / noise).

    Returns:
        List of unique positive integer IDs sorted ascending.
    """
    if tree_ids is None or len(tree_ids) == 0:
        return []

    unique_vals = np.unique(tree_ids)
    if exclude_unassigned:
        unique_vals = unique_vals[unique_vals > 0]

    return [int(v) for v in sorted(unique_vals)]
