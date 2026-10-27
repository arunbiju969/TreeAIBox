"""
Point cloud representation and I/O utilities for TreeAIBox Standalone Mode.
"""

import os
from typing import Dict, List, Optional, Union
import numpy as np


class ScalarFieldWrapper:
    """Wrapper around scalar field data replicating CloudCompare's ccScalarField interface."""

    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data)

    def asArray(self) -> np.ndarray:
        return self.data

    def computeMinAndMax(self):
        pass


class StandalonePointCloud:
    """
    Point cloud object wrapping coordinates and scalar fields for TreeAIBox standalone execution.
    Exposes method signatures compatible with CloudCompare's ccPointCloud.
    """

    def __init__(
        self,
        points: np.ndarray,
        name: str,
        file_path: Optional[str] = None,
        scalar_fields: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._points = np.asarray(points, dtype=np.float64)
        self._name = name
        self._file_path = file_path
        self._scalar_fields: Dict[str, np.ndarray] = scalar_fields if scalar_fields else {}

    def points(self) -> np.ndarray:
        return self._points

    def getName(self) -> str:
        return self._name

    def addScalarField(self, name: str) -> int:
        if name not in self._scalar_fields:
            self._scalar_fields[name] = np.zeros(len(self._points), dtype=np.float32)
        return list(self._scalar_fields.keys()).index(name)

    def getScalarFieldIndexByName(self, name: str) -> int:
        if name in self._scalar_fields:
            return list(self._scalar_fields.keys()).index(name)
        return -1

    def getScalarField(self, index: Union[int, str]) -> ScalarFieldWrapper:
        if isinstance(index, int):
            name = list(self._scalar_fields.keys())[index]
        else:
            name = index
        return ScalarFieldWrapper(self._scalar_fields[name])

    def setCurrentDisplayedScalarField(self, index: int):
        pass

    def showSF(self, show: bool):
        pass


def load_point_cloud_file(file_path: str) -> Optional[StandalonePointCloud]:
    """
    Load a point cloud from disk (LAS, LAZ, TXT, XYZ, PTS, NPY) with all scalar fields.

    Args:
        file_path: Path to the point cloud file.

    Returns:
        StandalonePointCloud instance, or None if loading failed.
    """
    if not file_path or not os.path.isfile(file_path):
        return None

    file_ext = os.path.splitext(file_path)[1].lower()
    scalar_fields: Dict[str, np.ndarray] = {}

    if file_ext in ['.txt', '.xyz', '.pts']:
        pcd = np.loadtxt(file_path)
    elif file_ext == '.npy':
        pcd = np.load(file_path)
    elif file_ext in ['.las', '.laz']:
        import laspy
        las = laspy.read(file_path)
        pcd = np.vstack([las.x, las.y, las.z]).T

        # Load RGB channels if present
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            scalar_fields['red'] = np.asarray(las.red)
            scalar_fields['green'] = np.asarray(las.green)
            scalar_fields['blue'] = np.asarray(las.blue)

        # Load extra dimensions as scalar fields
        for dim_name in las.point_format.extra_dimension_names:
            try:
                scalar_fields[dim_name] = np.asarray(getattr(las, dim_name))
            except Exception as e:
                print(f"[WARN] Could not load extra dimension '{dim_name}': {e}")
    else:
        raise ValueError(f"Unsupported point cloud format: {file_ext}")

    return StandalonePointCloud(
        points=pcd,
        name=os.path.basename(file_path),
        file_path=file_path,
        scalar_fields=scalar_fields,
    )


def save_point_cloud_to_las(pc: StandalonePointCloud, output_file: str) -> str:
    """
    Write a StandalonePointCloud to a LAS file with all scalar fields.

    Args:
        pc: StandalonePointCloud instance.
        output_file: Destination LAS file path.

    Returns:
        The output file path.
    """
    import laspy

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.min(pc._points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

    las = laspy.LasData(header)
    las.x = pc._points[:, 0]
    las.y = pc._points[:, 1]
    las.z = pc._points[:, 2]

    # Preserve RGB channels
    if 'red' in pc._scalar_fields and 'green' in pc._scalar_fields and 'blue' in pc._scalar_fields:
        try:
            las.red = pc._scalar_fields['red'].astype(np.uint16)
            las.green = pc._scalar_fields['green'].astype(np.uint16)
            las.blue = pc._scalar_fields['blue'].astype(np.uint16)
        except Exception as e:
            print(f"[WARN] Could not preserve RGB channels: {e}")

    # Add scalar fields as extra dimensions
    for field_name, field_data in pc._scalar_fields.items():
        if field_name in ['red', 'green', 'blue']:
            continue
        try:
            if field_name in ['itc', 'stemcls', 'stemoff', 'treeloc', 'treefilter']:
                dtype = np.int32
            else:
                dtype = np.float32

            las.add_extra_dim(laspy.ExtraBytesParams(name=field_name, type=dtype))
            setattr(las, field_name, field_data.astype(dtype))
        except Exception as e:
            print(f"[WARN] Could not add scalar field '{field_name}': {e}")

    las.write(output_file)
    pc._file_path = output_file
    pc._name = os.path.basename(output_file)
    return output_file
