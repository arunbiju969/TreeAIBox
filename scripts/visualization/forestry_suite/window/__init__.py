"""TreeVisualizerGUI assembled from cohesive mixins."""

from PyQt6.QtWidgets import QMainWindow
from forestry_suite.window.core import CoreMixin
from forestry_suite.window.ui import UiMixin
from forestry_suite.window.las_layers import LasLayersMixin
from forestry_suite.window.gdb import GdbMixin
from forestry_suite.window.visualization import VisualizationMixin
from forestry_suite.window.centroids import CentroidsMixin
from forestry_suite.window.selection import SelectionMixin
from forestry_suite.window.itc_edit import ItcEditMixin
from forestry_suite.window.trunk_detection import TrunkDetectionMixin
from forestry_suite.window.ai_quality import AiQualityMixin
from forestry_suite.window.batch import BatchMixin
from forestry_suite.window.volume import VolumeMixin
from forestry_suite.window.components_ui import ComponentsUiMixin


class TreeVisualizerGUI(
    CoreMixin,
    UiMixin,
    LasLayersMixin,
    GdbMixin,
    VisualizationMixin,
    CentroidsMixin,
    SelectionMixin,
    ItcEditMixin,
    TrunkDetectionMixin,
    AiQualityMixin,
    BatchMixin,
    VolumeMixin,
    ComponentsUiMixin,
    QMainWindow,
):
    """Main window for tree visualization GUI (assembled from mixins)."""
    pass
