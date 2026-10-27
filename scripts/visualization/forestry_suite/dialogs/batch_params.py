"""Parameter dialogs for batch trunk/branch analysis."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QDialogButtonBox, QPushButton
)
from PyQt6.QtCore import Qt
from forestry_suite.widgets import ModernButton

class BatchSkipOptionsDialog(QDialog):
    """Dialog to ask whether to skip certain trees during batch processing."""

    def __init__(self, parent=None, default_threshold=50):
        super().__init__(parent)
        self.default_threshold = int(default_threshold)
        self.selected = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Batch: Skip Options")
        self.setModal(True)
        layout = QVBoxLayout(self)

        title = QLabel("Batch skip options (applies to all trees):")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.skip_no_trunk_cb = QCheckBox("Skip trees with no trunk detected")
        self.skip_no_trunk_cb.setChecked(True)
        layout.addWidget(self.skip_no_trunk_cb)

        h = QHBoxLayout()
        self.skip_small_trunk_cb = QCheckBox("Skip trees when largest trunk has <")
        self.skip_small_trunk_cb.setChecked(True)
        h.addWidget(self.skip_small_trunk_cb)
        self.small_trunk_threshold = QSpinBox()
        self.small_trunk_threshold.setRange(1, 1000000)
        self.small_trunk_threshold.setValue(self.default_threshold)
        self.small_trunk_threshold.setSuffix(" pts")
        h.addWidget(self.small_trunk_threshold)
        layout.addLayout(h)

        info = QLabel("These settings are applied once at the start for all trees in the batch.")
        layout.addWidget(info)

        btns = QHBoxLayout()
        ok = ModernButton("OK")
        ok.clicked.connect(self.accept)
        btns.addWidget(ok)
        cancel = ModernButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def get_options(self):
        return {
            'skip_no_trunk': bool(self.skip_no_trunk_cb.isChecked()),
            'skip_small_trunk': bool(self.skip_small_trunk_cb.isChecked()),
            'small_trunk_threshold': int(self.small_trunk_threshold.value())
        }



class BatchTrunkParamsDialog(QDialog):
    """Dialog to configure trunk-detection parameters for batch runs."""

    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.defaults = defaults or {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Batch: Trunk Detection Parameters")
        self.setModal(True)
        layout = QVBoxLayout(self)

        # Slice height (cm)
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Slice height (cm):"))
        self.slice_height_spin = QSpinBox()
        self.slice_height_spin.setRange(1, 200)
        self.slice_height_spin.setValue(int(self.defaults.get('slice_height_cm', 25)))
        h1.addWidget(self.slice_height_spin)
        layout.addLayout(h1)

        # Base zone height (m)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Base zone height (m, 0=full):"))
        self.base_zone = QDoubleSpinBox()
        self.base_zone.setRange(0.0, 100.0)
        self.base_zone.setDecimals(2)
        self.base_zone.setValue(float(self.defaults.get('base_zone_height_m', 0.0)))
        h2.addWidget(self.base_zone)
        layout.addLayout(h2)

        # DBSCAN eps and min_samples
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("DBSCAN eps (m):"))
        self.eps = QDoubleSpinBox()
        self.eps.setRange(0.0, 10.0)
        self.eps.setDecimals(3)
        self.eps.setValue(float(self.defaults.get('eps_m', 0.18)))
        h3.addWidget(self.eps)
        h3.addWidget(QLabel("min_samples:"))
        self.min_samples = QSpinBox()
        self.min_samples.setRange(1, 1000)
        self.min_samples.setValue(int(self.defaults.get('min_samples', 8)))
        h3.addWidget(self.min_samples)
        layout.addLayout(h3)

        # Merge distance, min points, min span
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("merge_dist (m):"))
        self.merge_dist = QDoubleSpinBox()
        self.merge_dist.setRange(0.0, 10.0)
        self.merge_dist.setDecimals(3)
        self.merge_dist.setValue(float(self.defaults.get('merge_dist_m', 0.25)))
        h4.addWidget(self.merge_dist)
        h4.addWidget(QLabel("min_trunk_points:"))
        self.min_trunk_points = QSpinBox()
        self.min_trunk_points.setRange(1, 100000)
        self.min_trunk_points.setValue(int(self.defaults.get('min_trunk_points', 120)))
        h4.addWidget(self.min_trunk_points)
        layout.addLayout(h4)

        h5 = QHBoxLayout()
        h5.addWidget(QLabel("min_vertical_span_m:"))
        self.min_span = QDoubleSpinBox()
        self.min_span.setRange(0.0, 100.0)
        self.min_span.setDecimals(2)
        self.min_span.setValue(float(self.defaults.get('min_vertical_span_m', 1.0)))
        h5.addWidget(self.min_span)
        h5.addWidget(QLabel("full_height_assign_dist_m:"))
        self.full_assign = QDoubleSpinBox()
        self.full_assign.setRange(0.0, 10.0)
        self.full_assign.setDecimals(3)
        self.full_assign.setValue(float(self.defaults.get('full_height_assign_dist_m', 0.45)))
        h5.addWidget(self.full_assign)
        layout.addLayout(h5)

        # DBH reference height (m) — asked here too so batch always uses an
        # explicit value even if the user forgot to set it in the main window.
        h6 = QHBoxLayout()
        h6.addWidget(QLabel("DBH reference height (m):"))
        self.dbh_height = QDoubleSpinBox()
        self.dbh_height.setRange(0.1, 10.0)
        self.dbh_height.setSingleStep(0.1)
        self.dbh_height.setDecimals(2)
        self.dbh_height.setValue(float(self.defaults.get('dbh_reference_height', 1.3)))
        self.dbh_height.setToolTip("DBH reference height above ground (standard: 1.3m). Used for DBH calculation and export location/DBH details.")
        h6.addWidget(self.dbh_height)
        layout.addLayout(h6)

        btns = QHBoxLayout()
        ok = ModernButton("OK")
        ok.clicked.connect(self.accept)
        btns.addWidget(ok)
        cancel = ModernButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def get_params(self):
        return {
            'slice_height_cm': int(self.slice_height_spin.value()),
            'base_zone_height_m': float(self.base_zone.value()),
            'eps_m': float(self.eps.value()),
            'min_samples': int(self.min_samples.value()),
            'merge_dist_m': float(self.merge_dist.value()),
            'min_trunk_points': int(self.min_trunk_points.value()),
            'min_vertical_span_m': float(self.min_span.value()),
            'full_height_assign_dist_m': float(self.full_assign.value()),
            'dbh_reference_height': float(self.dbh_height.value())
        }



class BatchBranchParamsDialog(QDialog):
    """Dialog to configure branch-detection parameters for batch runs."""

    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.defaults = defaults or {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Batch: Branch Detection Parameters")
        self.setModal(True)
        layout = QVBoxLayout(self)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Slice height (cm):"))
        self.slice_height = QSpinBox()
        self.slice_height.setRange(1, 200)
        self.slice_height.setValue(int(self.defaults.get('slice_height_cm', 25)))
        h1.addWidget(self.slice_height)
        layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Branch DBSCAN eps (m):"))
        self.branch_eps = QDoubleSpinBox()
        self.branch_eps.setRange(0.0, 10.0)
        self.branch_eps.setDecimals(3)
        self.branch_eps.setValue(float(self.defaults.get('branch_eps', 0.12)))
        h2.addWidget(self.branch_eps)
        h2.addWidget(QLabel("min_samples:"))
        self.branch_min_samples = QSpinBox()
        self.branch_min_samples.setRange(1, 1000)
        self.branch_min_samples.setValue(int(self.defaults.get('branch_min_samples', 6)))
        h2.addWidget(self.branch_min_samples)
        layout.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("merge_dist (m):"))
        self.branch_merge = QDoubleSpinBox()
        self.branch_merge.setRange(0.0, 10.0)
        self.branch_merge.setDecimals(3)
        self.branch_merge.setValue(float(self.defaults.get('branch_merge_dist', 0.25)))
        h3.addWidget(self.branch_merge)
        layout.addLayout(h3)

        btns = QHBoxLayout()
        ok = ModernButton("OK")
        ok.clicked.connect(self.accept)
        btns.addWidget(ok)
        cancel = ModernButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def get_params(self):
        return {
            'slice_height_cm': int(self.slice_height.value()),
            'branch_eps': float(self.branch_eps.value()),
            'branch_min_samples': int(self.branch_min_samples.value()),
            'branch_merge_dist': float(self.branch_merge.value())
        }



