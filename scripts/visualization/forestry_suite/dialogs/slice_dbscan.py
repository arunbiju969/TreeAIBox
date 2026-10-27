"""Dialog for stepping through per-slice DBSCAN clustering results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)


class SliceDBSCANViewer(QDialog):
    """Dialog to step through per-slice DBSCAN results."""

    def __init__(self, parent, slices):
        super().__init__(parent)
        self.slices = slices
        self.idx = 0
        self.setWindowTitle("Slice DBSCAN Viewer")
        self.resize(600, 520)

        layout = QVBoxLayout(self)
        control = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.info_label = QLabel("")
        control.addWidget(self.prev_btn)
        control.addWidget(self.next_btn)
        control.addWidget(self.info_label)
        control.addStretch()
        layout.addLayout(control)

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.prev_btn.clicked.connect(self.prev)
        self.next_btn.clicked.connect(self.next)

        self.update_plot()

    def update_plot(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        s = self.slices[self.idx]
        pts = s['points']
        labs = s['labels']
        if pts is None or len(pts) == 0:
            ax.text(0.5, 0.5, 'No points in this slice', ha='center', va='center')
        else:
            try:
                unique = np.unique(labs)
                colors = plt.get_cmap('tab20')(np.linspace(0, 1, max(2, len(unique))))
                for i, uid in enumerate(unique):
                    mask = labs == uid
                    if uid == -1:
                        ax.scatter(pts[mask, 0], pts[mask, 1], c='k', s=6, label='noise')
                    else:
                        c = colors[i % len(colors)]
                        ax.scatter(pts[mask, 0], pts[mask, 1], color=[c], s=8, label=f'c{uid}')
            except Exception:
                ax.scatter(pts[:, 0], pts[:, 1], s=8)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend(loc='upper right', fontsize='small')

        ax.set_title(
            f"Slice {s['slice_idx']}: Z {s['z_start']:.2f}-{s['z_end']:.2f} | "
            f"clusters {s['num_clusters']} noise {s['num_noise']}"
        )
        self.canvas.draw()
        self.info_label.setText(f"{self.idx+1}/{len(self.slices)}")

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.update_plot()

    def next(self):
        if self.idx < len(self.slices) - 1:
            self.idx += 1
            self.update_plot()
