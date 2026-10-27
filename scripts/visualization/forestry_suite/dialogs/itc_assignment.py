"""Dialog for choosing an ITC value for the current selection."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget, QPushButton, QSpinBox
)
from PyQt6.QtCore import Qt
from forestry_suite.widgets import ModernButton

class ITCAssignmentDialog(QDialog):
    """Dialog for assigning ITC values to selected points."""

    def __init__(self, unique_itc_values, parent=None):
        super().__init__(parent)
        self.unique_itc_values = sorted(unique_itc_values)
        self.selected_itc = None
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Assign ITC Value")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select ITC value to assign to selected points:")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title)

        # Create scroll area for buttons
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Add buttons for each unique ITC value
        for itc_value in self.unique_itc_values:
            button = ModernButton(f"ITC {itc_value}")
            button.clicked.connect(lambda checked, val=itc_value: self.select_itc(val))
            scroll_layout.addWidget(button)

        # Add noise button
        noise_button = ModernButton("Noise (9999)")
        noise_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 0px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        noise_button.clicked.connect(lambda: self.select_itc(9999))
        scroll_layout.addWidget(noise_button)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(400)
        layout.addWidget(scroll_area)

        # Manual input section
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Or enter custom ITC value:"))
        
        self.custom_itc_input = QSpinBox()
        self.custom_itc_input.setRange(1, 999999)
        self.custom_itc_input.setValue(1)
        self.custom_itc_input.setFixedWidth(100)
        manual_layout.addWidget(self.custom_itc_input)
        
        assign_custom_button = ModernButton("Assign Custom")
        assign_custom_button.clicked.connect(lambda: self.select_itc(self.custom_itc_input.value()))
        manual_layout.addWidget(assign_custom_button)
        
        manual_layout.addStretch()
        layout.addLayout(manual_layout)

        # Cancel button
        cancel_button = ModernButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

    def select_itc(self, itc_value):
        """Select the ITC value and close dialog."""
        self.selected_itc = itc_value
        self.accept()


