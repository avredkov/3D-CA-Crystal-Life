from __future__ import annotations

from typing import Optional
from PySide6 import QtWidgets, QtCore, QtGui
from pathlib import Path


class AboutTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        # Image (optional)
        img_label = QtWidgets.QLabel()
        img_label.setAlignment(QtCore.Qt.AlignCenter)
        assets = Path(__file__).resolve().parents[1] / "assets" / "about.png"
        if assets.exists():
            img_label.setPixmap(QtGui.QPixmap(str(assets)).scaledToWidth(420, QtCore.Qt.SmoothTransformation))
        # Text
        label = QtWidgets.QLabel(
            """
            GUI for 3D Cellular Automaton (CA) + Monte Carlo Crystal Growth Simulator<br/>
            by A. Redkov, V. Ivanov, A. Pimpinelli, V. Tonchev, 2025.<br/><br/>
            please report bugs to <a href=\"mailto:avredkov@gmail.com\">avredkov@gmail.com</a><br/>
            and look for updates at <a href=\"https://github.com/avredkov/3D_CA_Crystal_Growth\">github.com/avredkov/3D_CA_Crystal_Growth</a>
            """
        )
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.RichText)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        label.setAlignment(QtCore.Qt.AlignCenter)
        # Center vertically: stretch, image, spacing, text, stretch
        layout.addStretch(1)
        layout.addWidget(img_label)
        layout.addSpacing(12)
        layout.addWidget(label)
        layout.addStretch(1)


