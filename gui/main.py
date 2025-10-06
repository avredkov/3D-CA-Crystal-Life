import sys
from pathlib import Path
from PySide6 import QtWidgets, QtGui

# Allow running either as `python -m gui.main` from project root
# or as `python main.py` from within the gui/ directory.
try:
    from gui.tabs.analysis import AnalysisTab
    from gui.tabs.rules import RulesTab
    from gui.tabs.simulation import SimulationTab
    from gui.tabs.about import AboutTab
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gui.tabs.analysis import AnalysisTab
    from gui.tabs.rules import RulesTab
    from gui.tabs.simulation import SimulationTab
    from gui.tabs.about import AboutTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D CA Simulator GUI")
        self.resize(1200, 600)
        # Set window icon if available
        icon_path = Path(__file__).resolve().parents[0] / "assets" / "Icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        tabs = QtWidgets.QTabWidget()
        sim = SimulationTab()
        tabs.addTab(sim, "Simulation")
        tabs.addTab(AnalysisTab(), "Analysis")
        tabs.addTab(RulesTab(), "Rule recipes")
        tabs.addTab(AboutTab(), "About")
        self.setCentralWidget(tabs)

        # Refresh recipe list/params only when Simulation tab becomes active
        def _on_tab_activated(index: int) -> None:
            try:
                if tabs.widget(index) is sim:
                    if hasattr(sim, "on_activated"):
                        sim.on_activated()
            except Exception:
                pass
        try:
            tabs.currentChanged.connect(_on_tab_activated)
        except Exception:
            pass


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    # Also set app icon for taskbar/dock
    icon_path = Path(__file__).resolve().parents[0] / "assets" / "Icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    # Bring window to front in case it opens behind others
    win.raise_()
    win.activateWindow()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())


