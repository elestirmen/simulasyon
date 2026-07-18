import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from mission_control_ui import MissionControlWindow
from simulasyon_yonlendirme_uclu_dashboard import SimulationConfig


def test_mission_control_exposes_semantic_controls() -> None:
    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow(
        SimulationConfig(),
        lambda **kwargs: None,
        lambda *args: None,
        {},
    )
    try:
        assert window.auto_button.accessibleName().startswith("Otonom kontrolü")
        assert window.kalman_button.accessibleName().startswith("Kalman kontrolü")
        assert window.map_canvas.accessibleName().startswith("Operasyon haritası")
        assert window.confidence_bar.accessibleName() == "Lokalizasyon güven yüzdesi"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
