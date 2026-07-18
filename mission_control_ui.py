"""Semantic, high-DPI Mission Control shell for the simulation engine."""

from __future__ import annotations

import queue
import sys
import traceback
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:  # pragma: no cover - compatibility fallback
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt5"


APP_STYLE = """
QMainWindow, QWidget#Root {
    background: #0B0F14;
    color: #E8EEF5;
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 13px;
}
QFrame#Header, QFrame#Footer, QFrame#SidePanel, QFrame#MapPanel {
    background: #111821;
    border: 1px solid #263342;
    border-radius: 12px;
}
QLabel#AppTitle { font-size: 18px; font-weight: 700; color: #F5F8FC; }
QLabel#Muted { color: #8796A8; }
QLabel#SectionTitle { font-size: 12px; font-weight: 700; color: #9EADBE; }
QLabel#StatusPill {
    background: #17382F; color: #71E1B2; border: 1px solid #28604F;
    border-radius: 10px; padding: 5px 10px; font-weight: 700;
}
QPushButton {
    background: #19232E; color: #DCE6F1; border: 1px solid #334354;
    border-radius: 8px; padding: 8px 12px; font-weight: 600;
}
QPushButton:hover { background: #223140; border-color: #4B647B; }
QPushButton:pressed { background: #0F6C66; }
QPushButton:checked { background: #0C5F59; border-color: #2EC4B6; color: #FFFFFF; }
QTabWidget::pane { border: 0; background: transparent; }
QTabBar::tab {
    background: #151E28; color: #8191A3; padding: 8px 10px;
    border: 0; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #E8EEF5; border-bottom-color: #2EC4B6; }
QProgressBar {
    background: #17212B; border: 1px solid #2D3B49; border-radius: 6px;
    color: #DDE7F1; text-align: center; min-height: 12px;
}
QProgressBar::chunk { background: #2EC4B6; border-radius: 5px; }
QSplitter::handle { background: #0B0F14; width: 7px; }
"""


def _as_bgr(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class ImagePane(QLabel):
    def __init__(self, empty_text: str, parent=None) -> None:
        super().__init__(empty_text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 140)
        self.setStyleSheet(
            "background:#0D131A; border:1px solid #263342; border-radius:9px; color:#657487;"
        )
        self._pixmap_source: Optional[QPixmap] = None

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        bgr = _as_bgr(frame)
        if bgr is None:
            return
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
        self._pixmap_source = QPixmap.fromImage(qimage)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap_source is not None:
            self.setPixmap(
                self._pixmap_source.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()


class MapCanvas(ImagePane):
    source_pressed = Signal(int, int)
    source_moved = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__("Haritalar ve model hazırlanıyor…", parent)
        self.setAccessibleName("Operasyon haritası; tıklayarak waypoint seçin")
        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)
        self._source_origin = (0, 0)
        self._image_size = (1, 1)

    def set_map_frame(self, frame: np.ndarray, source_origin=(0, 0)) -> None:
        self._source_origin = (int(source_origin[0]), int(source_origin[1]))
        self._image_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.set_frame(frame)

    def _to_source(self, window_x: int, window_y: int) -> tuple[int, int]:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return window_x, window_y
        pix_w, pix_h = pixmap.width(), pixmap.height()
        x_offset = (self.width() - pix_w) // 2
        y_offset = (self.height() - pix_h) // 2
        image_x = int((window_x - x_offset) * self._image_size[0] / max(1, pix_w))
        image_y = int((window_y - y_offset) * self._image_size[1] / max(1, pix_h))
        return image_x + self._source_origin[0], image_y + self._source_origin[1]

    @staticmethod
    def _event_xy(event) -> tuple[int, int]:
        if hasattr(event, "position"):
            point = event.position()
            return int(point.x()), int(point.y())
        return int(event.x()), int(event.y())

    def mousePressEvent(self, event) -> None:
        x, y = self._event_xy(event)
        source_x, source_y = self._to_source(x, y)
        self.source_pressed.emit(source_x, source_y)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        x, y = self._event_xy(event)
        source_x, source_y = self._to_source(x, y)
        self.source_moved.emit(source_x, source_y)
        super().mouseMoveEvent(event)


class MetricCard(QFrame):
    def __init__(self, title: str, initial: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame{background:#151E28;border:1px solid #293747;border-radius:9px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(2)
        label = QLabel(title.upper())
        label.setStyleSheet("color:#718196;font-size:10px;font-weight:700;border:0;")
        self.value = QLabel(initial)
        self.value.setStyleSheet("color:#F1F5F9;font-size:18px;font-weight:700;border:0;")
        self.value.setAccessibleName(title)
        layout.addWidget(label)
        layout.addWidget(self.value)


class SimulationWorker(QThread):
    frame_ready = Signal(object)
    telemetry_ready = Signal(object)
    status_ready = Signal(str, str)
    failed = Signal(str)

    def __init__(self, config, simulation_main: Callable[..., None], parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._simulation_main = simulation_main
        self._key_queue: queue.Queue[int] = queue.Queue()
        self.context_holder = [None]

    def post_key(self, key: int) -> None:
        self._key_queue.put(int(key))

    def _get_key(self, wait_ms: int) -> int:
        try:
            return self._key_queue.get(timeout=max(0.001, wait_ms / 1000.0))
        except queue.Empty:
            return -1

    def _display(self, dashboard: np.ndarray, label_state: dict) -> None:
        label_state.update(scale=1.0, x_off=0, y_off=0)
        context = self.context_holder[0] or {}
        map_rect = context.get("map_rect", (0, 0, dashboard.shape[1], dashboard.shape[0]))
        x, y, width, height = [int(value) for value in map_rect]
        x = max(0, min(x, dashboard.shape[1] - 1))
        y = max(0, min(y, dashboard.shape[0] - 1))
        width = max(1, min(width, dashboard.shape[1] - x))
        height = max(1, min(height, dashboard.shape[0] - y))
        map_frame = dashboard[y : y + height, x : x + width].copy()
        self.frame_ready.emit(
            {
                "map": map_frame,
                "map_origin": (x, y),
                "observation": context.get("observation_view"),
                "template": context.get("template_strip"),
                "reference_patch": context.get("ref_patch_image"),
            }
        )

    def run(self) -> None:
        try:
            self._simulation_main(
                config=self._config,
                _display_fn=self._display,
                _getkey_fn=self._get_key,
                _use_qt=True,
                _ctx_holder=self.context_holder,
                _telemetry_fn=self.telemetry_ready.emit,
                _status_fn=self.status_ready.emit,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class MissionControlWindow(QMainWindow):
    def __init__(
        self,
        config,
        simulation_main: Callable[..., None],
        runtime_mouse_callback: Callable[..., None],
        qt_key_map: Dict[int, int],
    ) -> None:
        super().__init__()
        self._config = config
        self._runtime_mouse_callback = runtime_mouse_callback
        self._qt_key_map = qt_key_map
        self._worker = SimulationWorker(config, simulation_main, self)
        self._closing = False

        self.setWindowTitle("GPS-Denied Mission Control")
        self.setMinimumSize(1180, 760)
        self.resize(1480, 920)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.telemetry_ready.connect(self._on_telemetry)
        self._worker.status_ready.connect(self._on_status)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self.map_canvas.source_pressed.connect(self._on_map_press)
        self.map_canvas.source_moved.connect(self._on_map_move)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 12)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 11, 14, 11)
        header_layout.setSpacing(10)
        brand = QLabel("◈")
        brand.setStyleSheet("font-size:26px;color:#2EC4B6;")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("GPS-Denied Mission Control")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Görsel lokalizasyon ve otonom görev konsolu")
        subtitle.setObjectName("Muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addWidget(brand)
        header_layout.addLayout(title_box)
        header_layout.addStretch(1)

        scenario = QLabel(str(self._config.scenario_mode).upper())
        scenario.setStyleSheet(
            "background:#182635;color:#A7C7E7;border:1px solid #304A61;"
            "border-radius:10px;padding:5px 10px;font-weight:700;"
        )
        scenario.setAccessibleName("Aktif senaryo")
        self.status_pill = QLabel("BAŞLATILIYOR")
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setAccessibleName("Lokalizasyon durumu")
        header_layout.addWidget(scenario)
        header_layout.addWidget(self.status_pill)

        self.auto_button = self._control_button("Otonom", "P", True)
        self.kalman_button = self._control_button("Kalman", "K", True)
        self.kalman_button.setChecked(bool(self._config.kalman_enabled))
        self.trajectory_button = self._control_button("Rota", "T", True)
        self.trajectory_button.setChecked(bool(self._config.show_trajectory))
        self.roi_button = self._control_button("Arama alanı", "O", True)
        self.roi_button.setChecked(bool(self._config.show_roi_frame))
        for button in (
            self.auto_button,
            self.kalman_button,
            self.trajectory_button,
            self.roi_button,
        ):
            header_layout.addWidget(button)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_panel = QFrame()
        left_panel.setObjectName("SidePanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(11, 11, 11, 11)
        left_title = QLabel("GÖRSEL KANIT")
        left_title.setObjectName("SectionTitle")
        left_layout.addWidget(left_title)
        tabs = QTabWidget()
        tabs.setAccessibleName("Görsel kanıt sekmeleri")
        self.observation_view = ImagePane("Canlı görüntü bekleniyor")
        self.observation_view.setAccessibleName("Canlı gözlem görüntüsü")
        self.template_view = ImagePane("Model çıktısı bekleniyor")
        self.template_view.setAccessibleName("Model şablon çıktısı")
        self.reference_patch_view = ImagePane("Eşleşme bekleniyor")
        self.reference_patch_view.setAccessibleName("Referansta eşleşen bölge")
        tabs.addTab(self.observation_view, "Gözlem")
        tabs.addTab(self.template_view, "Model")
        tabs.addTab(self.reference_patch_view, "Eşleşme")
        left_layout.addWidget(tabs, 1)

        map_panel = QFrame()
        map_panel.setObjectName("MapPanel")
        map_layout = QVBoxLayout(map_panel)
        map_layout.setContentsMargins(10, 10, 10, 10)
        map_header = QHBoxLayout()
        map_title = QLabel("OPERASYON HARİTASI")
        map_title.setObjectName("SectionTitle")
        self.map_hint = QLabel("Haritada hedef seçmek için tıklayın")
        self.map_hint.setObjectName("Muted")
        map_header.addWidget(map_title)
        map_header.addStretch(1)
        map_header.addWidget(self.map_hint)
        map_layout.addLayout(map_header)
        self.map_canvas = MapCanvas()
        map_layout.addWidget(self.map_canvas, 1)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setAccessibleName("Başlatma ilerlemesi")
        map_layout.addWidget(self.loading_bar)

        right_panel = QFrame()
        right_panel.setObjectName("SidePanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(11, 11, 11, 11)
        right_layout.setSpacing(8)
        nav_title = QLabel("NAVİGASYON")
        nav_title.setObjectName("SectionTitle")
        right_layout.addWidget(nav_title)
        self.heading_card = MetricCard("Başlık")
        self.altitude_card = MetricCard("AGL")
        self.gsd_card = MetricCard("GSD")
        self.error_card = MetricCard("Konum hatası")
        cards_row_1 = QHBoxLayout()
        cards_row_1.addWidget(self.heading_card)
        cards_row_1.addWidget(self.altitude_card)
        cards_row_2 = QHBoxLayout()
        cards_row_2.addWidget(self.gsd_card)
        cards_row_2.addWidget(self.error_card)
        right_layout.addLayout(cards_row_1)
        right_layout.addLayout(cards_row_2)

        confidence_title = QLabel("LOKALİZASYON GÜVENİ")
        confidence_title.setObjectName("SectionTitle")
        right_layout.addWidget(confidence_title)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setAccessibleName("Lokalizasyon güven yüzdesi")
        right_layout.addWidget(self.confidence_bar)
        self.detail_label = QLabel("Model ve raster kaynakları hazırlanıyor")
        self.detail_label.setWordWrap(True)
        self.detail_label.setObjectName("Muted")
        self.detail_label.setAccessibleName("Lokalizasyon ayrıntıları")
        right_layout.addWidget(self.detail_label)
        right_layout.addStretch(1)

        splitter.addWidget(left_panel)
        splitter.addWidget(map_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([255, 900, 290])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        root_layout.addWidget(splitter, 1)

        footer = QFrame()
        footer.setObjectName("Footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(13, 7, 13, 7)
        shortcuts = QLabel("WASD hareket  •  Q/E dönüş  •  P otonom  •  K Kalman  •  ESC çıkış")
        shortcuts.setObjectName("Muted")
        self.performance_label = QLabel(f"{QT_BINDING} • bekleniyor")
        self.performance_label.setObjectName("Muted")
        self.performance_label.setAccessibleName("İşleme performansı")
        footer_layout.addWidget(shortcuts)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.performance_label)
        root_layout.addWidget(footer)

    def _control_button(self, label: str, hotkey: str, checkable: bool) -> QPushButton:
        button = QPushButton(f"{label}  {hotkey}")
        button.setCheckable(checkable)
        button.setAccessibleName(f"{label} kontrolü, kısayol {hotkey}")
        button.clicked.connect(lambda _checked=False, key=hotkey: self._worker.post_key(ord(key)))
        return button

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._worker.isRunning():
            self._worker.start()

    def _on_frame(self, bundle: Dict[str, Any]) -> None:
        self.map_canvas.set_map_frame(bundle["map"], bundle.get("map_origin", (0, 0)))
        self.observation_view.set_frame(bundle.get("observation"))
        self.template_view.set_frame(bundle.get("template"))
        self.reference_patch_view.set_frame(bundle.get("reference_patch"))
        self.loading_bar.hide()

    def _on_telemetry(self, data: Dict[str, Any]) -> None:
        self.heading_card.value.setText(str(data.get("heading", "—")))
        self.altitude_card.value.setText(f"{float(data.get('altitude_m', 0.0)):.0f} m")
        self.gsd_card.value.setText(f"{float(data.get('gsd_cm', 0.0)):.1f} cm")
        self.error_card.value.setText(f"{float(data.get('error_m', 0.0)):.1f} m")
        confidence = max(0, min(100, int(round(float(data.get("confidence", 0.0)) * 100))))
        self.confidence_bar.setValue(confidence)
        reliable = bool(data.get("reliable"))
        if reliable:
            self.status_pill.setText("KİLİT SAĞLAM")
            self.status_pill.setStyleSheet(
                "background:#17382F;color:#71E1B2;border:1px solid #28604F;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            )
        else:
            self.status_pill.setText("YENİDEN KAZANIM")
            self.status_pill.setStyleSheet(
                "background:#432D16;color:#FFCB7A;border:1px solid #765125;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            )
        scores = data.get("scores", ())
        score_text = " / ".join(f"{float(score):.3f}" for score in scores)
        self.detail_label.setText(
            f"Adım {data.get('step', 0)} • {data.get('search_mode', '—')} • "
            f"{data.get('intersection_mode', '—')}\n"
            f"Skorlar {score_text or '—'} • Eylem {data.get('action', '—')}"
        )
        self.performance_label.setText(
            f"{QT_BINDING} • {float(data.get('processing_ms', 0.0)):.0f} ms • "
            f"{data.get('backend', '—')}"
        )
        for button, state in (
            (self.auto_button, bool(data.get("autonomous"))),
            (self.kalman_button, bool(data.get("kalman_on"))),
        ):
            button.blockSignals(True)
            button.setChecked(state)
            button.blockSignals(False)

    def _on_status(self, message: str, level: str) -> None:
        self.map_canvas.setText(message)
        self.map_canvas.setAccessibleDescription(message)
        if level == "loading":
            self.loading_bar.show()
            self.status_pill.setText("YÜKLENİYOR")
        elif level == "ready":
            self.loading_bar.hide()
            self.status_pill.setText("HAZIR")

    def _on_failed(self, traceback_text: str) -> None:
        self.loading_bar.hide()
        self.status_pill.setText("HATA")
        self.status_pill.setStyleSheet(
            "background:#451D25;color:#FF9BAA;border:1px solid #7D2E3C;"
            "border-radius:10px;padding:5px 10px;font-weight:700;"
        )
        summary = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "Bilinmeyen hata"
        self.map_canvas.setText(f"Simülasyon başlatılamadı\n{summary}")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Simülasyon başlatılamadı")
        dialog.setIcon(QMessageBox.Critical)
        dialog.setText(summary)
        dialog.setDetailedText(traceback_text)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.open()

    def _context(self) -> Optional[dict]:
        return self._worker.context_holder[0]

    def _on_map_press(self, x: int, y: int) -> None:
        context = self._context()
        if context is not None:
            self._runtime_mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, context)

    def _on_map_move(self, x: int, y: int) -> None:
        context = self._context()
        if context is not None:
            self._runtime_mouse_callback(cv2.EVENT_MOUSEMOVE, x, y, 0, context)

    def keyPressEvent(self, event) -> None:
        qt_key = int(event.key())
        cv2_key = self._qt_key_map.get(qt_key)
        if cv2_key is None:
            text = event.text()
            cv2_key = ord(text) if text and len(text) == 1 else qt_key
        self._worker.post_key(cv2_key)
        event.accept()

    def closeEvent(self, event) -> None:
        if self._worker.isRunning():
            self._worker.post_key(27)
            self._worker.wait(5000)
        event.accept()

    def _on_worker_finished(self) -> None:
        if self._closing:
            self.close()


def run_mission_control(
    config,
    simulation_main: Callable[..., None],
    runtime_mouse_callback: Callable[..., None],
    qt_key_map: Dict[int, int],
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GPS-Denied Mission Control")
    window = MissionControlWindow(
        config,
        simulation_main,
        runtime_mouse_callback,
        qt_key_map,
    )
    window.show()
    return int(app.exec())
