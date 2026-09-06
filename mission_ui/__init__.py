"""Mission UI 1.0: shared desktop shell for visual and terrain navigation.

This package is vendored identically in the simulator repositories so each app can be
installed independently. See DESIGN_SYSTEM.md for the synchronization contract.
"""

import sys

try:
    if "PyQt5.QtWidgets" in sys.modules:
        raise ImportError("Use the Qt binding already selected by the application")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout
except ImportError:  # pragma: no cover
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

COLORS = {
    "background": "#0B1120", "panel": "#111C2E", "well": "#0D1728",
    "border": "#26364C", "text": "#E8EFF8", "muted": "#93A8C2",
    "accent": "#4DD9C0", "success": "#79E2B1", "warning": "#F2C078",
    "danger": "#FF8E9B", "estimate": "#B5A0FF",
}

APP_STYLE = """
QWidget { color: #E8EFF8; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
QMainWindow, QWidget#Root { background: #0B1120; }
QLabel { background: transparent; border: none; }
QFrame#Header, QFrame#Footer, QFrame#SidePanel, QFrame#MapPanel {
    background: #111C2E; border: 1px solid #26364C; border-radius: 12px;
}
QFrame#Header { border-top: 2px solid #4DD9C0; }
QLabel#Brand { color: #4DD9C0; font-size: 22px; font-weight: 700; border: 1px solid #2D6659; border-radius: 8px; padding: 4px 9px; background: #15372F; }
QLabel#AppTitle { color: #E8EFF8; font-size: 23px; font-weight: 700; }
QLabel#Muted, QLabel#titleLabel { color: #93A8C2; font-size: 11px; }
QLabel#SectionTitle { color: #B3C7DD; font-size: 11px; font-weight: 700; }
QLabel#ControlGroupLabel { color: #93A8C2; font-size: 10px; font-weight: 700; }
QLabel#MethodBadge { color: #4DD9C0; background: #15372F; border: 1px solid #2D6659;
    border-radius: 7px; padding: 6px 12px; font-size: 11px; font-weight: 700; }
QLabel#StatusPill { color: #93A8C2; background: #1B2A40; border: 1px solid #344862;
    border-radius: 7px; padding: 6px 12px; font-weight: 700; }
QFrame#ControlGroup, QFrame#EvidenceCard, QFrame#MetricCard {
    background: #152238; border: 1px solid #26364C; border-radius: 9px;
}
QLabel#MetricValue { color: #E8EFF8; font-size: 24px; font-weight: 600; }
QLabel#metricLabel { color: #E8EFF8; font-size: 13px; font-weight: 600; }
QLabel#ImageWell { background: #0D1728; color: #93A8C2;
    border: 1px solid #26364C; border-radius: 8px; }
QPushButton { background: #1B2A40; color: #C7D7E9; border: 1px solid #344862;
    border-radius: 7px; padding: 8px 12px; font-weight: 600; }
QPushButton:hover { background: #243952; border-color: #7595B6; }
QPushButton:pressed { background: #304A62; }
QPushButton:focus, QComboBox:focus { border: 1px solid #4DD9C0; }
QPushButton:checked, QPushButton#btnStart { background: #4DD9C0; color: #092820;
    border-color: #4DD9C0; }
QPushButton:checked:hover, QPushButton#btnStart:hover { background: #80E8D5; }
QPushButton[controlRole="visual"]:checked { background: #254C57; color: #ABEEE5; border-color: #3C7981; }
QPushButton#btnStop { background: #39232F; color: #FFABB5; border-color: #6B3B4B; }
QPushButton#btnStop:hover { background: #57303E; }
QPushButton:disabled, QPushButton#btnStart:disabled, QPushButton#btnStop:disabled { background: #152035; color: #63748B; border-color: #26364C; }
QGroupBox { background: #111C2E; border: 1px solid #26364C; border-radius: 9px;
    margin-top: 20px; padding: 12px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #B3C7DD; }
QComboBox { background: #152238; border: 1px solid #344862; border-radius: 6px;
    padding: 7px 24px 7px 9px; min-height: 16px; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background: #1B2A40; color: #E8EFF8; selection-background-color: #254C57; }
QComboBox:disabled, QCheckBox:disabled { color: #63748B; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #7595B6; border-radius: 4px; background: #152238; }
QCheckBox::indicator:checked { background: #4DD9C0; border-color: #4DD9C0; }
QProgressBar { background: #0D1728; border: 1px solid #26364C; border-radius: 6px;
    color: #E8EFF8; text-align: center; min-height: 18px; }
QProgressBar::chunk { background: #298E82; border-radius: 5px; }
QSplitter::handle { background: #0B1120; }
QSplitter::handle:horizontal { width: 8px; }
QSplitter::handle:vertical { height: 8px; }
QSplitter::handle:hover { background: #298E82; }
QTabWidget::pane { border: 1px solid #26364C; background: #111C2E; border-radius: 8px; }
QTabBar::tab { background: #111C2E; color: #93A8C2; padding: 9px 12px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #4DD9C0; border-bottom-color: #4DD9C0; }
QTabBar::tab:hover { background: #1B2A40; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #111C2E; border: none; }
QScrollBar:vertical { width: 8px; background: #111C2E; margin: 0; }
QScrollBar:horizontal { height: 8px; background: #111C2E; margin: 0; }
QScrollBar::handle { background: #344862; border-radius: 4px; min-height: 24px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QTextEdit, QPlainTextEdit, QTableWidget { background: #0D1728; alternate-background-color: #152238;
    border: 1px solid #26364C; border-radius: 7px; padding: 6px; selection-background-color: #254C57; }
QTextEdit, QPlainTextEdit { font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px; }
QTableWidget { gridline-color: #26364C; }
QHeaderView::section { background: #1B2A40; color: #B3C7DD; border: none; padding: 7px; }
QToolTip { background: #243952; color: #E8EFF8; border: 1px solid #7595B6; padding: 7px; }
"""


def create_header(method: str, subtitle: str):
    """Return the shared identity shell, its layout and the live status label."""
    frame = QFrame()
    frame.setObjectName("Header")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 13, 18, 13)
    layout.setSpacing(12)
    row = QHBoxLayout()
    row.setSpacing(14)
    brand = QLabel("N")
    brand.setObjectName("Brand")
    row.addWidget(brand, 0, Qt.AlignVCenter)
    titles = QVBoxLayout()
    titles.setSpacing(2)
    title = QLabel("MISSION CONTROL")
    title.setObjectName("AppTitle")
    description = QLabel(subtitle)
    description.setObjectName("Muted")
    titles.addWidget(title)
    titles.addWidget(description)
    row.addLayout(titles)
    row.addStretch(1)
    badge = QLabel(method)
    badge.setObjectName("MethodBadge")
    badge.setAccessibleName("Konumlama yöntemi")
    row.addWidget(badge, 0, Qt.AlignVCenter)
    status = QLabel("HAZIR")
    status.setObjectName("StatusPill")
    status.setAccessibleName("Lokalizasyon durumu")
    row.addWidget(status, 0, Qt.AlignVCenter)
    layout.addLayout(row)
    return frame, layout, status


def create_footer(shortcuts: str):
    frame = QFrame()
    frame.setObjectName("Footer")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(14, 9, 14, 9)
    hint = QLabel(shortcuts)
    hint.setObjectName("Muted")
    hint.setWordWrap(True)
    layout.addWidget(hint, 1)
    performance = QLabel("NAVIGATION LAB")
    performance.setObjectName("Muted")
    layout.addWidget(performance)
    return frame, performance


def set_status(label, text: str, tone: str = "muted") -> None:
    label.setText(text)
    color = COLORS.get(tone, COLORS["muted"])
    label.setStyleSheet(
        f"color:{color};background:#1B2A40;border:1px solid #344862;"
        "border-radius:7px;padding:6px 12px;font-weight:700;"
    )


class MetricCard(QFrame):
    """A shared metric tile; an existing telemetry label can be reused directly."""

    def __init__(self, title: str, initial: str = "—", parent=None, value_label=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(3)
        label = QLabel(title.upper())
        label.setObjectName("ControlGroupLabel")
        self.value = value_label if value_label is not None else QLabel(initial)
        self.value.setObjectName("MetricValue")
        self.value.setAccessibleName(title)
        layout.addWidget(label)
        layout.addWidget(self.value)
