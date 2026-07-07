# =====================================================
# Qt 兼容层：优先 PySide6，降级 PySide2
# =====================================================
from __future__ import annotations

import sys

try:
    from PySide6.QtCore import (
        QObject,
        Qt,
        QThread,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_VERSION = "PySide6"

except ImportError:
    from PySide2.QtCore import (
        QObject,
        Qt,
        QThread,
        QTimer,
        Signal,
        Slot,
    )
    from PySide2.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
    from PySide2.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_VERSION = "PySide2"


def app_exec(app: QApplication) -> int:
    if QT_VERSION == "PySide6":
        return app.exec()
    return app.exec_()
