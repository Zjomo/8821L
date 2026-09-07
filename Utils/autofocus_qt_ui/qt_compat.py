# =====================================================
# Qt 兼容层：优先 PySide6，降级 PySide2
# =====================================================
from __future__ import annotations

import sys

try:
    from PySide6 import QtCore, QtGui
    from PySide6.QtCore import (
        QCoreApplication,
        QObject,
        Qt,
        QThread,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor, QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSlider,
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
    from PySide2 import QtCore, QtGui
    from PySide2.QtCore import (
        QCoreApplication,
        QObject,
        Qt,
        QThread,
        QTimer,
        Signal,
        Slot,
    )
    from PySide2.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor, QIcon
    from PySide2.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSlider,
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
