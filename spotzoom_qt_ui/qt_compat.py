# =====================================================
# Qt 兼容层：PySide6 ↔ PySide2
# =====================================================
from __future__ import annotations

import sys
from typing import Any

try:
    # 优先尝试 PySide6
    from PySide6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        QObject,
        Qt,
        QProcess,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QAction, QImage, QPixmap
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
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_VERSION = "PySide6"

except ImportError:
    # PySide6 不可用，降级到 PySide2
    from PySide2.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        QObject,
        Qt,
        QProcess,
        QTimer,
        Signal,
        Slot,
    )
    from PySide2.QtGui import QImage, QPixmap
    from PySide2.QtWidgets import (
        QAction,
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
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_VERSION = "PySide2"


def qt_info() -> str:
    """返回当前使用的 Qt 版本信息"""
    import platform
    return (
        f"Qt Backend: {QT_VERSION}\n"
        f"Python: {platform.python_version()}\n"
        f"System: {platform.platform()}"
    )


def app_exec(app: QApplication) -> int:
    """
    跨版本兼容的 QApplication.exec() 调用
    
    PySide6: app.exec()
    PySide2: app.exec_()
    """
    if QT_VERSION == "PySide6":
        return app.exec()
    else:
        return app.exec_()