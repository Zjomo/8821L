"""Qt 兼容层：按本机实际部署选择绑定（环境为 PyQt5，PySide6 为后备）。

不在调用方绕过兼容层直接导入某一绑定（审计 P1 问题的修复约定）。
"""
from __future__ import annotations

BINDING = None

try:  # 本机环境首选 PyQt5
    from PyQt5 import QtCore, QtGui, QtWidgets
    BINDING = "PyQt5"
except ImportError:  # pragma: no cover - 后备绑定
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        BINDING = "PySide6"
    except ImportError:
        from PySide2 import QtCore, QtGui, QtWidgets  # pragma: no cover
        BINDING = "PySide2"

Signal = QtCore.Signal if hasattr(QtCore, "Signal") else QtCore.pyqtSignal
Slot = QtCore.Slot if hasattr(QtCore, "Slot") else QtCore.pyqtSlot
Qt = QtCore.Qt
