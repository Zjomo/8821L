"""Qt 绑定兼容层。

自动检测并使用可用的 Qt 绑定：PyQt6 > PyQt5 > PySide6 > PySide2。
这样可以在 Anaconda（通常预装 PyQt5）和标准 Python 环境（PyQt6）中运行。
"""

from __future__ import annotations

import sys
from typing import Any

QtCore: Any = None
QtGui: Any = None
QtWidgets: Any = None
QT_BINDER = ""


def _try_import(module_name: str) -> bool:
    global QtCore, QtGui, QtWidgets, QT_BINDER
    try:
        core = __import__(f"{module_name}.QtCore", fromlist=["QtCore"])
        gui = __import__(f"{module_name}.QtGui", fromlist=["QtGui"])
        widgets = __import__(f"{module_name}.QtWidgets", fromlist=["QtWidgets"])
        # 额外校验：访问 QLibraryInfo 会强制加载 Qt 核心 DLL，
        # 若 DLL 损坏/不匹配则可在此捕获并回退到下一个绑定
        _ = core.QLibraryInfo
        QtCore, QtGui, QtWidgets, QT_BINDER = core, gui, widgets, module_name
        return True
    except Exception:
        QtCore, QtGui, QtWidgets, QT_BINDER = None, None, None, ""
        return False


for binder in ("PyQt6", "PyQt5", "PySide6", "PySide2"):
    if _try_import(binder):
        break

if not QT_BINDER:
    raise ImportError(
        "未找到可用的 Qt 绑定。请安装 PyQt6 或 PyQt5：\n"
        "  pip install PyQt6 pyqtgraph\n"
        "或\n"
        "  pip install PyQt5 pyqtgraph"
    )


# ---------------------------------------------------------------------------
# 兼容性封装
# ---------------------------------------------------------------------------
def Signal(*types, **kwargs):
    """跨绑定信号。"""
    if QT_BINDER.startswith("PyQt"):
        return QtCore.pyqtSignal(*types, **kwargs)
    else:
        return QtCore.Signal(*types, **kwargs)


def Slot(*types, **kwargs):
    """跨绑定槽。"""
    if QT_BINDER.startswith("PyQt"):
        return QtCore.pyqtSlot(*types, **kwargs)
    else:
        return QtCore.Slot(*types, **kwargs)


# 枚举兼容性
if hasattr(QtCore.Qt, "AlignmentFlag"):
    AlignCenter = QtCore.Qt.AlignmentFlag.AlignCenter
else:
    AlignCenter = QtCore.Qt.AlignCenter

if hasattr(QtCore.Qt, "DateFormat"):
    ISODate = QtCore.Qt.DateFormat.ISODate
    DefaultLocaleLongDate = QtCore.Qt.DateFormat.DefaultLocaleLongDate
else:
    ISODate = QtCore.Qt.ISODate
    DefaultLocaleLongDate = QtCore.Qt.DefaultLocaleLongDate


# QRunnable / QThreadPool 兼容
QRunnable = QtCore.QRunnable
QThreadPool = QtCore.QThreadPool
QObject = QtCore.QObject
QTimer = QtCore.QTimer
QDateTime = QtCore.QDateTime

# QAction 在 PyQt5 中位于 QtWidgets，在 PyQt6 中位于 QtGui
if QT_BINDER == "PyQt5":
    QAction = QtWidgets.QAction
else:
    QAction = QtGui.QAction
