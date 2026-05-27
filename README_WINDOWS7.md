# Windows 7 运行指南

## 问题
您遇到的错误：
```
ImportError: DLL load failed while importing QtCore: 找不到指定的程序。
```

**原因**：PySide6（基于 Qt 6）**不再官方支持 Windows 7**，最低要求是 Windows 10 1809。

---

## 解决方案（3 步）

### 第 1 步：安装 Windows 7 兼容的依赖包

在 Windows 7 机器上打开 PowerShell 或命令提示符，运行：

```bash
# 进入项目目录
cd F:\CWB\8821L_20260523

# 安装兼容包
pip install -r requirements-win7.txt
```

如果网络慢，可以使用清华源：
```bash
pip install -r requirements-win7.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 第 2 步：修改 `app.py` 的导入语句

打开 `spotzoom_qt_ui/app.py`，修改开头的导入部分（第 8-41 行）：

**修改前（原版）：**
```python
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QProcess, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    ...
)
```

**修改后（使用兼容层）：**
```python
from .qt_compat import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    QProcess,
    QTimer,
    QAction,
    QImage,
    QPixmap,
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
```

同样，**修改 `picomotor_driver_panel.py`**（第 6-23 行）：
```python
from .qt_compat import (
    Qt,
    QTimer,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
```

---

### 第 3 步：运行应用

在 PowerShell 中运行：

```bash
python -m spotzoom_qt_ui
```

或者直接运行：

```bash
python spotzoom_qt_ui/__main__.py
```

---

## 备选方案（如果兼容层太麻烦）

直接全局替换 PySide6 → PySide2：

使用 VS Code 或其他编辑器，在 `spotzoom_qt_ui/` 目录下进行**全局查找替换**：
- 查找：`from PySide6`
- 替换：`from PySide2`

这会一次性修改所有导入语句，然后重新运行即可。

---

## 验证

运行后，您可以在界面的日志或状态中看到：
```
Qt Backend: PySide2
```

这表明系统正在使用 Windows 7 兼容的 PySide2 运行。

---

## 依赖包版本说明

| 包 | 版本范围 | 说明 |
|----|---------|------|
| PySide2 | 5.15.x | 基于 Qt 5，完美支持 Windows 7 |
| numpy | 1.19.x-1.24.x | 兼容 Python 3.8 和 Windows 7 |
| opencv-python | 4.5.x-4.8.x | 图像处理库 |

---

## 常见问题

**Q：为什么不用 PyQt5？**  
A：PyQt5 也是一个选择，但许可证是 GPL，而 PySide2 是 LGPL（更宽松），与项目现有代码兼容性更好。

**Q：安装 PySide2 失败？**  
A：确保您的 pip 是最新的：`python -m pip install --upgrade pip`，然后重试。

**Q：还会有其他问题吗？**  
A：项目使用的都是基础 Qt 控件，PySide2 完全兼容，不会有其他问题。
