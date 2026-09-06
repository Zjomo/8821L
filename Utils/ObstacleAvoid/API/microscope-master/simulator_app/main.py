#!/usr/bin/env python3

"""显微镜 + 位移台模拟器入口。

用法::

    python simulator_app/main.py

依赖：PyQt5、numpy、scipy、Pillow（数据库文件默认生成在脚本目录 microscope.db）。
"""

import logging
import os
import sys
import threading
import argparse

# 保证可直接运行：microscope 包的路径由 devices.py 自动定位
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if __package__ in (None, ""):
    sys.path.insert(0, _PARENT)

from PyQt5.QtWidgets import QApplication

try:
    from .api import DEFAULT_STAGE_LIMITS, MicroscopeSimulator, SimulatorConfig
    from .gui import MainWindow
except ImportError:  # direct ``python simulator_app/main.py`` compatibility
    from simulator_app.api import DEFAULT_STAGE_LIMITS, MicroscopeSimulator, SimulatorConfig
    from simulator_app.gui import MainWindow

# 位移台量程（µm）：样本为 3000x2000 px、像素大小 0.5 µm/px = 1500x1000 µm
_STAGE_LIMITS = dict(DEFAULT_STAGE_LIMITS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="microscope-master simulator GUI")
    parser.add_argument("--db", default=os.path.join(_HERE, "microscope.db"),
                        help="SQLite state file, or :memory:")
    parser.add_argument("--slow-preview", action="store_true",
                        help="simulate camera exposure delay")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 后台预热 scipy（首次 Z 轴离焦的高斯模糊需要，避免第一帧离焦卡顿）
    threading.Thread(
        target=lambda: __import__("scipy.ndimage", fromlist=["x"]), daemon=True
    ).start()

    app = QApplication(sys.argv)
    simulator = MicroscopeSimulator(SimulatorConfig(
        db_path=args.db, stage_limits=_STAGE_LIMITS,
        fast_preview=not args.slow_preview))
    win = MainWindow(db=simulator.db, stage=simulator.stage,
                     camera=simulator.camera)
    # Keep the facade alive with the Qt window and release the SQLite
    # connection on shutdown.  The GUI owns device shutdown; the facade owns
    # the database connection.
    win._simulator = simulator
    win.show()
    try:
        return app.exec_()
    finally:
        simulator.close()


if __name__ == "__main__":
    sys.exit(main())
