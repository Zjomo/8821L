"""启动 PyQt 测试 UI。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pi_spectrometer.ui.main_window import main

if __name__ == "__main__":
    sys.exit(main())
