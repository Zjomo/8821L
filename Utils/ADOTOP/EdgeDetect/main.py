import sys

from PySide6.QtWidgets import QApplication

from ui_module import SegmentationWindow


def main():
    app = QApplication(sys.argv)
    window = SegmentationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
