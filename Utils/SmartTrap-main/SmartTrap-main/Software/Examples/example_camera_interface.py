# add to the path the parent directory
import os
import pathlib
import sys

sys.path.append('./..')

import argparse
from PyQt6.QtWidgets import QApplication
from camera_interface import MainWindow
from smarttrap_tracker import ParticleCNN # add to prevent errors

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the SmartTrap GUI.")
    parser.add_argument(
        "-testmode",
        action="store_true",
        help="Run the program in test mode."
    )
    args = parser.parse_args()
    if args.testmode:
        print("Test mode enabled")
    app = QApplication(sys.argv)
    w = MainWindow(args.testmode)
    w.show()
    app.exec()
    w.c_p['program_running'] = False
