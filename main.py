# main.py
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui_main import MainWindow
from typing import cast
import sys

QApplication.setAttribute(cast(Qt.ApplicationAttribute, Qt.AA_EnableHighDpiScaling), True)
QApplication.setAttribute(cast(Qt.ApplicationAttribute, Qt.AA_UseHighDpiPixmaps), True)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
