from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import sys
from finalv2 import drawing
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("2dSketchTo3dMesh")
MainWindow.resize(1440, 720)

drawButton = QtWidgets.QPushButton(MainWindow)
# pushButton.setGeometry(QtCore.QRect(100, 70, 113, 32))
drawButton.setObjectName("DrawButton")
drawButton.setText("Draw")
drawButton.setGeometry(300,500,300,150)
drawButton.clicked.connect(drawing)

MainWindow.show()
sys.exit(app.exec_())