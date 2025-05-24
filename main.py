from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit
import sys
from draw import drawing  
from gravity import drawing2
from cut import draw_and_extrude_curve
import re

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2dSketchTo3dMesh")
        self.resize(1440, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Input Bar
        input_bar = QLineEdit()
        input_bar.setPlaceholderText("Enter the name of mesh object.") 
        input_bar.setFixedHeight(40)
        input_bar.setFixedWidth(400)

        input_layout = QHBoxLayout()
        input_layout.addStretch()
        input_layout.addWidget(input_bar)
        input_layout.addStretch()

        layout.addSpacing(50)          
        layout.addLayout(input_layout) 
        layout.addSpacing(50)          

        # Buttons
        drawButton = QPushButton("Draw")
        drawButton.setFixedSize(300, 150)
        drawButton.clicked.connect(lambda: drawing(re.sub(r'\s+', '', input_bar.text()) or "teddy"))  
        drawButton.setStyleSheet("font-size: 20px;")

        cutButton = QPushButton("Cut")
        cutButton.setFixedSize(300, 150)
        cutButton.clicked.connect(lambda: draw_and_extrude_curve(re.sub(r'\s+', '', input_bar.text()) or "teddy"))
        cutButton.setStyleSheet("font-size: 20px;")

        gravityButton = QPushButton("Interact")
        gravityButton.setFixedSize(300, 150)
        gravityButton.clicked.connect(lambda: drawing2())
        gravityButton.setStyleSheet("font-size: 20px;")

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(drawButton)
        button_layout.addSpacing(100)
        button_layout.addWidget(cutButton)
        button_layout.addSpacing(100)
        button_layout.addWidget(gravityButton)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        layout.addStretch()  


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
