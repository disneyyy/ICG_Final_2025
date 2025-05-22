from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit
import sys
from draw import drawing  # 假設你已有此函式
from gravity import drawing2
from cut import draw_and_extrude_curve
import re

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2dSketchTo3dMesh")
        self.resize(1440, 720)

        # 中央 widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 垂直總體 layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # 👉 加入上方輸入欄（Input Bar）
        input_bar = QLineEdit()
        input_bar.setPlaceholderText("Enter the name of mesh object.")  # 顯示提示文字
        input_bar.setFixedHeight(40)
        input_bar.setFixedWidth(400)

        # 加在水平方塊中間
        input_layout = QHBoxLayout()
        input_layout.addStretch()
        input_layout.addWidget(input_bar)
        input_layout.addStretch()

        layout.addSpacing(50)          # 與視窗頂端距離
        layout.addLayout(input_layout) # 插入輸入欄排版
        layout.addSpacing(50)          # 與按鈕間距

        # 👉 按鈕區塊
        drawButton = QPushButton("Draw")
        drawButton.setFixedSize(300, 150)
        drawButton.clicked.connect(lambda: drawing(re.sub(r'\s+', '', input_bar.text()) or "teddy"))  # 連結到繪圖函式
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
        layout.addStretch()  # 下方彈性空間

# 主程式執行區
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
