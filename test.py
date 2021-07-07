import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout ,QMainWindow, QAction, qApp
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # ===== Button =====
        btn1 = QPushButton('&Button1', self)
        btn1.setCheckable(True)
        btn1.toggle()

        # ===== Radio Button =====
        rbtn1 = QRadioButton("First Radio Button", self)
        rbtn1.setChecked(True)

        # ===== Check Box =====
        check1 = QCheckBox('show title', self)
        check1.toggle()
        check1.stateChanged.connect(self.changeTitle)


        # ===== Label =====
        label1 = QLabel('Button', self)


        # ===== Status Bar Action =====
        exitAction = QAction(QIcon('./exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip("Exit Application")
        exitAction.triggered.connect(qApp.quit)
        self.statusBar()

        # ===== Menu Bar =====
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu("&File")
        filemenu.addAction(exitAction)

        # ===== Layout =====
        btn1.move(100, 50)
        label1.move(20, 50)
        check1.move(100, 100)
        rbtn1.move(200,100)

        vbox = QVBoxLayout()
        vbox.addWidget(btn1)



        self.setLayout(vbox)
        self.setWindowTitle('My First Application')
        self.setGeometry(300, 300, 500, 200)
        # self.move(300, 300)
        # self.resize(400, 200)
        self.show()

    def changeTitle(self, state):
        if state == Qt.Checked:
            self.setWindowTitle('QCheckBox')
        else:
            self.setWindowTitle(" ")

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())