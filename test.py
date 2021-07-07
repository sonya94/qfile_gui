import sys
import os
from PySide2 import QtUiTools, QtGui
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
from tkinter import *
from tkinter import filedialog


class MainView(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        global UI_set

        UI_set = QtUiTools.QUiLoader().load(resource_path("filedialog.ui"))

        # self.tr 메서드는 QMainWindow object를 받아 실행하기 때문에

        # 클래스 내부에 FilesOpen01 ~ FilesOpen06 메서드를 만듬
        UI_set.BTN_filedialog_s1.clicked.connect(self.FilesOpen01)
        UI_set.BTN_filedialog_s2.clicked.connect(self.FilesOpen02)
        UI_set.BTN_filedialog_s3.clicked.connect(self.FilesOpen03)

        UI_set.BTN_filedialog_ns1.clicked.connect(self.FilesOpen04)
        UI_set.BTN_filedialog_ns2.clicked.connect(self.FilesOpen05)
        UI_set.BTN_filedialog_ns3.clicked.connect(self.FilesOpen06)

        UI_set.BTN_tkinter_1.clicked.connect(FileOpen01)
        UI_set.BTN_tkinter_2.clicked.connect(FileOpen02)
        UI_set.BTN_tkinter_3.clicked.connect(FileOpen03)

        self.setCentralWidget(UI_set)
        self.setWindowTitle("GUI Program Test")
        self.setWindowIcon(QtGui.QPixmap(resource_path("./images/jbmpa.png")))
        self.resize(730, 420)
        self.show()

    # static function 단일 파일
    def FilesOpen01(self):
        fileName = QFileDialog.getOpenFileName(self, self.tr("Open Data files"), "./",
                                               self.tr("Data Files (*.csv *.xls *.xlsx);; Images (*.png *.xpm *.jpg *.gif);; All Files(*.*)"))
        UI_set.TBrowser.setText(str(fileName))

    # static function 다중 파일
    def FilesOpen02(self):
        fileNames = QFileDialog.getOpenFileNames(self, self.tr("Open Data files"), "./",
                                                 self.tr(
                                                     "Data Files (*.csv *.xls *.xlsx);; Images (*.png *.xpm *.jpg *.gif);; All Files(*.*)"))
        UI_set.TBrowser.setText(str(fileNames))

    # static function 디렉토리
    def FilesOpen03(self):
        dirName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./",
                                                   QFileDialog.ShowDirsOnly)
        UI_set.TBrowser.setText(str(dirName))

    # Non static function 단일 파일
    def FilesOpen04(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(
            self.tr("Data Files (*.csv *.xls *.xlsx);; Images (*.png *.xpm *.jpg *.gif);; All Files(*.*)"))
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            fileName = dialog.selectedFiles()
            UI_set.TBrowser.setText(str(fileName))

    # Non static function 다중 파일
    def FilesOpen05(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter(
            self.tr("Data Files (*.csv *.xls *.xlsx);; Images (*.png *.xpm *.jpg *.gif);; All Files(*.*)"))
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            UI_set.TBrowser.setText(str(fileNames))

    # Non static function 단일 파일
    def FilesOpen06(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            dirName = dialog.selectedFiles()
            UI_set.TBrowser.setText(str(dirName))


# tkinter 단일 파일
def FileOpen01():
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir="./", title="Open Data files",
                                               filetypes=(("data files", "*.csv;*.xls;*.xlsx"), ("all files", "*.*")))

    UI_set.TBrowser.setText(str(root.filename))


# tkinter 다중 파일
def FileOpen02():
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilenames(initialdir="./", title="Open Data files",
                                                filetypes=(("data files", "*.csv;*.xls;*.xlsx"), ("all files", "*.*")))

    UI_set.TBrowser.setText(str(root.filename))


# tkinter 디렉토리 파일
def FileOpen03():
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askdirectory(initialdir="./", title="Open Data files")

    UI_set.TBrowser.setText(str(root.filename))


# 파일 경로
# pyinstaller로 원파일로 압축할때 경로 필요함
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainView()
    # main.show()
    sys.exit(app.exec_())