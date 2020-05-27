# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RoboConGui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(250, 275)
        MainWindow.setStyleSheet(_fromUtf8("background-color: qconicalgradient(cx:0.375, cy:0.340909,"
        " angle:333.1, stop:0 rgba(77, 80, 0, 208), stop:0.36 rgba(247, 255, 33, 255), stop:0.375 "
        "rgba(2, 0, 200, 255), stop:0.645 rgba(28, 45, 255, 164), stop:0.65 rgba(255, 0, 0, 255), stop:1"
        " rgba(88, 0, 0, 207));\n"""))

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.newGame = QtGui.QPushButton(self.centralwidget)
        self.newGame.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5,"
        " radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(236, 255, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
        "border: 1px transparant black;\n"
        "color: black;\n"
        "font: 75 20pt \"Purisa\";"))

        self.newGame.setObjectName(_fromUtf8("newGame"))
        self.verticalLayout.addWidget(self.newGame)
        self.alignBoard = QtGui.QPushButton(self.centralwidget)
        self.alignBoard.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5,"
        " radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 80, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
        "border: 1px transparant black;\n"
        "color: black;\n"
        "font: 75 20pt \"Purisa\";"))

        self.alignBoard.setObjectName(_fromUtf8("alignBoard"))
        self.verticalLayout.addWidget(self.alignBoard)
        self.calCam = QtGui.QPushButton(self.centralwidget)
        self.calCam.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, "
        "radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(236, 255, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
        "border: 1px transparant black;\n"
        "color: black;\n"
        "font: 75 20pt \"Purisa\";"))

        self.calCam.setObjectName(_fromUtf8("calCam"))
        self.verticalLayout.addWidget(self.calCam)
        self.exitGame = QtGui.QPushButton(self.centralwidget)
        self.exitGame.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, "
        "radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 80, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
        "border: 1px transparant black;\n"
        "color: black;\n"
        "font: 75 20pt \"Purisa\";"))

        self.exitGame.setObjectName(_fromUtf8("exitGame"))
        self.verticalLayout.addWidget(self.exitGame)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.newGame.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>", None))
        self.newGame.setText(_translate("MainWindow", "New Game", None))
        self.alignBoard.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>", None))
        self.alignBoard.setText(_translate("MainWindow", "Align Board", None))
        self.calCam.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>", None))
        self.calCam.setText(_translate("MainWindow", "Calibrate Camera", None))
        self.exitGame.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>", None))
        self.exitGame.setText(_translate("MainWindow", "Exit Game", None))
