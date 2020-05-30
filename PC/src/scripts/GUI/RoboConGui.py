#################################################################
##                AUTHOR : FAISAL FAZAL-UR-REHMAN              ##
#################################################################
# THIS WAS NOT DEVELOPED AS PART OF THIS PROJECT.               #
#                                                               #
# Please refer to the RoboCon (Vision) repository link below:   #
# https://github.com/Faisal-f-rehman/10538828_RoboConVision     #
#################################################################
##                              PC                             ##
#################################################################

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RoboConGui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
#######################
from time import sleep
import os
######################

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
        MainWindow.setStyleSheet(_fromUtf8("background-color: qconicalgradient(cx:0.375, cy:0.340909, angle:333.1, stop:0 rgba(77, 80, 0, 208), stop:0.36 rgba(247, 255, 33, 255), stop:0.375 rgba(2, 0, 200, 255), stop:0.645 rgba(28, 45, 255, 164), stop:0.65 rgba(255, 0, 0, 255), stop:1 rgba(88, 0, 0, 207));\n"
""))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.newGame = QtGui.QPushButton(self.centralwidget)
        self.newGame.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(236, 255, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
"border: 1px transparant black;\n"
"color: black;\n"
"font: 75 20pt \"Purisa\";"))
        self.newGame.setObjectName(_fromUtf8("newGame"))
        self.verticalLayout.addWidget(self.newGame)
        self.alignBoard = QtGui.QPushButton(self.centralwidget)
        self.alignBoard.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 80, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
"border: 1px transparant black;\n"
"color: black;\n"
"font: 75 20pt \"Purisa\";"))
        self.alignBoard.setObjectName(_fromUtf8("alignBoard"))
        self.verticalLayout.addWidget(self.alignBoard)
        self.calCam = QtGui.QPushButton(self.centralwidget)
        self.calCam.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(236, 255, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
"border: 1px transparant black;\n"
"color: black;\n"
"font: 75 20pt \"Purisa\";"))
        self.calCam.setObjectName(_fromUtf8("calCam"))
        self.verticalLayout.addWidget(self.calCam)
        self.exitGame = QtGui.QPushButton(self.centralwidget)
        self.exitGame.setStyleSheet(_fromUtf8("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 80, 80, 255), stop:0.995 rgba(255, 255, 255, 72));\n"
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
        self.guiInitComms(MainWindow) ####################################################

#------------------------------------------------------------------------------#
    def guiInitComms(self, MainWindow):
        self.write2File("guiComms.txt","busy\n")
        self.write2File("guiState.txt","busy\n")

        guiComms = "busy"
        while((guiComms == "busy") or (guiComms == "newData")):
            guiComms = self.readFile("guiComms.txt")


        guiState = "busy"
        while(guiState == "busy"):
            guiState = self.readFile("guiState.txt")


        self.newGame.clicked.connect(self.runNewGame)
        self.alignBoard.clicked.connect(self.runAlignBoard)
        self.calCam.clicked.connect(self.runCalCam)
        self.exitGame.clicked.connect(self.runExitGame)
    	

    #-----------------------------------------------------------#
    def runNewGame(self):
        self.disableButtons()
        #print("I am in runNewGame")
        taskComplete = self.readFile("guiState.txt")
        print(taskComplete)
        if ((taskComplete == "complete") or (taskComplete == "complete\n")):
            self.checkBusyFlag()
            self.write2File("guiState.txt","newGame")
            self.setNewDataFlag()
            print("New game initiated... please check second terminal for instructions on how to exit the game.")
        else:
            print("Still completing last task...")
        self.enableButtons()


    def runAlignBoard(self):
        self.disableButtons()
        #print("I am in runAlignBoard")
        taskComplete = self.readFile("guiState.txt")
        if ((taskComplete == "complete") or (taskComplete == "complete\n")):
            self.checkBusyFlag()
            self.write2File("guiState.txt","align")
            self.setNewDataFlag()
            print("Alignment sequence initiated... please align the circles with the grid by positioning the board so that each circle fits inside it's box. Once alignment is complete follow instructions on the second terminal.")
        else:
            print("Still completing last task...")
        self.enableButtons()


    def runCalCam(self):
        self.disableButtons()
        
        taskComplete = self.readFile("guiState.txt")
        #print(taskComplete)
        if ((taskComplete == "complete") or (taskComplete == "complete\n")):
            self.checkBusyFlag()
            self.write2File("guiState.txt","calibrate")
            self.setNewDataFlag()
            print("Calibration initiated... calibration sequence for yellow discs will run first followed by calibration sequence for red discs. This sequence will runs until calibration is successful or until i and j both are = 255, which ever is first. Please check the second terminal for progress.")
        else:
            print("Still completing last task...")
        self.enableButtons()


    def runExitGame(self):
        self.disableButtons()
        self.exitGame.setEnabled(False)
        taskComplete = self.readFile("guiState.txt")
        #print("I am in runExitGame")
        
        if ((taskComplete == "complete") or (taskComplete == "complete\n")):
            self.checkBusyFlag()
            self.write2File("guiState.txt","exit")
            self.setNewDataFlag()
            print("Closing game")
            #___________________________________
            if os.path.exists("guiComms.txt"):
                os.remove("guiComms.txt")
                print("Exit sequence 1 of 2 successful")
            else:
                print("Could not delete guiComms.txt, file not found")
            
            #___________________________________
            if os.path.exists("guiState.txt"):
                os.remove("guiState.txt")
                print("Exit sequence 2 of 2 successful")
            else:
                print("Could not delete guiState.txt, file not found")
            #-----------------------------------

            sys.exit();

        else:
            print("Still completing last task...")

        self.exitGame.setEnabled(True);
        self.enableButtons()
    #-----------------------------------------------------------#
    def disableButtons(self):
        self.newGame.setEnabled(False)
        self.alignBoard.setEnabled(False)
        self.calCam.setEnabled(False)
        #self.exitGame.setEnabled(False)

    #------------------------------------------#
    def enableButtons(self):
        self.newGame.setEnabled(True)
        self.alignBoard.setEnabled(True)
        self.calCam.setEnabled(True)
        #self.exitGame.setEnabled(True)

    def write2File(self,filename,text):
        try:
            myfile = open(filename, "w")
        except IOError:
            print "Could not open to write to file!"

        with myfile:
            myfile.write(text)
            myfile.close()
    #------------------------------------------#
    def readFile(self,filename):
        try:
            myfile = open(filename, "r")
        except IOError:
            print "Could not open to read file!"

        with myfile:
            self.dataRead = myfile.read()
            myfile.close()
            return self.dataRead
    #------------------------------------------#
    def checkBusyFlag(self):
        data = "busy"
        while ((data == "busy") or (data == "newData")):
            data = self.readFile("guiComms.txt");
            sleep(0.3)
    #------------------------------------------#
    def setNewDataFlag(self):
        self.checkBusyFlag()
        self.write2File("guiComms.txt","newData")



#------------MAIN------------#
if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    #ui.setWindowFlags(ui.windowFlags() | QtCore.Qt.CustomizeWindowHint)
    #ui.setWindowFlags(QtCore.Qt.WindowCloseButtonHint, False)
    MainWindow.show()
    sys.exit(app.exec_())
