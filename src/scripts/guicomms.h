#ifndef GUICOMMS_H
#define GUICOMMS_H
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>

class GuiComms{
private:
public:
    GuiComms();
    ~GuiComms();

    void init();
    void write2file(std::string filename, std::string text);
    std::string readFile(std::string filename);
    void setBusyFlag();
    void clrBusyFlag();
    void setTaskCompleteFlag();
    std::string checkState();
    std::string checkComms();
    void exitSequence();

private:
    std::string _guiData;
    std::string _commsFile = "./GUI/guiComms.txt";
    std::string _stateFile = "./GUI/guiState.txt";
    const char* _commsFileChar = "./GUI/guiComms.txt";
    const char* _stateFileChar = "./GUI/guiState.txt";
};

#endif // GUICOMMS_H
