//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
//                AUTHOR : FAISAL FAZAL-UR-REHMAN               //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
// THIS WAS NOT DEVELOPED AS PART OF THIS PROJECT.              //
//                                                              //
// Please refer to the RoboCon (Vision) repository link below:  //
// https://github.com/Faisal-f-rehman/10538828_RoboConVision    //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
//                              PC                              //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//

#include "guicomms.h"

GuiComms::GuiComms(){}

GuiComms::~GuiComms(){}


void GuiComms::init(){
    std::string flag = readFile("guiComms.txt");
    while (flag == "busy"){
        flag = readFile(_commsFile);
    }

    write2file(_commsFile,"free");
    write2file(_stateFile,"complete");
}


void GuiComms::write2file(std::string filename, std::string text){
    std::ofstream myfile(filename);
    myfile << text;
    myfile.close();
}


std::string GuiComms::readFile(std::string filename){
    std::string text;
    std::ifstream myfile(filename);
    std::getline(myfile,text);
    myfile.close();
    return text;
}


void GuiComms::setBusyFlag(){
    write2file(_commsFile,"busy");
}


void GuiComms::clrBusyFlag(){
    write2file(_commsFile,"free");
}


void GuiComms::setTaskCompleteFlag(){
    write2file(_stateFile,"complete");
}


std::string GuiComms::checkState(){
    std::string state = readFile(_stateFile);
    return state;
}


std::string GuiComms::checkComms(){
    std::string comms = readFile(_commsFile);
    return comms;
}

void GuiComms::exitSequence(){
    int result = remove( _commsFileChar );
    if( result == 0 ){
        printf( "Exit sequence 1 of 2 successful.\n" );
    } else {
        printf( "Could not delete guiComms.txt, file not found\n\r" );
    }

    result = remove( _stateFileChar );
    if( result == 0 ){
        printf( "Exit sequence 2 of 2 successful.\n" );
    } else {
        printf( "Could not delete guiState.txt, file not found\n\r" );
    }
}
