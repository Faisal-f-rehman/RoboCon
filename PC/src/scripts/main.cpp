//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
//       AUTHOR : FAISAL FAZAL-UR-REHMAN        //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
// Main, software entry point for all PC C++    //
// scripts. This script initiates threads and   //
// joins them and then waits for them to finish //  
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
//                      PC                      //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//

#include "vision.h"
#include <thread>
#include "robosocket.hpp"

int main(){
   
    //___Define and initiate threads___________________________________________________________________
    std::thread visionThread(&Vision::visionWorker,vision);                                          // Initiate vision thread with call to thread worker
    std::thread socketReadDataThread(&RoboSocket::socketReadDataWorker,std::ref(rSock));             // Initiate pc c++ <--> python pc  read socket thread with call to thread worker
    std::thread socketSendDataThread(&RoboSocket::socketSendDataWorker,std::ref(rSock));             // Initiate pc c++ <--> python pc  send socket thread with call to thread worker
    std::thread socketSendBackScreenThread(&RoboSocket::socketSendBackScreenWorker,std::ref(rSock)); // Initiate pc c++ <--> python rpi read socket  thread with call to thread worker
    //-----------------------------------------------------------------------------------------------//

    //___Join threads and wait for them to finish____
    visionThread.join();                           //
    socketReadDataThread.join();                   //
    socketSendDataThread.join();                   //
    socketSendBackScreenThread.join();             //
    //---------------------------------------------//
    
    return 0;
}