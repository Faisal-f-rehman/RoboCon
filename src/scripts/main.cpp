#include "vision.h"
#include <thread>
#include "robosocket.hpp"

int main(){

    // std::thread visionThread(&Vision::visionWorker,vision);
    // std::thread socketReadDataThread(&RoboSocket::socketReadDataWorker,std::ref(rSock));
    // std::thread socketSendDataThread(&RoboSocket::socketSendDataWorker,std::ref(rSock)); 
    
    std::thread visionThread(&Vision::visionWorker,vision);
    std::thread socketReadDataThread(&RoboSocket::socketReadDataWorker,std::ref(rSock));
    std::thread socketSendDataThread(&RoboSocket::socketSendDataWorker,std::ref(rSock)); 
    std::thread socketSendBackScreenThread(&RoboSocket::socketSendBackScreenWorker,std::ref(rSock));

    visionThread.join();
    socketReadDataThread.join();
    socketSendDataThread.join();
    socketSendBackScreenThread.join();
    return 0;
}