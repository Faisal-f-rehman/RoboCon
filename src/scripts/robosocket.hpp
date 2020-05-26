#ifndef __ROBOSOCKET__HPP
#define __ROBOSOCKET__HPP

#include <iostream>
#include <boost/asio.hpp>
#include <thread>
#include <condition_variable>
#include <mutex>

class RoboSocket{
public:
    
    RoboSocket(const uint16_t portNumber);
    ~RoboSocket();
    void socketReadDataWorker();
    void socketSendDataWorker();
    void socketSendBackScreenWorker();
    void sendData(std::vector<uint8_t> data2send);
    void sendString(std::string& str2send);
    std::vector<uint8_t> readData();
    bool getVisionInitStatus();
    void setVisionInitStatus(bool status);
    bool exitSocketReadDataWorkerFlag = false;
    bool exitSocketSendDataWorkerFlag = false;

    std::mutex rSockLock;
    std::condition_variable init_conditionalVar;
    std::condition_variable sendData_conditionalVar;
    std::condition_variable dataSent_conditionalVar;
    std::condition_variable readData_conditionalVar;
    std::condition_variable sendBackScreenData_conditionalVar;
    std::condition_variable backScreenDataSent_conditionalVar;
     
private:
    std::mutex _sockLock;

    // boost::asio::io_service io_service;  


    //Private Methods
    std::string _read(boost::asio::ip::tcp::socket & socket);
    void _send(boost::asio::ip::tcp::socket & socket, const std::string& message);
    std::string _attachHeader(std::string msgIn);
    void _sendDataToPy(boost::asio::ip::tcp::socket& socket, std::string msg);
    std::string _receiveDataFromPy(boost::asio::ip::tcp::socket& socket);

    //___Templates
    template<typename nType>
    std::string _num2str(const nType num);
    template<typename vType>
    std::vector<vType> _str2vec(std::string str);
    template<typename vType>
    std::string _vec2str(const std::vector<vType> v);

    //Private Member Variables
    uint16_t _portNumber = 0;
    std::string _data2send = "";
    bool _visionInitialised = false;

};


extern RoboSocket rSock;

#endif