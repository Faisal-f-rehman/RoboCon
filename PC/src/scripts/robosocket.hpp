//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
//       AUTHOR : FAISAL FAZAL-UR-REHMAN        //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
// RoboSocket class provides server and client  //
// for communication with python scripts and    //
// for communicating with RPI. It also provides //
// a seperate thread routine for receiving      //
// data from a client. The sockets are written  //
// with Boost library.                          //
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//

#ifndef __ROBOSOCKET__HPP
#define __ROBOSOCKET__HPP

#include <iostream>
#include <boost/asio.hpp>
#include <thread>
#include <condition_variable>
#include <mutex>

class RoboSocket{
public:
    
    RoboSocket(const uint16_t portNumber);          // constructor
    ~RoboSocket();                                  // destructor
    void socketReadDataWorker();                    // thread routine for server to read data from python side of the software (currently not in use)
    void socketSendDataWorker();                    // thread routine for server to send data to python side of software 
    void socketSendBackScreenWorker();              // thread routine for client that sends data to RPI server for game fixture motor control
    void sendData(std::vector<uint8_t> data2send);  // stores 8-bit unsigned integer vector as std::string in member variable to send through one of the sockets
    void sendString(std::string& str2send);         // stores string message in member variable to send through one of the sockets
    bool getVisionInitStatus();                     // checks if vision system has been initiated and returns true if it has
    void setVisionInitStatus(bool status);          // sets vision system's initialization status
    bool exitSocketReadDataWorkerFlag = false;
    bool exitSocketSendDataWorkerFlag = false;

    std::mutex rSockLock;                                       // mutex lock
    std::condition_variable init_conditionalVar;                // used to sync vision and socket initiation
    std::condition_variable sendData_conditionalVar;            // used to wait for data from vision thread to send over sockets 
    std::condition_variable dataSent_conditionalVar;            // used to indicate vision thread that the data has been sent
    std::condition_variable readData_conditionalVar;            // used by vision thread to wait until data has been read by sockets thread
    std::condition_variable sendBackScreenData_conditionalVar;  // used to wait for game fixture data from vision to send over sockets
    std::condition_variable backScreenDataSent_conditionalVar;  // used to indicate vision thread that game fixture data has been sent 
    
    //std::vector<uint8_t> readData(); 

private:
    std::mutex _sockLock;

    //Private Methods
    
    std::string _read(boost::asio::ip::tcp::socket & socket);                      // Reads message on the provided socket
    void _send(boost::asio::ip::tcp::socket & socket, const std::string& message); // Sends provided message on the provided socket
    std::string _attachHeader(std::string msgIn);                                  // concatinates message size and header space before message
    void _sendDataToPy(boost::asio::ip::tcp::socket& socket, std::string msg); // used by client that communicated with the RPI to send messages
    std::string _receiveDataFromPy(boost::asio::ip::tcp::socket& socket);          // Used by client that communicates with RPI to receive messages
    
    //___Templates
    template<typename nType>                            // Member Template function 
    std::string _num2str(const nType num);              // to convert nType num to string
    
    template<typename vType>                            // Member Template function to convert
    std::vector<vType> _str2vec(std::string str);       // string to vType vector
    
    template<typename vType>                            // Member Template function to 
    std::string _vec2str(const std::vector<vType> v);   // convert vType vector to string

    //Private Member Variables
    uint16_t _portNumber = 0;
    std::string _data2send = "";
    bool _visionInitialised = false;

};


extern RoboSocket rSock; // Create global instance of the RoboSocket class to use in the thread routine

#endif