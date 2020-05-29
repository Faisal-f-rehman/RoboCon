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

#include "robosocket.hpp"
#include <chrono>

#define HEADER_SIZE 10                  // empty space between message size and the message
#define PORT_NO 1235                    // port number  (PC <--> PC)  (C++ <--> python)
#define BACK_SCREEN_SERVER_PORT "1237"  // port number  (PC <--> RPI) (C++ <--> python)
#define BACK_SCREEN_SERVER_HOST "ffr"   // hostname RPI (PC <--> RPI) (C++ <--> python)  

RoboSocket rSock(PORT_NO);              // Create global instance of the RoboSocket class to use in the thread routine at the end of this file

//___Constructor___________________________________
RoboSocket::RoboSocket(const uint16_t portNumber){
    _portNumber = portNumber;
}//###################################################################################################//



//___Destructor____________
RoboSocket::~RoboSocket(){}
//###################################################################################################//



//___Reads message on the provided socket
std::string RoboSocket::_read(boost::asio::ip::tcp::socket& socket) {

    //___local variables______________________
    boost::system::error_code error;        //
    boost::asio::streambuf receive_buffer;  //
    //--------------------------------------//

    boost::asio::read(socket, receive_buffer, boost::asio::transfer_at_least(1), error); // read message and store in buffer
    
    if( error && error != boost::asio::error::eof ) {                                    // enter if error occured
        std::cout << "receive failed: " << error.message() << std::endl;                 // print error
    }
    else {                                                                               // enter if message was received successfully
        const char* data = boost::asio::buffer_cast<const char*>(receive_buffer.data()); // convert buffer data to char array
        // std::cout << data << std::endl;                                               // for debugging                                
        return data;                                                                     // return message received
    }
    
    return "";  // if this point is reached then error has occured
}//###################################################################################################//



//___Sends provided message on the provided socket
void RoboSocket::_send(boost::asio::ip::tcp::socket& socket, const std::string& message) {
    const std::string msg = message + "\n";                 // concatinate carriage return at the end of message
    boost::asio::write( socket, boost::asio::buffer(msg) ); // send message on the socket
}//###################################################################################################//



//___Concatinates message size and header space before message  
std::string RoboSocket::_attachHeader(std::string msgIn){
    //___local variables_____
    std::string msg = "";  //         
    std::stringstream ss;  //
    //---------------------//

    int msg_len = (msgIn.length());             // extract message length
    ss << msg_len;                              // convert int to stringstream
    ss >> msg;                                  // convert stringstream to string

    //__create a string of empty spaces of size HEADER_SIZE___
    for (uint8_t i = 0; i < HEADER_SIZE;i++){               //
        msg += " ";                                         //
    }//-----------------------------------------------------//

    msg += msgIn;                               // concatinate actual message after meassage size and header spaces
    return msg;                                 // return final string 
}//###################################################################################################//



//___Used by client that communicated with the RPI to send messages
void RoboSocket::_sendDataToPy(boost::asio::ip::tcp::socket& socket, std::string msg){
    msg = _attachHeader(msg);   // concatinate header (size and space)
    _send(socket, msg);         // write operation
}//###################################################################################################//



//___Used by client that communicates with RPI to receive messages
std::string RoboSocket::_receiveDataFromPy(boost::asio::ip::tcp::socket& socket){
    std::string msg = _read(socket);    // read message
    return msg;                         // return message
}//###################################################################################################//



//___Thread routine for server to read data from python side of the software (currently not in use)
void RoboSocket::socketReadDataWorker(){
    
    boost::asio::io_service io_service;    
    boost::asio::ip::tcp::acceptor acceptor_(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), _portNumber - 1)); // listen for new connection
    boost::asio::ip::tcp::socket socket_(io_service);   // socket creation     
    acceptor_.accept(socket_);                          // wait for connection
    
    while (true){
        std::this_thread::sleep_for(std::chrono::seconds(10)); // put thread to sleep for 10 seconds
        if(exitSocketReadDataWorkerFlag){                      // enter if flag is set by the vision system thread
            break;                                             // exit thread
        }
    }    
}//###################################################################################################//



//___Thread routine for server to send data to python side of software 
void RoboSocket::socketSendDataWorker(){

    boost::asio::io_service io_service;
    boost::asio::ip::tcp::acceptor acceptor_(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), _portNumber));//listen for new connection    
    boost::asio::ip::tcp::socket socket_(io_service);   // socket creation 
    acceptor_.accept(socket_);                          // wait for connection
    while(!getVisionInitStatus()){}                     // block and wait, released by vision system thread, used during initiation for syncing
    init_conditionalVar.notify_all();                   // notify vision system thread, used during initiation
    
    while (true){
        std::unique_lock<std::mutex> locker(_sockLock); // acquire mutex lock 
        sendData_conditionalVar.wait(locker);           // puts thread to sleep and releases mutex lock while asleep,
                                                        // awoken by vision thread it grabs the lock back before continuing 
        if(exitSocketSendDataWorkerFlag){               // enter if flag is set by the vision system thread, this flag indicates shutdown
            break;                                      // exit thread
        }
        
        _sendDataToPy(socket_,_data2send);              // send data to client
        locker.unlock();                                // release lock
        
        if ((_receiveDataFromPy(socket_) == "received") && (!exitSocketSendDataWorkerFlag)){  // wait for a reply from client
            dataSent_conditionalVar.notify_all();                                             // notify vision thread that task has been completed
        }
        else{
            std::perror("Error: in RoboSocket::sendDataThreadWorker() meesage to client not received");
        }
    }
}//###################################################################################################//



//___Thread routine for client that sends data to RPI server for game fixture motor control
void RoboSocket::socketSendBackScreenWorker(){    
      
    boost::asio::io_service io_backscreen_service;

    boost::asio::ip::tcp::resolver resolver(io_backscreen_service); // provides the ability to resolve query to a list of endpoints

    std::string hostname = BACK_SCREEN_SERVER_HOST; // hostname
    std::string portNum = BACK_SCREEN_SERVER_PORT;  // port number

    while (true){
        try{
            //__find endpoints for given hostname and portnumber___________________________________________________________________
            boost::asio::ip::tcp::resolver::query query(hostname, portNum, boost::asio::ip::resolver_query_base::all_matching);  //
            boost::asio::ip::tcp::resolver::iterator iter = resolver.resolve(query);                                             //
            boost::asio::ip::tcp::resolver::iterator end; // End marker                                                          //
            //-------------------------------------------------------------------------------------------------------------------//

            //__iterate through all endpoints____________________________
            while (iter != end){                                       //
                boost::asio::ip::tcp::endpoint endpoint = *iter++;     //
            }//--------------------------------------------------------//

            boost::asio::ip::tcp::socket _socket_(io_backscreen_service);     // create socket
            
            bool connected_to_server = false;                                 // exit condition flag boolean variable
            while (!connected_to_server){                                     // loops until connection is established (used for init and dropped connections)
                try{
                    boost::asio::connect(_socket_, resolver.resolve(query));  // try connecting to server 
                    connected_to_server = true;                               // if this point is reached connection would have been established 
                    std::cout << "connection with " << hostname << "@" << portNum << " established\n";
                }
                catch (...){                                                  // enter if exception was thrown i.e. connection failed
                    connected_to_server = false;                              // exit condition unsatisfied, if connection was unsuccessful 
                    std::cout << "trying to connect with : " << hostname << "@" << portNum << "\n";
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));         // don't loop faster than 1 second per loop
            }
            
            while (true){

                std::unique_lock<std::mutex> locker(_sockLock); // acquire mutex lock
                
                sendBackScreenData_conditionalVar.wait(locker); // puts thread to sleep and releases mutex lock while asleep,
                                                                // its awoken by vision thread, it grabs the lock back before continuing 
                
                if(exitSocketSendDataWorkerFlag){               // enter if exit flag is set by vision thread
                    break;                                      // exit thread
                }
                
                _sendDataToPy(_socket_,_data2send);             // send data to server
                locker.unlock();                                // release lock
                
                _receiveDataFromPy(_socket_);                   // block and wait until server has replied
                if (!exitSocketSendDataWorkerFlag){             // enter if exit flag is NOT set
                    backScreenDataSent_conditionalVar.notify_all(); // indicate vision thread that game fixture task has been completed
                }      
            }
        }
        catch (...){
            if(exitSocketSendDataWorkerFlag){ // enter if and exit flag is set (set by vision thread)
                break;                        // exit thread
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));  // normally reached when RPI server is off, in that its
                                                                    // better to sleep for longer before checking for it again
            std::cout << "\n\n "<< hostname << "@" << portNum << "not found, is server on RPI running?\n\n";
        }
        if(exitSocketSendDataWorkerFlag){   // enter if and exit flag is set (set by vision thread)
            break;                          // exit thread
        }
    }
}

//___Checks if vision system has been initiated and returns true if it has
bool RoboSocket::getVisionInitStatus(){
    std::lock_guard<std::mutex> guard(_sockLock);
    return _visionInitialised;
}


//___Sets vision system's initialization status
void RoboSocket::setVisionInitStatus(bool status){
    std::lock_guard<std::mutex> guard(_sockLock);
    _visionInitialised = status;
}


//___stores 8-bit unsigned integer vector as std::string in member variable to send through one of the sockets
void RoboSocket::sendData(std::vector<uint8_t> data2send){
    std::lock_guard<std::mutex> guard(_sockLock);
    _data2send = _vec2str<uint8_t>(data2send);
}


//____stores string message in member variable to send through one of the sockets
void RoboSocket::sendString(std::string& str2send){
    std::lock_guard<std::mutex> guard(_sockLock);
    _data2send = str2send;
}


//___Member Template function to convert nType num to string
template<typename nType>
std::string RoboSocket::_num2str(const nType num){  
    std::stringstream strS;  
    std::string str;  
    strS<<num;  
    strS>>str;
    return str;  
}

//___Member Template function to convert string to vType vector
template<typename vType>
std::vector<vType> RoboSocket::_str2vec(std::string str){
    std::vector<vType> v;
    int number = 0;

    str.erase(str.length()-8);
    std::stringstream vStream(str);

    while ( vStream >> number ){
        v.push_back(number);
        // printf("---------->num %u\t---------->str %s\n", number, str.c_str());
    }  
    return v; 
}


//___Member Template function to convert vType vector to string
template<typename vType>
std::string RoboSocket::_vec2str(const std::vector<vType> v){
    std::stringstream vStream;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(vStream, " "));
    std::string vStr = vStream.str();
    vStr = vStr.substr(0, vStr.length()-1);    
    return vStr; 
}




