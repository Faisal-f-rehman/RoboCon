#include "robosocket.hpp"
#include <chrono>

#define HEADER_SIZE 10
#define PORT_NO 1235
#define BACK_SCREEN_SERVER_PORT "1237"
#define BACK_SCREEN_SERVER_HOST "ffr"

RoboSocket rSock(PORT_NO);

//___Constructor
RoboSocket::RoboSocket(const uint16_t portNumber){
    _portNumber = portNumber;
}


//___Destructor
RoboSocket::~RoboSocket(){}


//___
std::string RoboSocket::_read(boost::asio::ip::tcp::socket& socket) {
    // boost::asio::streambuf buf;
    // boost::asio::read_until( socket, buf, "\n" );
    // std::string data = boost::asio::buffer_cast<const char*>(buf.data());
    boost::system::error_code error;
    boost::asio::streambuf receive_buffer;
    
    // boost::asio::read(socket, receive_buffer, boost::asio::transfer_all(), error);
    boost::asio::read(socket, receive_buffer, boost::asio::transfer_at_least(1), error);
    
    if( error && error != boost::asio::error::eof ) {
        std::cout << "receive failed: " << error.message() << std::endl;
    }
    else {
        const char* data = boost::asio::buffer_cast<const char*>(receive_buffer.data());
        std::cout << data << std::endl;
        return data;
    }
    
    return "";
}


//___
void RoboSocket::_send(boost::asio::ip::tcp::socket& socket, const std::string& message) {
    const std::string msg = message + "\n";
    // std::cout << "Socket send : "<< msg << std::endl;
    boost::asio::write( socket, boost::asio::buffer(msg) );
}


//___
std::string RoboSocket::_attachHeader(std::string msgIn){
    std::string msg = "";
    std::stringstream ss;
    int msg_len = (msgIn.length());

    ss << msg_len;    
    ss >> msg; 

    for (uint8_t i = 0; i < HEADER_SIZE;i++){
        msg += " ";
    }

    msg += msgIn;

    return msg;
}


//___
void RoboSocket::_sendDataToPy(boost::asio::ip::tcp::socket& socket, std::string msg){
    
    
    msg = _attachHeader(msg);
    
    //write operation
    _send(socket, msg);
    // std::cout << "Servent sent Hello message to Client!" << std::endl;

}

//___
std::string RoboSocket::_receiveDataFromPy(boost::asio::ip::tcp::socket& socket){    

    std::string msg = _read(socket);
    // std::cout << msg << std::endl;
    return msg;
}

//___
void RoboSocket::socketReadDataWorker(){
    
    boost::asio::io_service io_service;
    
    //listen for new connection
    boost::asio::ip::tcp::acceptor acceptor_(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), _portNumber - 1));
    
    //socket creation 
    boost::asio::ip::tcp::socket socket_(io_service);
    
    //waiting for connection
    acceptor_.accept(socket_);

    // std::cout << "I was in socketWorket()" << std::endl;

       
    
    while (true){
               
        std::this_thread::sleep_for(std::chrono::seconds(10));
        // _receiveDataFromPy(socket_);
        if(exitSocketReadDataWorkerFlag){
            break;
        }
    }

    // std::terminate();
    
}


void RoboSocket::socketSendDataWorker(){//(boost::asio::ip::tcp::socket& socket_){
    boost::asio::io_service io_service;
    
    //listen for new connection
    boost::asio::ip::tcp::acceptor acceptor_(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), _portNumber));
    
    //socket creation 
    boost::asio::ip::tcp::socket socket_(io_service);
    
    //waiting for connection
    acceptor_.accept(socket_);
    
    while(!getVisionInitStatus()){}

    init_conditionalVar.notify_all();
    while (true){
        // std::cout << "I was in _sendWorker()" << std::endl;
        std::unique_lock<std::mutex> locker(_sockLock);
        // std::cout << "I was in _sendWorker() after lock" << std::endl;
        
        sendData_conditionalVar.wait(locker);
        
        if(exitSocketSendDataWorkerFlag){
            break;
        }
        
        _sendDataToPy(socket_,_data2send);
        locker.unlock();
        
        if ((_receiveDataFromPy(socket_) == "received") && (!exitSocketSendDataWorkerFlag)){
            dataSent_conditionalVar.notify_all();
        }
        else{
            std::perror("Error: in RoboSocket::sendDataThreadWorker() meesage to client not received");
        }
        
        
    }
    // std::terminate();
}


void RoboSocket::socketSendBackScreenWorker(){    
      
    boost::asio::io_service io_backscreen_service;
   
    boost::asio::ip::tcp::resolver resolver(io_backscreen_service);
    

    std::string hostname = BACK_SCREEN_SERVER_HOST;
    std::string portNum = BACK_SCREEN_SERVER_PORT;
    // boost::asio::ip::tcp::resolver::query query(hostname, portNum);
    // boost::asio::ip::tcp::resolver::query query(boost::asio::ip::tcp::v4(), hostname, portNum);
    // boost::asio::ip::tcp::resolver::query query(hostname, portNum,boost::asio::ip::resolver_query_base::flags());
    boost::asio::ip::tcp::resolver::query query(hostname, portNum, boost::asio::ip::resolver_query_base::all_matching);
    
    boost::asio::ip::tcp::resolver::iterator iter = resolver.resolve(query);

    boost::asio::ip::tcp::resolver::iterator end; // End marker.
    while (iter != end){
        boost::asio::ip::tcp::endpoint endpoint = *iter++;
        // std::cout << "endpoint: " << endpoint << std::endl;
    }

    std::cout << "I WAS HERE" << std::endl;

    boost::asio::ip::tcp::socket _socket_(io_backscreen_service);
    bool connected_to_server = false;
    while (!connected_to_server){
        try{
            boost::asio::connect(_socket_, resolver.resolve(query));
            connected_to_server = true;
            std::cout << "connection with " << hostname << "@" << portNum << " established\n";
        }
        catch (...){
            connected_to_server = false;
            std::cout << "trying to connect with : " << hostname << "@" << portNum << "\n";
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    // boost::system::error_code lErrorCode;
    // _socket_.close(lErrorCode);
    // std::cout <<  lErrorCode.message() << std::endl;
    //listen for new connection
    // boost::asio::ip::tcp::acceptor acceptor_(io_service, backScreenEndpoint);
    // boost::asio::ip::tcp::acceptor acceptor_(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), BACK_SCREEN_SERVER_PORT));
    
    while (true){
        std::cout << "BEGINNING socketSendBackScreenWorker()" << std::endl;
        std::unique_lock<std::mutex> locker(_sockLock);
        
        sendBackScreenData_conditionalVar.wait(locker);
        
        if(exitSocketSendDataWorkerFlag){
            break;
        }
        
        _sendDataToPy(_socket_,_data2send);
        locker.unlock();
        
        _receiveDataFromPy(_socket_);
        if (!exitSocketSendDataWorkerFlag){
            backScreenDataSent_conditionalVar.notify_all();
        }
        std::cout << "END socketSendBackScreenWorker()" << std::endl;      
    }
    // std::terminate();
}


bool RoboSocket::getVisionInitStatus(){
    std::lock_guard<std::mutex> guard(_sockLock);
    return _visionInitialised;
}


void RoboSocket::setVisionInitStatus(bool status){
    std::lock_guard<std::mutex> guard(_sockLock);
    _visionInitialised = status;
}

void RoboSocket::sendData(std::vector<uint8_t> data2send){
    std::lock_guard<std::mutex> guard(_sockLock);
    _data2send = _vec2str<uint8_t>(data2send);
    
    // std::cout << "data set to send in sendData() --> " << _data2send << std::endl;
    // std::cout << "data in sendData() --> ";
    // for (uint8_t i = 0; i < data2send.size(); i++){
    //     std::cout << unsigned(data2send[i]);
    //  }
    // std::cout << std::endl;
}



void RoboSocket::sendString(std::string& str2send){
    std::lock_guard<std::mutex> guard(_sockLock);
    _data2send = str2send;
}


//___
template<typename nType>
std::string RoboSocket::_num2str(const nType num){  
    std::stringstream strS;  
    std::string str;  
    strS<<num;  
    strS>>str;
    return str;  
}

//___
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


//___
template<typename vType>
std::string RoboSocket::_vec2str(const std::vector<vType> v){
    std::stringstream vStream;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(vStream, " "));
    std::string vStr = vStream.str();
    vStr = vStr.substr(0, vStr.length()-1);    
    return vStr; 
}



