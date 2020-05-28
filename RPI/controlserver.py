###############################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN      ##
###############################################
# Creates a server and expects connections    #
# from two clients on the PC. One for joint   #
# angles and the second one for the game      #
# fixture servo.                              #
############################################### 

import socket
import queue
import time

HEADER_SIZE = 10        # empty space between message size and the message
SHUTDOWN_CODE = 147258  # This number is used through out the RoboCon software which tells each script to shutdown

class ControlServer:
    #Constructor
    def __init__(self,port_num = 1236):
        self.port = port_num        # store port number in member variable

    #Destructor
    def __del__(self):
        try:
            self.rpiSocket.close()  # close the server before destroying the class
            print("SERVER SHUTDOWN SUCCESSFUL")
        except: 
            print("SERVER CLOSED")


    # Server thread routine
    def routine(self, actionQue, instructionQue):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Create a socket, AF_INET = IPV4 and SOCK_STREAM = TCP
        s.bind(('',self.port))                                # Bind the socket with open hostname and given port number
        s.listen(5)                                           # Start listening for connections (blocking)
        self.rpiSocket, address = s.accept()                  # Establish connection with the client
        print("connection from {} has been established".format(address))

        done = False    # Used as exit condition
        new_msg = True  # Used to indicate if the message in the buffer is a new message 
        full_msg = ""   # Used for storing the full message 

        while (not done):
            s.settimeout(5)                 # Sets timout on any subsequent socket operations (used for the recv function)
            msg = self.rpiSocket.recv(16)   # Wait for a message from client(s) with buffer size of 16 bytes
            
            connectionEstablished = False
            while (not msg):                                # Enter if connection was dropped and keep in loop until a connection is made and message is received
                s.settimeout(1)                             # Sets timout on any subsequent socket operations (used for the recv function)
                try:
                    self.rpiSocket, address = s.accept()    # Try to establish connection with the client
                    msg = self.rpiSocket.recv(16)           # Wait for message, backed by timeout 
                    connectionEstablished = True            # If this point is reached then set flag to indicate that the connection was established 
                except:
                    print("RPI CLIENT DISCONNECTED, NOW WAITING FOR CONNECTION...")

            if (connectionEstablished):
                print("connection from {} has been established".format(address))

            if (new_msg):                       # enter if its a new message
                msglen = int(msg[:HEADER_SIZE]) # extract message size, message format: message size 
                                                # followed by HEADER_SIZE spaces then the message, example = 2          Hi
                new_msg = False                 # Indicates the buffer currently has an old message 
                
            full_msg += msg.decode("utf-8")     # convert message encoding
            
            #___DEBUG____________________________________________________________________________________________________________________________________
            # print(f"full_msg = {full_msg}, len(full_msg) = {len(full_msg)}, msglen = {msglen}, HEADER_SIZE = {HEADER_SIZE}, lenMsgLen = {lenMsgLen}")##
            #############################################################################################################################################
            
            if (len(full_msg) >= msglen + HEADER_SIZE):     # Enter if message full message received (length received has reached expected length)
                try:
                    action = int(full_msg[HEADER_SIZE:])    # Extract the actual message
                    print("RoboCon Moving to column : {}".format(action))
                except:
                    print("Warning: Could not convert message from controlserver into action")
                    print("Move received : {}".format(action))
                
                if (action == SHUTDOWN_CODE):      # enter is message received is the shutdown code
                    done = True                    # set True to exit loop
                    actionQue.put(SHUTDOWN_CODE)   # indicate roboControl thread to shutdown
                else:
                    actionQue.put(action)          # send action required to roboControl
                
                full_msg = ""                      # clear variable
                
                instructionQue.get()               # wait for roboControl to indicate when it has completed the task required
                
                # Tell the client that message was received indicating # 
                # that the task was completed                          #
                ########################################################
                replyMsg = "Message Received"                         ##
                replyMsg = f"{len(replyMsg):<{HEADER_SIZE}}"+replyMsg ##
                self.rpiSocket.send(bytes(replyMsg,"utf-8"))          ##
                ########################################################

            #--------------------------------------------------------------
        time.sleep(2)          
        self.rpiSocket.close()  # close the socket down
#---------------------------------------------------------------------------------------------------------------------------------------------------

####################
#___Test script:
# cS = ControlServer()
# cS.routine()
# del cS