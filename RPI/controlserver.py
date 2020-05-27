import socket
import queue
import time

HEADER_SIZE = 10
SHUTDOWN_CODE = 147258

class ControlServer:
    #Constructor
    def __init__(self,port_num = 1236):
        self.port = port_num

    #Destructor
    def __del__(self):
        try:
            self.rpiSocket.close()
            print("SERVER SHUTDOWN SUCCESSFUL")
        except: 
            print("SERVER CLOSED")


    def routine(self, actionQue, instructionQue):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET = IPV4 and SOCK_STREAM = TCP
        # s.bind((socket.gethostname(),self.port))
        s.bind(('',self.port))
        
        s.listen(5)
        
        self.rpiSocket, address = s.accept()
        print("connection from {} has been established".format(address))

        done = False
        new_msg = True
        full_msg = ""

        while (not done):
            s.settimeout(5)
            msg = self.rpiSocket.recv(16)
            
            connectionEstablished = False
            while (not msg):
                s.settimeout(1)
                try:
                    self.rpiSocket, address = s.accept()
                    msg = self.rpiSocket.recv(16)
                    connectionEstablished = True
                except:
                    print("RPI CLIENT DISCONNECTED, NOW WAITING FOR CONNECTION...")

            if (connectionEstablished):
                print("connection from {} has been established".format(address))

            if (new_msg):
                msglen = int(msg[:HEADER_SIZE])
                # lenMsgLen = len(str(abs(msglen)))
                new_msg = False
                # print("msglen = {}".format(msglen))
                
            full_msg += msg.decode("utf-8")
            
            # print(f"full_msg = {full_msg}, len(full_msg) = {len(full_msg)}, msglen = {msglen}, HEADER_SIZE = {HEADER_SIZE}, lenMsgLen = {lenMsgLen}")
            if (len(full_msg) >= msglen + HEADER_SIZE):
                try:
                    action = int(full_msg[HEADER_SIZE:])
                    print("RoboCon Moving to column : {}".format(action))
                except:
                    print("Warning: Could not convert message from controlserver into action")
                    print("Move received : {}".format(action))

                # print("Full message received : {}".format(full_msg))
                
                if (action == SHUTDOWN_CODE):
                    done = True
                    actionQue.put(SHUTDOWN_CODE)
                else:
                    actionQue.put(action)
                
                full_msg = ""
                
                instructionQue.get()
                ########################################################
                replyMsg = "Message Received"                         ##
                replyMsg = f"{len(replyMsg):<{HEADER_SIZE}}"+replyMsg ##
                self.rpiSocket.send(bytes(replyMsg,"utf-8"))          ##
                ########################################################

            #--------------------------------------------------------------
        time.sleep(2)
        self.rpiSocket.close()

####################
#___Test script:
# cS = ControlServer()
# cS.routine()
# del cS

'''
new_msg = True
full_msg = ""
while (True):
    msg = s.recv(16)
    if (new_msg):
        msglen = int(msg[:HEADER_SIZE])
        msglenstr = len(str(abs(msglen)))
        new_msg = False
        print("msglen = {}".format(msglen))
        
    full_msg += msg.decode("utf-8")
    
    if (len(full_msg) >= msglen + HEADER_SIZE + msglenstr):
        print("Full message received : {}".format(full_msg))
        print("Relevant message : {}".format(full_msg[HEADER_SIZE+msglenstr:]))
        print("I was here")

        if (full_msg[HEADER_SIZE+msglenstr:] == "quit"):
            print("SHUTTING CLIENT DOWN")
            break
        
        full_msg = ""
        new_msg = True

        print("and here")
        #############################################
        msg = "From Client --->> Message recieved" ##
        s.send(bytes(msg,"utf-8"))                 ##
        msg = ""                                   ##
        #############################################
        print("plus here")
'''