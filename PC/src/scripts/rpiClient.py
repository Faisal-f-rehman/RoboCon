import socket

HEADER_SIZE = 10
SHUTDOWN_CODE = 147258

class RpiClient:
    def __init__(self, port_num=1236, host_name='pi'):
        self.hostName = host_name
        self.port = port_num 

    def __del__(self):
        print("SHUTTING RPI CLIENT AND SERVER DOWN")
        new_msg = True
        full_msg = ""
        
        msg = SHUTDOWN_CODE
        msg = str(msg)
        msg = f"{len(msg):<{HEADER_SIZE}}"+msg
        
        self.s.send(bytes(msg,"utf-8"))
        
        #-----------------

        msgReceived = False
        while (not msgReceived):
            msgRcv = self.s.recv(16)
            
            if (new_msg):
                msglen = int(msgRcv[:HEADER_SIZE])
                msglenstr = len(str(abs(msglen)))
                new_msg = False
                # print("msglen = {}".format(msglen))
            
            full_msg += msgRcv.decode("utf-8")

            if (len(full_msg) >= msglen + HEADER_SIZE):
                print("Message from RPI : {}".format(full_msg))
                # print("Relevant message : {}".format(full_msg[HEADER_SIZE+msglenstr:]))
                msgReceived = True
                full_msg = ""
                new_msg = True

    def routine(self, data2rpiQue,rpiTaskCompleteQue):
        controlServerFound = False
        while (not controlServerFound):
            try:
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET = IPV4 and SOCK_STREAM = TCP
                self.s.connect((self.hostName,self.port))
                controlServerFound = True
            except:
                controlServerFound = False

        new_msg = True
        full_msg = ""
        done = False
        while (not done):
            numFlag = False
            while(not numFlag):
                try:
                    # msg = int(input("Enter num to send:"))
                    msg = data2rpiQue.get()
                    print(f"==============>>>>msg to send to RPI : {msg}")
                    if ((msg <= 7) and (msg >= 1) or (msg == SHUTDOWN_CODE)):
                        numFlag = True
                    elif(msg != 0):
                        raise Exception("Error: in rpiClient, RoboCon move is invalid, must be between 1 to 7")
                except:
                    numFlag = False
                    raise Exception("Error: in rpiClient, RoboCon move is invalid, must be between 1 to 7 or equal to SHUTDOWN_CODE")

            # msg = data2rpiQue.get()
            # print('{: <{}} {}'.format(HEADER_SIZE,len(msg), msg))
            
            msg = str(msg)
            msg = f"{len(msg):<{HEADER_SIZE}}"+msg
            
            self.s.send(bytes(msg,"utf-8"))

            if (msg == SHUTDOWN_CODE):
                done = True
            
            #-----------------

            msgReceived = False
            while (not msgReceived):
                msgRcv = self.s.recv(16)
                
                if (new_msg):
                    msglen = int(msgRcv[:HEADER_SIZE])
                    msglenstr = len(str(abs(msglen)))
                    new_msg = False
                    # print("msglen = {}".format(msglen))
                
                full_msg += msgRcv.decode("utf-8")

                if (len(full_msg) >= msglen + HEADER_SIZE):
                    print("Message from RPI : {}".format(full_msg))
                    # print("Relevant message : {}".format(full_msg[HEADER_SIZE+msglenstr:]))
                    msgReceived = True
                    full_msg = ""
                    new_msg = True
                    rpiTaskCompleteQue.put(1)
                    if (done):
                        break
            if (done):
                break
