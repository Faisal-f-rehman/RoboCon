###############################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN      ##
###############################################
# Used for connecting the C++ side to the     #
# python side of the software. It Creates a   #
# client and connects to roboSocket server    #
# (c++ side of the code). Expects game board  #
# states and extracts action from the board   #
# state and sends it to the DQN.              #
###############################################
##                    PC                     ##
###############################################

import socket
import time
import numpy as np
import concurrent.futures
import queue

PORT_NO = 1235      # Port to connect to
HEADER_SIZE = 10    # empty space between message size and the message
ROWS = 6            # Defines the rows on connect 4 board
COLS = 7            # Defines the cols on connect 4 board

# Client thread routine
def RoboPyClient(visionBoardQue,instructionQue):
    time.sleep(2)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Create a socket, AF_INET = IPV4 and SOCK_STREAM = TCP
    s.connect((socket.gethostname(),PORT_NO))             # Connect to port, as the server is on the same PC can use same hostname as client
    board = np.zeros(42,int)                              # Define board as a numpy array of size 42

    exitFlag = False                                      # Defines exit condition for the programme's main loop
    new_msg = True                                        # Used to indicate if the message in the buffer is a new message   
    full_msg = ""                                         # Used for storing the full message
    while (not exitFlag):                                 # programme main loop
        msg = s.recv(11)                                  # block and wait for a message from the server with buffer size of 11 bytes
        if (new_msg):                                     # enter if a new message is received
            print("client msg = {}".format(msg))           
            msglen = int(msg[:HEADER_SIZE])               # extract message length
            msglenstr = len(str(abs(msglen)))             # convert message length (int) to message length (string) 
            new_msg = False                               # set flag to say the current message is not a new message any more
            
        full_msg += msg.decode("utf-8")                   # convert message encoding
        
        if (len(full_msg) >= msglen + HEADER_SIZE + msglenstr): # Enter if full message received (i.e. length received has reached expected length)
            
            #___DEBUG___________________________________________________________________
            # print("Full message received : {}".format(full_msg))                    ##
            # print("Relevant message : {}".format(full_msg[HEADER_SIZE+msglenstr:])) ##  
            ############################################################################

            board_str = full_msg[HEADER_SIZE+msglenstr:]        # extract the actual message (board state) 
            board_str = board_str.replace(" ","")               # remove spaces in the board state string
            board_i_ = 0                                        # reset board count

            try:           
                while (board_i_ < (len(board))):                # loop for the length of the board states (0 to 43 for connect 4)
                    board[board_i_] = int(board_str[board_i_])  # convert string to int      
                    board_i_+=1                                 # increment board count
                visionBoardQue.put(board)                       # send action extracted to DQN
                instructionQue.put(0)                           # send instructions like, game finished, shutdown etc, to DQN 

                              

            except:

                #___DEBUG_______________________________________________________________________
                # print("Instruction recieved : {}".format(full_msg[HEADER_SIZE+msglenstr:])) ##
                ################################################################################
                
                #___Enter if the message received was to quit
                if ("quit" in full_msg[HEADER_SIZE+msglenstr:]):        
                    print("SHUTTING CLIENT DOWN")   
                    visionBoardQue.put(9)                               # Unblock DQN if waiting for a turn during the game
                    instructionQue.put(9)                               # Tell DQN to shutdown / exit script
                    exitFlag = True                                     # set exit condition for this script
                    new_msg = False                                     # reset variable
                    
                    ###################################################
                    ## Indicate server that the message was received ##
                    ###################################################
                    msg = "received"                                 ##
                    s.send(bytes(msg,"utf-8"))                       ##
                    msg = ""                                         ##
                    ###################################################

                    s.close()                                           # close connection
                    return None                                         # exit thread routine function

                #___Enter if the message received was to start a new game
                elif ("newgame" in full_msg[HEADER_SIZE+msglenstr:]):   
                    board = np.zeros(42)                                # clear board variable
                    print("============>>> NEW GAME <<<============")
                    instructionQue.put(1)                               # tell DQN to start a new game
                    
                #___Enter if the message received was to exit game and go to main menu
                elif ("mainMenu" in full_msg[HEADER_SIZE+msglenstr:]):
                    board = np.zeros(42)                                # clear board variable
                    visionBoardQue.put(9)                               # unblock deque.get() in DQN
                    instructionQue.put(2)                               # tell DQN to move to main menu

            full_msg = ""   # clear variable                                    
            new_msg = True  # reset variable
            
            ###################################################
            ## Indicate server that the message was received ##
            ###################################################
            msg = "received"                                 ##
            s.send(bytes(msg,"utf-8"))                       ##
            msg = ""                                         ##
            ###################################################

#########################################################################
