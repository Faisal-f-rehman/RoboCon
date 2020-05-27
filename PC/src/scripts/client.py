import socket
import time
import numpy as np
#import threading
import concurrent.futures
import queue

PORT_NO = 1235
HEADER_SIZE = 10
ROWS = 6
COLS = 7



def RoboPyClient(visionBoardQue,instructionQue):
    time.sleep(2)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET = IPV4 and SOCK_STREAM = TCP
    s.connect((socket.gethostname(),PORT_NO))

    board = np.zeros(42,int)
    move = 0
    first_move = 0

    exitFlag = False
    new_msg = True
    full_msg = ""
    while (not exitFlag):
        msg = s.recv(11)
        if (new_msg):
            print("client msg = {}".format(msg))
            msglen = int(msg[:HEADER_SIZE])
            msglenstr = len(str(abs(msglen)))
            new_msg = False
            # print("msglen = {}".format(msglen))
            
        full_msg += msg.decode("utf-8")
        
        if (len(full_msg) >= msglen + HEADER_SIZE + msglenstr):
            # print("Full message received : {}".format(full_msg))
            # print("Relevant message : {}".format(full_msg[HEADER_SIZE+msglenstr:]))
            
            board_str = full_msg[HEADER_SIZE+msglenstr:]
            board_str = board_str.replace(" ","")

            board_i_ = 0
            # breakFlag = False
    
            try:           
                board_i_ = 0
                # print("board index : {},  length of board string : {}".format(board_i_, len(board)))
                while (board_i_ < (len(board))):
                    board[board_i_] = int(board_str[board_i_])         
                    board_i_+=1
                # print("Current Board = {}".format(board)) 
                visionBoardQue.put(board)
                instructionQue.put(0)

                              

            except:
                print("Instruction recieved : {}".format(full_msg[HEADER_SIZE+msglenstr:]))
                if ("quit" in full_msg[HEADER_SIZE+msglenstr:]):
                    print("SHUTTING CLIENT DOWN")
                    visionBoardQue.put(9)
                    instructionQue.put(9)
                    exitFlag = True
                    new_msg = False
                    #############################################
                    msg = "received"                           ##
                    s.send(bytes(msg,"utf-8"))                 ##
                    msg = ""                                   ##
                    #############################################
                    s.close()
                    return None

                elif ("newgame" in full_msg[HEADER_SIZE+msglenstr:]):
                    board = np.zeros(42)
                    move = 0
                    first_move = 0
                    # print("New Game = {}".format(board))
                    print("============>>> NEW GAME <<<============")
                    instructionQue.put(1)
                    

                elif ("mainMenu" in full_msg[HEADER_SIZE+msglenstr:]):
                    board = np.zeros(42)
                    move = 0
                    visionBoardQue.put(9)
                    instructionQue.put(2)

            full_msg = ""
            new_msg = True
            
            #############################################
            msg = "received"                           ##
            s.send(bytes(msg,"utf-8"))                 ##
            msg = ""                                   ##
            #############################################

#########################################################################






# if (first_move == 1):
                #     print("first move RoBoCon")
                #     RoboMove = roboActionQue.get()
                #     slots_in_action_col = [(RoboMove - 1 + (COLS * r)) for r in range(ROWS)]
                #     for r in slots_in_action_col:
                #         if (board[r] == 0):
                #             board[r] = 1

                # board_i_ = 0
                # while (board_i_ < (len(board_str)) and (not breakFlag)):
                #     if ((board[board_i_] == 0) and (int(board_str[board_i_]) == 2)):
                #         board[board_i_] = int(board_str[board_i_]) 
                #         move = int(board_i_ % COLS)
                #         visionBoardQue.put(move+1)
                #         breakFlag = True
                #     board_i_+=1