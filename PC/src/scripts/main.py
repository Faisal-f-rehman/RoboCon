import client
import connect4dqn as dqn
import rpiClient
import threading
import queue

SHUTDOWN_RPI_CODE = 147258

visionBoardQue = queue.Queue()
instructionQue = queue.Queue()
data2rpiQue = queue.Queue()
rpiTaskCompleteQue = queue.Queue()


rpiC = rpiClient.RpiClient(1236,"ffr") #faisal-Inspiron

clientThread = threading.Thread(target=client.RoboPyClient, args=(visionBoardQue,instructionQue))
dqnThread = threading.Thread(target=dqn.connect4player,args=(visionBoardQue,instructionQue,data2rpiQue,rpiTaskCompleteQue))
rpiClientThread = threading.Thread(target=rpiC.routine, args=(data2rpiQue,rpiTaskCompleteQue)) 

clientThread.start()
dqnThread.start()
rpiClientThread.start()

clientThread.join()
#___SHUTDOWN RPI SOCKETS___________##
data2rpiQue.put(SHUTDOWN_RPI_CODE) ##
rpiTaskCompleteQue.get()           ##
#####################################
# rpiClientThread.join()
# dqn.join()