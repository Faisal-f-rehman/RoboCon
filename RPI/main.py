################################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN       ##
################################################
# Main software entry point for all RPI python #
# scripts. This script initiates threads and   #
# joins them and waits for them to finish.     #                    
################################################
##                    RPI                     ##
################################################

import robocontrol as ctrl
import controlserver as cSrv
import time
import threading
import queue

actionQue = queue.Queue()       # used to transfer action required from the server to the control thread 
instructionQue = queue.Queue()  # used to indicate server thread when the task is completed by the control thread

srv = cSrv.ControlServer()      # server object

#___define threads, pass deques to allow comms between threads and tell it what thread routine to call________#
serverThread = threading.Thread(target=srv.routine, args=(actionQue,instructionQue))                # server  #
controlThread = threading.Thread(target=ctrl.controlThreadRoutine,args=(actionQue,instructionQue))  # control #
#-------------------------------------------------------------------------------------------------------------#

#___start threads_______#
serverThread.start()    #
controlThread.start()   #
#-----------------------#

#___join threads and wait for them to finish___#
serverThread.join()                            #
controlThread.join()                           #
#----------------------------------------------#

del srv # call server class destructor





########## TEST SCRIPT ##########
#--------------------------------    
# mDriver = MotorDriver()            

# for i in range(120):
#     mDriver.setBaseAngle(i)
#     mDriver.setElbowAngle(i)

# time.sleep(2)
# del mDriver
#--------------------------------
