import robocontrol as ctrl
# import controlserver as cSrv
import time
import threading
import queue

actionQue = queue.Queue()
instructionQue = queue.Queue()

# srv = cSrv.ControlServer()
# serverThread = threading.Thread(target=srv.routine, args=(actionQue,instructionQue))
controlThread = threading.Thread(target=ctrl.controlThreadRoutine,args=(actionQue,instructionQue))

# serverThread.start()
controlThread.start()

# serverThread.join()
controlThread.join()

# del srv
#--------------------------------
# mDriver = MotorDriver()

# for i in range(120):
#     mDriver.setBaseAngle(i)
#     mDriver.setElbowAngle(i)

# time.sleep(2)
# del mDriver
#--------------------------------
