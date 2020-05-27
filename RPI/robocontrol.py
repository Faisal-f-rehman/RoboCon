from motordriver import MotorDriver
import matplotlib.pyplot as plt
import math
from numpy import arange
import time
import queue

SHUTDOWN_CODE = 147258
CONST_R2D = 57.2957795131
CONST_D2R = 0.01745329251
TEST_IK = False
TEST_TRAJ = False
TRAJ_STEPS_BETWEEM_EACH_POINT = 120
WRIST_TO_BOARD_HEIGHT = 35

class RoboControl(MotorDriver):
#class RoboControl:    
    #------------------------------------------    
    #Constructor
    def __init__(self):
        MotorDriver.__init__(self)
        self.L1 = 0.17996
        self.L2 = 0.21085
        self.WORKSPACE = 0.39081
        self.JOINT_RANGE_MAX = 170
        self.JOINT_RANGE_MIN = 10
        self.Q_BASE_ZERO_POS_ADJ = 45 * CONST_D2R
        self.Q_ELBOW_ZERO_POS_ADJ = self.JOINT_RANGE_MAX * CONST_D2R
        self.POS1 = [-0.218,0.202]
        self.POS2 = [-0.195,0.210]
        self.POS3 = [-0.150,0.225]
        self.POS4 = [-0.115,0.235]
        self.POS5 = [-0.085,0.250]
        self.POS6 = [-0.053,0.265]
        self.POS7 = [-0.025,0.275]
        
        self.curPos = [0.006312345009324774,0.04546727240325704]
        self.HOME_POS = [0.006312345009324774,0.04546727240325704]
    #------------------------------------------    
    #Destructor
    def __del__(self):
        MotorDriver.__del__(self)
    
    #------------------------------------------
    def setPose(self,x,y):
        MotorDriver.resetWristHeight(self)
        
        qBase, qElbow, qWrist = self.sLineTraj(x,y)
        
        for i in range(len(qBase)):
            MotorDriver.setBaseAngle(self,qBase[i])
            MotorDriver.setElbowAngle(self,qElbow[i])
            MotorDriver.setWristAngle(self,qWrist[i])
        
        
        if (x < -0.14):
            MotorDriver.setWristHeight(self,WRIST_TO_BOARD_HEIGHT+15)
        else:
            MotorDriver.setWristHeight(self,WRIST_TO_BOARD_HEIGHT)
        
        # input()
        MotorDriver.resetWristHeight(self)
        if ((self.curPos[0] != self.HOME_POS[0]) and (self.curPos[1] != self.HOME_POS[1])):
            
            qBase, qElbow, qWrist = self.sLineTraj(self.HOME_POS[0],self.HOME_POS[1])
            
            for i in range(len(qBase)):
                MotorDriver.setBaseAngle(self,qBase[i])
                MotorDriver.setElbowAngle(self,qElbow[i])
                MotorDriver.setWristAngle(self,qWrist[i])
            time.sleep(0.2)
            
            
        return qBase, qElbow, qWrist

    #------------------------------------------    
    #Takes required end pos, creates straight line trajactory
    def sLineTraj(self,xE,yE):
        qW = 0
        steps = TRAJ_STEPS_BETWEEM_EACH_POINT
        trajX = []
        trajY = []
        
        xE,yE = self.setCoordinateLimits(xE,yE)
        
        xS = self.curPos[0]
        yS = self.curPos[1]
        
        mY = yE - yS
        mX = xE - xS
        if (mY == 0.000000):
            mY = 0.000001
        if (mX == 0.000000):
            mX = 0.000001
        
        
        #Find gradient of the line 
        m = mY / mX

        if abs(xE - xS) > abs(yE - yS):
            resolution = abs(xE - xS) / steps;
            if (xE < xS):
                for i in arange(xS,xE,-resolution):
                    trajX.append(i)
            else:
                for i in arange(xS,xE,resolution):  
                    trajX.append(i)
            
            for i in range(len(trajX)):
                trajY.append(yS + m*(trajX[i] - xS))
    
        else:
            resolution = abs(yE - yS) / steps
            if (yE < yS):
                for i in arange(yS,yE,-resolution):    
                    trajY.append(i)
            else:
                for i in arange(yS,yE,resolution):
                    trajY.append(i)

            for i in range(len(trajY)):
                trajX.append(((trajY[i] - yS) / m) + xS)
        
        self.curPos[0] = trajX[len(trajX)-1]
        self.curPos[1] = trajY[len(trajY)-1]

        print("X = {}   Y = {}".format(self.curPos[0],self.curPos[1]))

        if (TEST_TRAJ):
            plt.plot(trajX,trajY)
            plt.xlim(-self.WORKSPACE, self.WORKSPACE)
            plt.ylim(-self.WORKSPACE, self.WORKSPACE)
            plt.show()
                
        
        self.debugIDX = 0
        qBase = [None]*len(trajX)
        qElbow = [None]*len(trajX)
        qWrist = [None]*len(trajX)
        for i in range(len(trajX)):
            self.debugIDX += 1
            qBase[i], qElbow[i], qWrist[i] = self.ik(trajX[i],trajY[i],qW)
            qBase[i] = int(qBase[i])
            qElbow[i] = int(qElbow[i])
            qWrist[i] = int(qWrist[i])
            # print("{}\t trajX : {}\t trajY : {}\t qBase : {}\t qElbow : {}".format(i,trajX[i],trajY[i],qBase[i],qElbow[i]))
        
        
#         if (qWrist >= qWs):
#             distance = qWrist - qWs
#             qWRes = int(distance / len(trajX))
#             for i in range(qWrist,qWs,qWRes):
#                 qWristTraj.append(i)
#                 
#         elif (qWrist < qWs):
#             distance = qWs - qWrist
#             qWRes = int(distance / len(trajX))
#             for i in range(qWs,qWrist,qWRes):
#                 qWristTraj.append(i)
#         
        
#         self.curWrist_q = qWrist(len(qWrist)-1)

        return qBase, qElbow, qWrist

    #------------------------------------------    
    #Calculate inverse kinematics
    #q0 = atan2(y,x) - atan2(L2sin(q1) , L1L2cos(q1))
    #q1 = (L1^2 + L2^2 - x^2 - y^2) / (2L1L2)
    #q2 = (q0 - q1) + required angle
    def ik(self,x,y,qWrist):
        q = [0.0,0.0,0.0]

        Ka = self.L1**2 + self.L2**2 - x**2 - y**2
        Kb = 2 * self.L1 * self.L2
        
        if (Ka == 0): Ka = 0.000001
        
        K = Ka / Kb
        
        if ((K > -1) and (K < 1)):
            q[1] = math.pi - math.acos(K) 
        elif (K >= 1):
            q[1] = math.acos(0.999999)
        elif (K <= -1):
            q[1] = math.acos(-0.999999)

        q[0] = math.atan2(y,x) - math.atan2((self.L2*math.sin(q[1])), (self.L1 + self.L2 * math.cos(q[1])));

        q[2] = (q[1] - q[0]) + qWrist

        q[0] = q[0] + self.Q_BASE_ZERO_POS_ADJ
        q[1] = self.Q_ELBOW_ZERO_POS_ADJ - q[1]

        for i in range(len(q)):
            q[i] = q[i] * CONST_R2D

        self.setJointLimits(q)
        
        if (TEST_IK):
            self.fk(q[0],q[1])
   
        # print("qBase = {}   qElbow = {}    qWrist = {}".format(q[0],q[1],q[2]))      
        # print("{}\t X : {}\t Y : {}\t qBase : {}\t qElbow : {}".format(i,x,y,q[0],q[1]))

        return q[0], q[1], q[2]      


    #------------------------------------------    
    #Calculate forward kinematics (used only for checking IK)
    #x0 = l1cos(q0)
    #y0 = l1sin(q0)
    #x1 = l1cos(q0) + l2cos(q0 + q1)
    #y1 = l1sin(q0) + l2sin(q0 + q1)
    def fk(self,q0,q1):
        
        q0 *= CONST_D2R
        q1 *= CONST_D2R
        q0 = q0 - self.Q_BASE_ZERO_POS_ADJ
        q1 = self.Q_ELBOW_ZERO_POS_ADJ - q1

        x0 = self.L1 * math.cos(q0)
        y0 = self.L1 * math.sin(q0)
        x1 = self.L1 * math.cos(q0) + self.L2 * math.cos(q0 + q1)
        y1 = self.L1 * math.sin(q0) + self.L2 * math.sin(q0 + q1)
        
        # self.curPos[0] = x1
        # self.curPos[1] = y1

        if (TEST_IK):
            print("FK: x = {}   y = {}".format(x1,y1))
            self.drawArm(x0,y0,x1,y1)
        
        return x0,y0,x1,y1


    #------------------------------------------    
    #draw 2D arm
    def drawArm(self,x0,y0,x1,y1):
        plt.plot([0.0,x0,x1],[0.0,y0,y1])
        plt.xlim(-self.WORKSPACE, self.WORKSPACE)
        plt.ylim(-self.WORKSPACE, self.WORKSPACE)
        plt.show()


    #------------------------------------------    
    #Sets limit on joint angles
    def setJointLimits(self,q):
        for i in range(len(q)):
            # print("In setJointLimits, checking q{} = {}".format(i,q[i]))
            if (q[i] > self.JOINT_RANGE_MAX):
                #if (i < 2):
                    #print("MAX LIMIT SET for {} q{}: was {} now {}".format(self.debugIDX,i,q[i],self.JOINT_RANGE_MAX))
                q[i] = self.JOINT_RANGE_MAX
                
            elif (q[i] < self.JOINT_RANGE_MIN):
                 #if (i < 2):
                    #print("MIN LIMIT SET for {} q{}: was {} now {}".format(self.debugIDX,i,q[i],self.JOINT_RANGE_MIN))
                 q[i] = self.JOINT_RANGE_MIN
                 

    #------------------------------------------    
    #Sets limit on cartisean coordinates
    def setCoordinateLimits(self,x,y):
        xy = math.sqrt(x**2 + y**2)
        OFFSET = 0.001

        while(xy >= self.WORKSPACE):
            if (abs(x) > abs(y)):
                if (x > 0):
                    x -= OFFSET
                else:
                    x += OFFSET
            else:
                if (y > 0):
                    y -= OFFSET
                else:
                    y += OFFSET
                
            xy = math.sqrt(x**2 + y**2)

        return x,y
        




########################################################################
def controlThreadRoutine(actionQue, instructionQue):
    
    ########################
    rCtrl = RoboControl() ##
    ########################

    if (True):
        while (True):
            actionAccepted = False
            while (not actionAccepted):
                try:
                    # action = int(input("User's turn, enter move between 1 to 7 or enter 0 to quit the game : "))
                    action = actionQue.get()
                    if(action == SHUTDOWN_CODE):
                        actionAccepted = True
                    elif (action > 7):
                        action = 7
                    elif (action < 1):
                        action = 1
                    actionAccepted = True

                except ValueError:
                    print("===>>> Error: in controlThreadRoutine, RoboCon Move is invalid <<<===")

            if (action == SHUTDOWN_CODE):
                instructionQue.put(1)
                break
            elif (action == 1):
                rCtrl.setPose(rCtrl.POS1[0],rCtrl.POS1[1])
            elif (action == 2):            
                rCtrl.setPose(rCtrl.POS2[0],rCtrl.POS2[1])
            elif (action == 3):
                rCtrl.setPose(rCtrl.POS3[0],rCtrl.POS3[1])
            elif (action == 4):
                rCtrl.setPose(rCtrl.POS4[0],rCtrl.POS4[1])
            elif (action == 5):
                rCtrl.setPose(rCtrl.POS5[0],rCtrl.POS5[1])
            elif (action == 6):
                rCtrl.setPose(rCtrl.POS6[0],rCtrl.POS6[1])
            elif (action == 7):
                rCtrl.setPose(rCtrl.POS7[0],rCtrl.POS7[1])
            instructionQue.put(1)

        
    del rCtrl
    time.sleep(1)


    # #__FK TEST____________________________________________________________________________
    # if (False):
    #     x0,y0,x1,y1 = rCtrl.fk(0,0)
    #     print("x0 = {}  y0 = {}  x1 = {}  y1 = {}".format(x0,y0,x1,y1))



    # #__IK TEST____________________________________________________________________________
    # if (False):
    #     rCtrl.ik(0.1,0.2,0.0)



    # #__TRAJACTORY TEST____________________________________________________________________ 
    # if (False):
    #     qB1, qE1, _ = rCtrl.sLineTraj(0.1,0.3)
    #     qB2, qE2, _ = rCtrl.sLineTraj(-0.1,0.3)
    #     qBase = qB1+qB2
    #     qElbow = qE1+qE2

    #     #for i in range(len(qBase)):
    #     #    print("qBase[{}] : {} \t qElbow[{}] : {}".format(i,qBase[i],i,qElbow[i]))
        
    #     x0 = [None]*len(qBase)
    #     y0 = [None]*len(qBase)
    #     x1 = [None]*len(qBase)
    #     y1 = [None]*len(qBase)

    #     def make_fig():
    #         plt.xlim(-rCtrl.WORKSPACE, rCtrl.WORKSPACE)
    #         plt.ylim(-rCtrl.WORKSPACE, rCtrl.WORKSPACE)
    #         plt.plot([0.0,x0[i],x1[i]],[0.0,y0[i],y1[i]])  # I think you meant this

    #     plt.ion()  # enable interactivity
    #     fig = plt.figure()  # make a figure

    #     for i in range(len(qBase)):
    #         x0[i],y0[i],x1[i],y1[i], = rCtrl.fk(qBase[i],qElbow[i])
    #         # print("x0[{}] : {},\t y0[{}] : {},\t x1[{}] : {},\t y1[{}] : {}".format(i,x0[i],i,y0[i],i,x1[i],i,y1[i]))


    #     for i in range(len(qBase)):
    #         drawnow(make_fig)



    # #___SET POSE TEST______________________________________________________________________
    # if(False):
    #     qB1, qE1, _ = rCtrl.setPose(0.1,0.3,0.0)
    #     qB2, qE2, _ = rCtrl.setPose(-0.1,0.3,0.0)
    #     qBase = qB1+qB2
    #     qElbow = qE1+qE2

    #     for i in range(len(qBase)):
    #         print("qBase[{}] : {} \t qElbow[{}] : {}".format(i,qBase[i],i,qElbow[i]))
    #     x0 = [None]*len(qBase)
    #     y0 = [None]*len(qBase)
    #     x1 = [None]*len(qBase)
    #     y1 = [None]*len(qBase)

    #     def make_fig():
    #         plt.xlim(-rCtrl.WORKSPACE, rCtrl.WORKSPACE)
    #         plt.ylim(-rCtrl.WORKSPACE, rCtrl.WORKSPACE)
    #         plt.plot([0.0,x0[i],x1[i]],[0.0,y0[i],y1[i]])  # I think you meant this

    #     plt.ion()  # enable interactivity
    #     fig = plt.figure()  # make a figure

    #     for i in range(len(qBase)):
    #         x0[i],y0[i],x1[i],y1[i], = rCtrl.fk(qBase[i],qElbow[i])
    #         # print("x0[{}] : {},\t y0[{}] : {},\t x1[{}] : {},\t y1[{}] : {}".format(i,x0[i],i,y0[i],i,x1[i],i,y1[i]))


    #     for i in range(len(qBase)):
    #         drawnow(make_fig)

    # if (False):
    #     rCtrl.setPose(rCtrl.POS1[0],rCtrl.POS1[1],rCtrl.POS1[2])
    #     time.sleep(3)
    #     rCtrl.setPose(rCtrl.POS2[0],rCtrl.POS2[1],rCtrl.POS2[2])
    #     time.sleep(3)
    #     rCtrl.setPose(rCtrl.POS3[0],rCtrl.POS3[1],rCtrl.POS3[2])
    #     time.sleep(3)
    #     rCtrl.setPose(rCtrl.POS4[0],rCtrl.POS4[1],rCtrl.POS4[2])
    #     time.sleep(3)
    #     rCtrl.setPose(rCtrl.POS5[0],rCtrl.POS5[1],rCtrl.POS5[2])
    #     time.sleep(3)
    #     rCtrl.setPose(rCtrl.POS6[0],rCtrl.POS6[1],rCtrl.POS6[2])
    #     time.sleep(3)
    #     rCtrl.setPose(rCtrl.POS7[0],rCtrl.POS7[1],rCtrl.POS7[2])
    #     time.sleep(3)
    #     #qB2, qE2, _ = rCtrl.setPose(-0.1,0.3,0.0)
    #     #time.sleep(1)

    