###############################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN      ##
###############################################
'''
RoboControl class:                                                                        
This class inherits the MotorDriver class to drive the motors. Given x and y coordinates  
it calculates straight-line trajectory with help of inverse kinematics to get arrays of   
joint angles, which are then used to drive the motors to move the end-effector at the     
desired location.
''' 
###############################################
##                    RPI                    ##
###############################################

from motordriver import MotorDriver
import matplotlib.pyplot as plt
import math
from numpy import arange
import time
import queue

SHUTDOWN_CODE = 147258                  # This number is used through out the RoboCon software which tells each script to shutdown
CONST_R2D = 57.2957795131               # degree = radians * 180 / pi, where 180 / pi = 57.2957795131 (10 dp)
CONST_D2R = 0.01745329251               # radian = degrees * pi / 180, where pi / 180 = 0.01745329251 (10 dp)
TEST_IK = False                         # used with the test scripts at the EOF, set true if testing IK
TEST_TRAJ = False                       # used with the test scripts at the EOF, set true if testing trajectory
TRAJ_STEPS_BETWEEM_EACH_POINT = 120     # this sets the resolution of the straight line trajectory, greater number slower speed more percision and vice versa
WRIST_TO_BOARD_HEIGHT = 35              # defines the angle of the linear actuator when dropping a disc in the connect 4 game board

class RoboControl(MotorDriver):
    #==================================================  
    #Constructor
    def __init__(self):
        MotorDriver.__init__(self)
        
        #___link lengths, used in various places including for kinematics, if________# 
        #   design of the links have changed then changing these will adjust the IK  #
        self.L1 = 0.17996  # link between base and L2                                #
        self.L2 = 0.21085  # link between L1 and end-effector                        #
        #----------------------------------------------------------------------------#

        self.WORKSPACE = 0.39081    # defines the radius of the robot's workspace
        self.JOINT_RANGE_MAX = 170  # upper limit revolte joint
        self.JOINT_RANGE_MIN = 10   # lower limit revolte joint

        #___home position joint angles for the base and elbow joints___#
        self.Q_BASE_ZERO_POS_ADJ = 45 * CONST_D2R                      #
        self.Q_ELBOW_ZERO_POS_ADJ = self.JOINT_RANGE_MAX * CONST_D2R   #
        #--------------------------------------------------------------#

        #___defines the 7 slot position___#
        #   coordinates (x,y) of the      # 
        #   connect 4 game board          #
        self.POS1 = [-0.218,0.202]        #
        self.POS2 = [-0.195,0.210]        #
        self.POS3 = [-0.150,0.225]        #
        self.POS4 = [-0.115,0.235]        #
        self.POS5 = [-0.085,0.250]        #
        self.POS6 = [-0.053,0.265]        #
        self.POS7 = [-0.025,0.275]        #
        #---------------------------------#
        
        #___define home position and set current position to home position___#          
        self.curPos = [0.006312345009324774,0.04546727240325704]             #
        self.HOME_POS = [0.006312345009324774,0.04546727240325704]           #
        #--------------------------------------------------------------------#



    #==================================================
    #Destructor
    def __del__(self):
        MotorDriver.__del__(self)   # call MotorDriver class destructor
    


    #==================================================
    # This method is the only one that is called outside
    # the class. It takes x and y coordinates and calls 
    # other member functions to calculate straight-line
    # trajectory to drive the motors.
    def setPose(self,x,y):
        
        MotorDriver.resetWristHeight(self)              # set linear actuator to home position
        qBase, qElbow, qWrist = self.sLineTraj(x,y)     # get straight-line trajectory
        
        #___drive motors with trajectory arrays_______#
        for i in range(len(qBase)):                   # 
            MotorDriver.setBaseAngle(self,qBase[i])   #
            MotorDriver.setElbowAngle(self,qElbow[i]) #
            MotorDriver.setWristAngle(self,qWrist[i]) #
        #---------------------------------------------#
        
        #___move linear actuator closer to the connect 4 game board___#
        if (x < -0.14):                                               #      
            MotorDriver.setWristHeight(self,WRIST_TO_BOARD_HEIGHT+15) #
        else:                                                         #
            MotorDriver.setWristHeight(self,WRIST_TO_BOARD_HEIGHT)    #
        #-------------------------------------------------------------#

        # input() # uncomment to stop the robot arm when the EE reaches the game board

        MotorDriver.resetWristHeight(self)  # reset linear actuator to home position

        #___if current position is not home position, move the robot back to home position___#
        if ((self.curPos[0] != self.HOME_POS[0]) and (self.curPos[1] != self.HOME_POS[1])):  #
            qBase, qElbow, qWrist = self.sLineTraj(self.HOME_POS[0],self.HOME_POS[1])        #
                                                                                             #
            for i in range(len(qBase)):                                                      #
                MotorDriver.setBaseAngle(self,qBase[i])                                      #
                MotorDriver.setElbowAngle(self,qElbow[i])                                    #
                MotorDriver.setWristAngle(self,qWrist[i])                                    #
            time.sleep(0.2)                                                                  #
        #------------------------------------------------------------------------------------#
            
        return qBase, qElbow, qWrist    # return last trajectory calculated, not really required at the moment



    #=========================================================
    # Takes x,y coordinates for the end-effector position, and
    # returns arrays or straight line trajactory joint angles
    def sLineTraj(self,xE,yE):
        
        #___define local variables and constants___#
        qW = 0                                     #
        steps = TRAJ_STEPS_BETWEEM_EACH_POINT      #
        trajX = []                                 #
        trajY = []                                 #
        #------------------------------------------#
        
        xE,yE = self.setCoordinateLimits(xE,yE) # keeps coordinates within the workspace
        
        #___set starting position___#
        #   as current position     #
        xS = self.curPos[0]         #
        yS = self.curPos[1]         #
        #---------------------------#

        #___calculate gradient___#
        #   of the line          #
        mY = yE - yS             #
        mX = xE - xS             #
                                 #
        if (mY == 0.000000):     #
            mY = 0.000001        #
        if (mX == 0.000000):     #
            mX = 0.000001        #
                                 #
        m = mY / mX              #
        #------------------------#

        # Use greater distance, x_end - x_start or y_end - y_start: 
        if abs(xE - xS) > abs(yE - yS):             # use distance x 
            resolution = abs(xE - xS) / steps       # step size between two points that lie on x-axis
            
            # store all points between the two points in x-axis direction______#
            if (xE < xS):                           # direction: +'ve to -'ve  #
                for i in arange(xS,xE,-resolution):                            #
                    trajX.append(i)                                            #
            else:                                   # direction: -'ve to +'ve  #
                for i in arange(xS,xE,resolution):                             #
                    trajX.append(i)                                            #
            #------------------------------------------------------------------#
            
            #___calculate points in y-axis for all points in trajX___#
            for i in range(len(trajX)):                              #
                trajY.append(yS + m*(trajX[i] - xS))                 #
            #--------------------------------------------------------#

        else:                                       # use distance y 
            resolution = abs(yE - yS) / steps       # step size between two points that lie on y-axis

            # store all points between the two points in x-axis direction______#
            if (yE < yS):                           # direction: +'ve to -'ve  #
                for i in arange(yS,yE,-resolution):                            #
                    trajY.append(i)                                            #
            else:                                   # direction: -'ve to +'ve  #
                for i in arange(yS,yE,resolution):                             #
                    trajY.append(i)                                            #
            #------------------------------------------------------------------#

            #___calculate points in x-axis for all points in trajY___#
            for i in range(len(trajY)):                              #
                trajX.append(((trajY[i] - yS) / m) + xS)             #
            #--------------------------------------------------------#

        #___update current and home positions___#
        self.curPos[0] = trajX[len(trajX)-1]    #
        self.curPos[1] = trajY[len(trajY)-1]    #
        #---------------------------------------#

        #___DEBUG_______________________________________________________#
        print("X = {}   Y = {}".format(self.curPos[0],self.curPos[1]))  #
        #---------------------------------------------------------------#

        #___TEST - SIMULATE TRAJECTORY________________#
        if (TEST_TRAJ):                               #
            plt.plot(trajX,trajY)                     #
            plt.xlim(-self.WORKSPACE, self.WORKSPACE) #
            plt.ylim(-self.WORKSPACE, self.WORKSPACE) #
            plt.show()                                #
        #---------------------------------------------#
        
        #___Convert trajectory positions to trajectory in joint angles using IK____
        #___Define list variables__#                                              |
        qBase = [None]*len(trajX)  #                                              |          
        qElbow = [None]*len(trajX) #                                              |
        qWrist = [None]*len(trajX) #                                              |
        #                                                                         |
        #___perform IK on every point in trajectory and store______________#      |
        #   joint angles for each position in lists / arrays               #      |
        for i in range(len(trajX)):                                        #      |
            qBase[i], qElbow[i], qWrist[i] = self.ik(trajX[i],trajY[i],qW) #      |
            qBase[i] = int(qBase[i])                                       #      |
            qElbow[i] = int(qElbow[i])                                     #      |
            qWrist[i] = int(qWrist[i])                                     #      |
        #-------------------------------------------------------------------------|
        
        #___DEBUG_________________________________________________________________________________________________________#
        # print("{}\t trajX : {}\t trajY : {}\t qBase : {}\t qElbow : {}".format(i,trajX[i],trajY[i],qBase[i],qElbow[i])) #
        #-----------------------------------------------------------------------------------------------------------------#
        
        return qBase, qElbow, qWrist # return trajectory joint angles



    #==================================================
    #Calculate inverse kinematics
    #q0 = atan2(y,x) - atan2(L2sin(q1) , L1L2cos(q1))
    #q1 = (L1^2 + L2^2 - x^2 - y^2) / (2L1L2)
    #q2 = (q0 - q1) + required angle
    def ik(self,x,y,qWrist):
        q = [0.0,0.0,0.0]

        #___Calculate q1_____________________________________________
        Ka = self.L1**2 + self.L2**2 - x**2 - y**2  # numerator     |
        Kb = 2 * self.L1 * self.L2                  # denominator   |
        #                                                           |
        if (Ka == 0): Ka = 0.000001 # numerator must not be zero    |
        K = Ka / Kb                 # q1 before offset              |
        #                                                           |
        #___quadrunt offset_________________#                       |
        if ((K > -1) and (K < 1)):          #                       |
            q[1] = math.pi - math.acos(K)   #                       |
        elif (K >= 1):                      #                       |
            q[1] = math.acos(0.999999)      #                       |
        elif (K <= -1):                     #                       |
            q[1] = math.acos(-0.999999)     #                       |
        #-----------------------------------------------------------|

        #___Calculate q0_____________________________________________________________________________________#
        q[0] = math.atan2(y,x) - math.atan2((self.L2*math.sin(q[1])), (self.L1 + self.L2 * math.cos(q[1])))  #
        #----------------------------------------------------------------------------------------------------#

        q[2] = (q[1] - q[0]) + qWrist   # calculate q2 such that the end-effector is always forward

        #___adjustments due to home position offsets___#
        q[0] = q[0] + self.Q_BASE_ZERO_POS_ADJ         #
        q[1] = self.Q_ELBOW_ZERO_POS_ADJ - q[1]        #
        #----------------------------------------------#

        #___convert all joint angles___# 
        #   from radians to degrees    #
        for i in range(len(q)):        #
            q[i] = q[i] * CONST_R2D    #
        #------------------------------#
        
        self.setJointLimits(q) # check and adjust for joint limits 
        
        #___TEST - SIMULATE IK___#
        if (TEST_IK):            #
            self.fk(q[0],q[1])   #
        #------------------------#

        #___DEBUG__________________________________________________________________________#
        # print("qBase = {}   qElbow = {}    qWrist = {}".format(q[0],q[1],q[2]))          #
        # print("{}\t X : {}\t Y : {}\t qBase : {}\t qElbow : {}".format(i,x,y,q[0],q[1])) #
        #----------------------------------------------------------------------------------#

        return q[0], q[1], q[2] # return three calculated joint angles 



    #==================================================
    # Calculate forward kinematics (used only for checking IK)(TEST SIMULATION)
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

        if (TEST_IK):
            print("FK: x = {}   y = {}".format(x1,y1))
            self.drawArm(x0,y0,x1,y1)
        
        return x0,y0,x1,y1


    #==================================================
    # draw 2D arm (TEST SIMULATION)
    def drawArm(self,x0,y0,x1,y1):
        plt.plot([0.0,x0,x1],[0.0,y0,y1])
        plt.xlim(-self.WORKSPACE, self.WORKSPACE)
        plt.ylim(-self.WORKSPACE, self.WORKSPACE)
        plt.show()


    #==================================================
    #Sets limit on joint angles
    def setJointLimits(self,q):
        for i in range(len(q)):
            if (q[i] > self.JOINT_RANGE_MAX):
                q[i] = self.JOINT_RANGE_MAX
            elif (q[i] < self.JOINT_RANGE_MIN):
                 q[i] = self.JOINT_RANGE_MIN
                 

    #==================================================
    # Sets limit on cartisean coordinates. Checks hypotenuse 
    # of x and y coordinates is less than the radius of the 
    # workspace and if not then gradualy reduces the distance
    # until the hypotenuse length is less than the workspace 
    # radius.
    def setCoordinateLimits(self,x,y):
        xy = math.sqrt(x**2 + y**2) # pythagoras theorem
        STEP_SIZE = 0.001           # adjustment step size

        # enter and keep looping while hypotenuse 
        # is greater than the workspace radius    
        while(xy >= self.WORKSPACE): 
            if (abs(x) > abs(y)):   # reduce the greater (x or y axis) distance 
                if (x > 0):         # check sign (quadrunt)
                    x -= STEP_SIZE
                else:
                    x += STEP_SIZE
            else:
                if (y > 0):
                    y -= STEP_SIZE
                else:
                    y += STEP_SIZE
                
            xy = math.sqrt(x**2 + y**2) # re-calculate hypotenuse

        return x,y  # return new or the same coordinates accordingly 
        
##################################### END OF CLASS #####################################


'''
CONTROL THREAD ROUTINE ---------------------------------------------------------------------
'''
def controlThreadRoutine(actionQue, instructionQue):
    
    ########################
    rCtrl = RoboControl() ## Create control class object / instance
    ########################

    
    while (True):               # thread routine main loop 
        actionAccepted = False  # exit condition for loop to check valid action
        while (not actionAccepted): # keep in loop until the action received makes sense
            try:
                #___Comment out "action = actionQue.get()" and uncomment next line to manually actions_________#
                # action = int(input("User's turn, enter move between 1 to 7 or enter 0 to quit the game : ")) #
                #----------------------------------------------------------------------------------------------#

                action = actionQue.get()     # block and wait until server thread puts an action in the deque
                if(action == SHUTDOWN_CODE): # shutdown if shutdown code recieved as action
                    actionAccepted = True    # set exit loop flag
                elif (action > 7):           # limit action to less than 8
                    action = 7              
                elif (action < 1):           # limit action to greater than 0
                    action = 1
                actionAccepted = True        # if no exception was thrown, set exit loop flag        

            except ValueError:
                print("===>>> Error: in controlThreadRoutine, RoboCon Move is invalid <<<===")

        #___Use action received with appropriate action_________#
        if (action == SHUTDOWN_CODE):                           #
            # Indicates server thread, that required action is  #
            # complete which unblocks the server thread         #
            instructionQue.put(1)                               #
            break                                               #
        elif (action == 1):                                     #
            rCtrl.setPose(rCtrl.POS1[0],rCtrl.POS1[1])          #    
        elif (action == 2):                                     #
            rCtrl.setPose(rCtrl.POS2[0],rCtrl.POS2[1])          #
        elif (action == 3):                                     #
            rCtrl.setPose(rCtrl.POS3[0],rCtrl.POS3[1])          #
        elif (action == 4):                                     #
            rCtrl.setPose(rCtrl.POS4[0],rCtrl.POS4[1])          #
        elif (action == 5):                                     #
            rCtrl.setPose(rCtrl.POS5[0],rCtrl.POS5[1])          #
        elif (action == 6):                                     #
            rCtrl.setPose(rCtrl.POS6[0],rCtrl.POS6[1])          #
        elif (action == 7):                                     #
            rCtrl.setPose(rCtrl.POS7[0],rCtrl.POS7[1])          #
        instructionQue.put(1)                                   #
        #-------------------------------------------------------#
        
    del rCtrl       # call control class destructor
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

    