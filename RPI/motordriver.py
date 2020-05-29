#################################################################################
##                       AUTHOR : FAISAL FAZAL-UR-REHMAN                       ##
#################################################################################
# MotorDriver class is writen with the adafruit servokit library to drive the   #
# PWM/Servo Pi Hat:                                                             #
# https://learn.adafruit.com/adafruit-16-channel-pwm-servo-hat-for-raspberry-pi #
#                                                                               #
# Currently setup for 5 revolute motors and 1 linear motor for RoboCon's Robot  #
# This class is inherited by the RoboControl class which provides the joint     #
# angles for the motors.                                                        #
#################################################################################
##                                      RPI                                    ##
#################################################################################

from adafruit_servokit import ServoKit # Adafruit's servo driver library, please see link in the title above
import time

JOINT_MOVE_DELAY = 0.01     # in sec    
SERVO_RANGE_MAX = 170       # upper limit revolte joint
SERVO_RANGE_MIN = 10        # lower limit revolte joint
LIN_ACT_RANGE_MAX = 180     # upper limit linear  joint
LIN_ACT_RANGE_MIN = 30      # lower limit linear  joint

#___servo IDs for the pi hat___#
SERVO_ID_BASE_A = 0            #
SERVO_ID_BASE_B = 1            #
SERVO_ID_ELBOW = 2             #
SERVO_ID_WRIST_REV = 3         #
SERVO_ID_WRIST_LIN = 4         #
SERVO_ID_EE = 5                #
#------------------------------#

class MotorDriver:
    
    #==============================================    
    #Constructor
    def __init__(self):
        self.sKit = ServoKit(channels=16) # servokit object with 16 channels allowing the use of upto 16 motors.
        
        #___Set PWM minimum and maximum ranges for each motor___________________#
        self.sKit.servo[SERVO_ID_BASE_A].set_pulse_width_range(500, 2500)       #
        self.sKit.servo[SERVO_ID_BASE_B].set_pulse_width_range(500, 2500)       #
        self.sKit.servo[SERVO_ID_ELBOW].set_pulse_width_range(500, 2500)        #
                                                                                #
        self.sKit.servo[SERVO_ID_WRIST_REV].set_pulse_width_range(500, 2500)    #
        self.sKit.servo[SERVO_ID_WRIST_LIN].set_pulse_width_range(1000, 2000)   #
        self.sKit.servo[SERVO_ID_EE].set_pulse_width_range(600, 2500)           #
        #-----------------------------------------------------------------------#

        #___set all joint angles to home position___#
        self.qBase = SERVO_RANGE_MIN                #
        self.qElbow = SERVO_RANGE_MIN               #
        self.qWristLin = LIN_ACT_RANGE_MIN          #
        self.qWristRev = SERVO_RANGE_MAX            #
        #-------------------------------------------#

        #___drive all motors to home position________________________________________#         
        self.setBaseAngle(self.qBase)                                                #
        self.setElbowAngle(self.qElbow)                                              #
        for i in range(int((SERVO_RANGE_MAX + SERVO_RANGE_MIN) /2),SERVO_RANGE_MAX): #
            self.setWristAngle(i)                                                    #
        self.setWristHeight(self.qWristLin)                                          #
        #----------------------------------------------------------------------------#

    
    
    #==============================================
    #Destructor
    def __del__(self):
        
        #___grab current joint angles____________________________#
        qBaseBuff = self.qBase                                   #
        qElbowBuff = self.qElbow                                 #
        qWristBuff = self.qWristRev                              #
        qWristMid = int((SERVO_RANGE_MAX + SERVO_RANGE_MIN) /2)  #
        #--------------------------------------------------------#


        #___move all joints from current joint_________#
        #   angles to home positions / joint angles    #           
        for i in range(qElbowBuff,SERVO_RANGE_MIN,-1): #
            self.setElbowAngle(i)                      #
            time.sleep(JOINT_MOVE_DELAY)               #
                                                       #
        for i in range(qBaseBuff,SERVO_RANGE_MIN,-1):  #
            self.setBaseAngle(i)                       #
            time.sleep(JOINT_MOVE_DELAY)               #
                                                       #
        if (qWristBuff > qWristMid):                   #
            for i in range(qWristBuff,qWristMid,-1):   #
                self.setWristAngle(SERVO_RANGE_MAX-i)  #
                time.sleep(JOINT_MOVE_DELAY)           #
        elif (qWristMid > qWristBuff):                 #
            for i in range(qWristBuff,qWristMid):      #
                self.setWristAngle(SERVO_RANGE_MAX-i)  #
                time.sleep(JOINT_MOVE_DELAY)           #
                                                       #
        self.setWristHeight(LIN_ACT_RANGE_MIN)         #
        #----------------------------------------------#

    
    
    #==============================================
    #Sets base joint angle for both base joint servo motors
    def setBaseAngle(self,q):
        
        #___set joint limits________#
        if (q < SERVO_RANGE_MIN):   #
            q = SERVO_RANGE_MIN     #
        elif (q > SERVO_RANGE_MAX): #
            q = SERVO_RANGE_MAX     #
        #---------------------------#

        #___drive motors_________________________________________________#
        self.sKit.servo[SERVO_ID_BASE_A].angle = q                       #
        self.sKit.servo[SERVO_ID_BASE_B].angle = SERVO_RANGE_MAX - q + 8 #
        #----------------------------------------------------------------#

        self.qBase = q                  # update current base joint angle
        time.sleep(JOINT_MOVE_DELAY)    # allow time for the motors to reach the required position 


    
    #==============================================
    #Sets elbow joint angle
    def setElbowAngle(self,q):

        #___set joint limits________#
        if (q < SERVO_RANGE_MIN):   #
            q = SERVO_RANGE_MIN     #
        elif (q > SERVO_RANGE_MAX): #
            q = SERVO_RANGE_MAX     #
        #---------------------------#

        #___drive motor_____________________________________________#
        self.sKit.servo[SERVO_ID_ELBOW].angle = SERVO_RANGE_MAX - q #
        #-----------------------------------------------------------#

        self.qElbow = q                 # update current elbow joint angle
        time.sleep(JOINT_MOVE_DELAY)    # allow time for the motors to reach the required position 
        
    
    
    #==============================================
    #Sets wrist joint angle
    def setWristAngle(self,q):
        
        #___add offset for fine tuning___#
        offset = 0                       #
        q = SERVO_RANGE_MAX - q + offset #
        #--------------------------------#

        #___set joint limits________#
        if (q < SERVO_RANGE_MIN):   #
            q = SERVO_RANGE_MIN     #
        elif (q > SERVO_RANGE_MAX): #
            q = SERVO_RANGE_MAX     #
        #---------------------------#

        #___drive motor_________________________________#
        self.sKit.servo[SERVO_ID_WRIST_REV].angle = q   #
        #-----------------------------------------------#

        self.qWristRev = q              # update current wrist joint angle
        time.sleep(JOINT_MOVE_DELAY)    # allow time for the motors to reach the required position  
    
    
    
    #==============================================
    #Sets wrist height only
    def setWristHeight(self,h):
        
        #___set joint limits__________#
        if (h < LIN_ACT_RANGE_MIN):   #
            h = LIN_ACT_RANGE_MIN     #
        elif (h > LIN_ACT_RANGE_MAX): #
            h = LIN_ACT_RANGE_MAX     #
        #-----------------------------#

        #___drive motor_________________________________#
        self.sKit.servo[SERVO_ID_WRIST_LIN].angle = h   #
        #-----------------------------------------------#
        
        self.qWristLin = h  # update current wrist joint angle
        time.sleep(1.0)     # allow time for the motors to reach the required position  
    
    
    
    #==============================================
    #Reset wrist height only
    def resetWristHeight(self):
        
        #___drive motor_______________________________________________#
        self.sKit.servo[SERVO_ID_WRIST_LIN].angle = LIN_ACT_RANGE_MIN #
        #-------------------------------------------------------------#
        
        time.sleep(1.0) # allow time for the motors to reach the required position  
    

    
    #==============================================
    #Drops one disk
    def eeDropDisk(self):

        for i in range(1): # loop not required currently but could be used for dropping multiple discs
            kit.servo[SERVO_ID_EE].angle = 25   # hit the disc
            time.sleep(0.3)                     # let it move
            kit.servo[SERVO_ID_EE].angle = 180  # get back to home position
            time.sleep(0.3)                     # let it move
