from adafruit_servokit import ServoKit
import time

JOINT_MOVE_DELAY = 0.01 #sec
SERVO_RANGE_MAX = 170
SERVO_RANGE_MIN = 10
LIN_ACT_RANGE_MAX = 180
LIN_ACT_RANGE_MIN = 30

SERVO_ID_BASE_A = 0
SERVO_ID_BASE_B = 1
SERVO_ID_ELBOW = 2
SERVO_ID_WRIST_REV = 3
SERVO_ID_WRIST_LIN = 4
SERVO_ID_EE = 5

class MotorDriver:
    
    #------------------------------------------    
    #Constructor
    def __init__(self):
        self.sKit = ServoKit(channels=16)
        
        #Set the PWM range to a minimum of 500 and a maximum of 2500:
        self.sKit.servo[SERVO_ID_BASE_A].set_pulse_width_range(500, 2500)
        self.sKit.servo[SERVO_ID_BASE_B].set_pulse_width_range(500, 2500)
        self.sKit.servo[SERVO_ID_ELBOW].set_pulse_width_range(500, 2500)
        
        self.sKit.servo[SERVO_ID_WRIST_REV].set_pulse_width_range(500, 2500)
        self.sKit.servo[SERVO_ID_WRIST_LIN].set_pulse_width_range(1000, 2000)
        self.sKit.servo[SERVO_ID_EE].set_pulse_width_range(600, 2500)
        
        self.qBase = SERVO_RANGE_MIN
        self.qElbow = SERVO_RANGE_MIN
        self.qWristLin = LIN_ACT_RANGE_MIN
        self.qWristRev = SERVO_RANGE_MAX
       
                
        self.setBaseAngle(self.qBase)
        self.setElbowAngle(self.qElbow)
        for i in range(int((SERVO_RANGE_MAX + SERVO_RANGE_MIN) /2),SERVO_RANGE_MAX):
            self.setWristAngle(i)
        self.setWristHeight(self.qWristLin)

    #------------------------------------------    
    #Destructor
    def __del__(self):
        qBaseBuff = self.qBase
        qElbowBuff = self.qElbow
        qWristBuff = self.qWristRev
        qWristMid = int((SERVO_RANGE_MAX + SERVO_RANGE_MIN) /2)
        
        for i in range(qElbowBuff,SERVO_RANGE_MIN,-1):
            self.setElbowAngle(i)
            time.sleep(JOINT_MOVE_DELAY)

        for i in range(qBaseBuff,SERVO_RANGE_MIN,-1):
            self.setBaseAngle(i)
            time.sleep(JOINT_MOVE_DELAY)
                
        if (qWristBuff > qWristMid):    
            for i in range(qWristBuff,qWristMid,-1):
                self.setWristAngle(SERVO_RANGE_MAX-i)
                time.sleep(JOINT_MOVE_DELAY)
        elif (qWristMid > qWristBuff):
            for i in range(qWristBuff,qWristMid):
                self.setWristAngle(SERVO_RANGE_MAX-i)
                time.sleep(JOINT_MOVE_DELAY)
        
        
        self.setWristHeight(LIN_ACT_RANGE_MIN)
        

    #------------------------------------------    
    #Sets base joint angle for both base joint servo motors
    def setBaseAngle(self,q):
        if (q < SERVO_RANGE_MIN):
            q = SERVO_RANGE_MIN
        elif (q > SERVO_RANGE_MAX):
            q = SERVO_RANGE_MAX
    
        self.sKit.servo[SERVO_ID_BASE_A].angle = q
        self.sKit.servo[SERVO_ID_BASE_B].angle = SERVO_RANGE_MAX - q + 8

        self.qBase = q

        time.sleep(JOINT_MOVE_DELAY)


    #------------------------------------------    
    #Sets elbow joint angle
    def setElbowAngle(self,q):
        if (q < SERVO_RANGE_MIN):
            q = SERVO_RANGE_MIN
        elif (q > SERVO_RANGE_MAX):
            q = SERVO_RANGE_MAX

        self.sKit.servo[SERVO_ID_ELBOW].angle = SERVO_RANGE_MAX - q

        self.qElbow = q

        time.sleep(JOINT_MOVE_DELAY)
        
    #------------------------------------------    
    #Sets wrist joint angle
    def setWristAngle(self,q):
        offset = 5
        
        q = SERVO_RANGE_MAX - q + offset
        if (q < SERVO_RANGE_MIN):
            q = SERVO_RANGE_MIN
        elif (q > SERVO_RANGE_MAX):
            q = SERVO_RANGE_MAX
        
        self.sKit.servo[SERVO_ID_WRIST_REV].angle = q
        
        self.qWristRev = q
        
        time.sleep(JOINT_MOVE_DELAY)
    
    
    #------------------------------------------    
    #Sets wrist height only
    def setWristHeight(self,h):
        
        if (h < LIN_ACT_RANGE_MIN):
            h = LIN_ACT_RANGE_MIN
        elif (h > LIN_ACT_RANGE_MAX):
            h = LIN_ACT_RANGE_MAX
            
        self.sKit.servo[SERVO_ID_WRIST_LIN].angle = h
        self.qWristLin = h
        time.sleep(1.0)
    
    
    #------------------------------------------    
    #Reset wrist height only
    def resetWristHeight(self):
        self.sKit.servo[SERVO_ID_WRIST_LIN].angle = LIN_ACT_RANGE_MIN
        time.sleep(1.0)
    

    #------------------------------------------
    #Drop one disk
    def eeDropDisk(self):
        for i in range(1):
            self.sKit.servo[SERVO_ID_EE].angle = 25
            time.sleep(0.3)
            self.sKit.servo[SERVO_ID_EE].angle = 180
            time.sleep(0.3)
