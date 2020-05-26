import cv2
import numpy as np
import time
import sys

FRAME_RATE = 3
VIDEO_PATH = './TrainingVideos/'

class GameRecord():
    def __init__(self,board_size=30):
        self.BOX_SIZE = board_size # Row height and column width in pixels
        self.BOX_CENTER = int(board_size*0.5)
        self.ROWs = 6
        self.COLs = 7
        self.BOARD_SIZE = (self.BOX_SIZE*self.ROWs,self.BOX_SIZE*self.COLs,3)
        self.LINE_COLOUR = (255,0,0)
        self.LINE_THICKNESS = 2

        self.DISC_RADIUS = int(board_size*0.45)
        self.RED_DISC_COLOUR = (0,0,255)
        self.YELLOW_DISC_COLOUR = (0,255,255) # dont know yet
        
        self.reset()

    
    #___
    def reset(self):
        #initiate image with black background
        self.gameImg = np.zeros(self.BOARD_SIZE, dtype="uint8")
        
        self.gameVid = []
        vid_height, vid_width, _ = self.gameImg.shape
        self.vid_size = (vid_width,vid_height)

        self._draw_Grid()

    #___
    def _draw_Grid(self):
        for i in range(1,self.ROWs):
            #draw columns
            self.gameImg = cv2.line(self.gameImg,(self.BOX_SIZE*i,0),(self.BOX_SIZE*i,self.BOX_SIZE*self.ROWs),self.LINE_COLOUR,self.LINE_THICKNESS)
            #draw rows
            self.gameImg = cv2.line(self.gameImg,(0,self.BOX_SIZE*i),(self.BOX_SIZE*self.COLs,self.BOX_SIZE*i),self.LINE_COLOUR,self.LINE_THICKNESS)

        # draw last column
        self.gameImg = cv2.line(self.gameImg,(self.BOX_SIZE*self.ROWs,0),(self.BOX_SIZE*self.ROWs,self.BOX_SIZE*self.ROWs),self.LINE_COLOUR,self.LINE_THICKNESS)
    
    
    #___
    def __del__(self):
        out = cv2.VideoWriter(VIDEO_PATH+"trainingVideo.mp4",cv2.VideoWriter_fourcc('m','p','4','v'), FRAME_RATE, self.vid_size)
 
        for i in range(len(self.gameVid)):
            out.write(self.gameVid[i])
        out.release()

        print("\n\n>> TRAINING VIDEO SAVED <<\n")


    #___
    def signal_handler(self,sig, frame):
        self.__del__()
        # print("You pressed Ctrl+C: SAVING TRAINING VIDEO...")
        sys.exit(0)
    

    #___
    def save_video(self,name):
        out = cv2.VideoWriter(VIDEO_PATH+name+".mp4",cv2.VideoWriter_fourcc('m','p','4','v'), FRAME_RATE, self.vid_size)
 
        for i in range(len(self.gameVid)):
            out.write(self.gameVid[i])
        out.release()

        print("\n\n>> TRAINING VIDEO SAVED <<\n")

    #___
    def update_board(self,state):
        self.gameImg = np.zeros(self.BOARD_SIZE, dtype="uint8")
        self._draw_Grid()
        
        for i in range(len(state)):
            disc_location = self._disc_state2location(i)
            if (state[i] == 1):
                self._draw_disc("red",disc_location)
                # print(f"^^ {i} ^^RED {state[i]}^^^^^")
                # input()
            elif (state[i] == 2):
                self._draw_disc("yellow",disc_location)
                # print(f"^^ {i} ^^YELLOW {state[i]}^^^^^")
                # input()
        self.gameImgFlipped = cv2.flip(self.gameImg,0)

        #---------------------------------------------#
        self.gameVid.append(self.gameImgFlipped)

        # print(f"\nStates received for VID : {state}")        
        # # time.sleep(1)
        # cv2.imshow("Connect4",self.gameImgFlipped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    #___
    def gameFinished(self,episode,whoWonTheGame):
        outroImg = np.zeros(self.BOARD_SIZE, dtype="uint8") #np.copy(self.gameImgFlipped)

        outroImgLoc = (self.BOX_CENTER, self.BOX_SIZE)
        outroImgFont = cv2.FONT_HERSHEY_SIMPLEX
        outroImgFontSize = (outroImg.shape[0] * outroImg.shape[1]) / (1000*130)
        outroImgFontColour = (255,255,255)
        outroImgLineType = 1

        if (episode == 1):
            if (whoWonTheGame == 2):
                outroImg = cv2.putText(outroImg, f">> RED WON << EPISODE : {str(episode)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 1):
                outroImg = cv2.putText(outroImg, f">> YELLOW WON << EPISODE : {str(episode)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 0):
                outroImg = cv2.putText(outroImg, f">> ITS A DRAW << EPISODE : {str(episode)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
        else:
            if (whoWonTheGame == 2):
                outroImg = cv2.putText(outroImg, f">> RED WON << EPISODE : {str(episode-1)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 1):
                outroImg = cv2.putText(outroImg, f">> YELLOW WON << EPISODE : {str(episode-1)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 0):
                outroImg = cv2.putText(outroImg, f">> ITS A DRAW << EPISODE : {str(episode-1)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
        for i in range(3):
            self.gameVid.append(outroImg)

    #___
    def _draw_disc(self,discColour,discLocation):
        disc_colour = (0,0,0)
        if (discColour == "red"):
            disc_colour = self.RED_DISC_COLOUR
        elif (discColour == "yellow"):
            disc_colour = self.YELLOW_DISC_COLOUR
        
        cv2.circle(self.gameImg,discLocation,self.DISC_RADIUS,disc_colour,-1)


    #___
    def _disc_state2location(self,stateNum):
    
        row, col = self.state_2_row_col(stateNum)
        # print(f"state: {stateNum}, row : {row}, col : {col}")
        
        if (stateNum <= 0):
            loc = (self.BOX_CENTER,self.BOX_CENTER)

        elif (stateNum < self.COLs):
            loc = (self.BOX_CENTER+(stateNum*self.BOX_SIZE),self.BOX_CENTER)
        
        elif (col == 0):
            loc = (self.BOX_CENTER,self.BOX_CENTER+(row*self.BOX_SIZE))
        
        elif (col > 0):
            loc = (self.BOX_CENTER+(col*self.BOX_SIZE),self.BOX_CENTER+(row*self.BOX_SIZE))
        
        else:
            loc = (0,0)
            print("disk location unknown")        
        
        return loc


    #___convert coordinates to states, index for state, rows and cols start from 0
    def row_col_2_state(self, rows, cols):
        return (rows * self.COLs) + cols


    #___convert states to coordinates, index for state, rows and cols start from 0
    def state_2_row_col(self, state):
        col = state % self.COLs
        row = (state - col) / 7
        return int(row) , int(col)


##############################
##          TEST            ##
##############################
# import random
# rec = GameRecord(40)
# boardState = np.zeros(42)

# for i in range(len(boardState)):
#     boardState[i] = random.randint(0,2) 
# rec.update_board(boardState)

# for i in range(len(boardState)):
#     boardState[i] = random.randint(0,2) 
# rec.update_board(boardState)

# for i in range(len(boardState)):
#     boardState[i] = random.randint(0,2) 
# rec.update_board(boardState)

# for i in range(len(boardState)):
#     boardState[i] = random.randint(0,2) 
# rec.update_board(boardState)