################################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN       ##
################################################
# GameRecord class:                            #
# Written in python with OpenCV library, it is #
# part of the RoboCon DQN training scripts. It # 
# creates and stores images of DQN training    # 
# session in the background without effecting  #
# training speed. At the end of a training     #
# session it stiches all the stored images     #
# together and exports a video in the DQN      #
# training directory.                          #
################################################
##                    PC                      ##
################################################

import cv2
import numpy as np
import time
import sys

FRAME_RATE = 3                       # used to set the frame rate while exporting the video  
VIDEO_PATH = './TrainingVideos/'     # file path to save the video

class GameRecord():
    def __init__(self,board_size=30):         
        self.BOX_SIZE = board_size            # row height and column width in pixels
        self.BOX_CENTER = int(board_size*0.5) # centre of a box in pixel (box: grid box around a single slot)
        self.ROWs = 6                         # rows on connect 4 board used
        self.COLs = 7                         # columns on connect 4 board used
        self.BOARD_SIZE = (self.BOX_SIZE*self.ROWs,self.BOX_SIZE*self.COLs,3) # width and height of the board in pixels
        self.LINE_COLOUR = (255,0,0)          # colour in BGR
        self.LINE_THICKNESS = 2
        self.DISC_RADIUS = int(board_size*0.45) # disc size in pixels
        self.RED_DISC_COLOUR = (0,0,255)        # colour in BGR
        self.YELLOW_DISC_COLOUR = (0,255,255) 
        self.reset()
    #=======================================================================


    
    #___reset member variables
    def reset(self):
        #initiate image with black background
        self.gameImg = np.zeros(self.BOARD_SIZE, dtype="uint8")        
        vid_height, vid_width, _ = self.gameImg.shape 
        self.vid_size = (vid_width,vid_height)
        self.gameVid = []
        self._draw_Grid()
    #=======================================================================



    #___draws the board on the image
    def _draw_Grid(self):
        for i in range(1,self.ROWs):
            #draw columns
            self.gameImg = cv2.line(self.gameImg,(self.BOX_SIZE*i,0),(self.BOX_SIZE*i,self.BOX_SIZE*self.ROWs),self.LINE_COLOUR,self.LINE_THICKNESS)
            #draw rows
            self.gameImg = cv2.line(self.gameImg,(0,self.BOX_SIZE*i),(self.BOX_SIZE*self.COLs,self.BOX_SIZE*i),self.LINE_COLOUR,self.LINE_THICKNESS)

        # draw last column
        self.gameImg = cv2.line(self.gameImg,(self.BOX_SIZE*self.ROWs,0),(self.BOX_SIZE*self.ROWs,self.BOX_SIZE*self.ROWs),self.LINE_COLOUR,self.LINE_THICKNESS)
    #=======================================================================
    


    #___destructor
    def __del__(self):
        
        # stitch images together and create a video
        out = cv2.VideoWriter(VIDEO_PATH+"trainingVideo.mp4",cv2.VideoWriter_fourcc('m','p','4','v'), FRAME_RATE, self.vid_size)

        #___export video
        for i in range(len(self.gameVid)):
            out.write(self.gameVid[i])
        out.release()

        print("\n\n>> TRAINING VIDEO SAVED <<\n")
    #=======================================================================



    #___Triggered if user pressed ctrl+c to cancel training
    def signal_handler(self,sig, frame):
        self.__del__()  # call destructor
        # print("You pressed Ctrl+C: SAVING TRAINING VIDEO...")
        sys.exit(0)
    #=======================================================================
    


    #___Stitches images and exports video
    def save_video(self,name):
        
        # stitch images together and create a video
        out = cv2.VideoWriter(VIDEO_PATH+name+".mp4",cv2.VideoWriter_fourcc('m','p','4','v'), FRAME_RATE, self.vid_size)
        
        #___export video
        for i in range(len(self.gameVid)):
            out.write(self.gameVid[i])
        out.release()

        print("\n\n>> TRAINING VIDEO SAVED <<\n")
    #=======================================================================



    #___creates image of the game with the updated state and stores in the the array
    def update_board(self,state):
    
        self.gameImg = np.zeros(self.BOARD_SIZE, dtype="uint8") # initiate image with black background
        self._draw_Grid()                                       # draw the board
        
        for i in range(len(state)):
            disc_location = self._disc_state2location(i)        # convert state on the board to location in pixels
            #___set disc colour, 1 red and 2 for yellow___#
            if (state[i] == 1):                           #
                self._draw_disc("red",disc_location)      #
            elif (state[i] == 2):                         #
                self._draw_disc("yellow",disc_location)   #

        # up to this point the image is created with board upside down due to Y-axis starting from top
        self.gameImgFlipped = cv2.flip(self.gameImg,0)

        #---------------------------------------------#
        self.gameVid.append(self.gameImgFlipped) # store / concatinate image in the list that is used to create the video at the end

    #=======================================================================



    #___called when the game is finished, it creates a result page, prints the winning colour and episode number  
    def gameFinished(self,episode,whoWonTheGame):
        #___image and text properties
        outroImg = np.zeros(self.BOARD_SIZE, dtype="uint8")                     # initiate image with black background
        outroImgLoc = (self.BOX_CENTER, self.BOX_SIZE)                          # text location
        outroImgFont = cv2.FONT_HERSHEY_SIMPLEX                                 # text font
        outroImgFontSize = (outroImg.shape[0] * outroImg.shape[1]) / (1000*130) # text size
        outroImgFontColour = (255,255,255)                                      # text colour
        outroImgLineType = 1                                                    # text thickness

        #___create result page
        if (episode == 1):  
            if (whoWonTheGame == 2):    # red
                outroImg = cv2.putText(outroImg, f">> RED WON << EPISODE : {str(episode)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 1):  # yellow
                outroImg = cv2.putText(outroImg, f">> YELLOW WON << EPISODE : {str(episode)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 0):  # draw
                outroImg = cv2.putText(outroImg, f">> ITS A DRAW << EPISODE : {str(episode)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
        else:
            if (whoWonTheGame == 2):    # red
                outroImg = cv2.putText(outroImg, f">> RED WON << EPISODE : {str(episode-1)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 1):  # yellow
                outroImg = cv2.putText(outroImg, f">> YELLOW WON << EPISODE : {str(episode-1)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
            elif (whoWonTheGame == 0):  # draw
                outroImg = cv2.putText(outroImg, f">> ITS A DRAW << EPISODE : {str(episode-1)}", outroImgLoc, outroImgFont, outroImgFontSize, outroImgFontColour, outroImgLineType)
        
        #___concatinate result page 
        for i in range(3):
            self.gameVid.append(outroImg)
    #=======================================================================



    #___draws discs on the board, params : (discColour - "red" or "yellow") , (discLocation is centre of the circle : (x,y)) 
    def _draw_disc(self,discColour,discLocation):
        disc_colour = (0,0,0)                       # initiate disc colour as black
        if (discColour == "red"):                   # enter if red
            disc_colour = self.RED_DISC_COLOUR      
        elif (discColour == "yellow"):              # enter if yellow
            disc_colour = self.YELLOW_DISC_COLOUR
        
        #___draw discc
        cv2.circle(self.gameImg,discLocation,self.DISC_RADIUS,disc_colour,-1)
    #=======================================================================



    #___converts state (connect 4 board states) to location on image in pixels (x,y)
    def _disc_state2location(self,stateNum):
    
        row, col = self.state_2_row_col(stateNum)   # convert state (1-42) to coordinate (row,col)
        
        if (stateNum <= 0):             # state = 0
            loc = (self.BOX_CENTER, self.BOX_CENTER)

        elif (stateNum < self.COLs):    # bottom row (not state 0)
            loc = (self.BOX_CENTER+(stateNum*self.BOX_SIZE), self.BOX_CENTER)
        
        elif (col == 0):                # left most column (not bottom row or state 0)
            loc = (self.BOX_CENTER, self.BOX_CENTER+(row*self.BOX_SIZE))
        
        elif (col > 0):                 # columns 2 to 7 (not bottom row or state 0)
            loc = (self.BOX_CENTER+(col*self.BOX_SIZE), self.BOX_CENTER+(row*self.BOX_SIZE))
        
        else:                           # something went wrong
            loc = (0,0)
            print("disk location unknown")        
        
        return loc                      
    #=======================================================================



    #___convert coordinates to states, index for state, rows and cols start from 0
    def row_col_2_state(self, rows, cols):
        return (rows * self.COLs) + cols
    #=======================================================================



    #___convert states to coordinates, index for state, rows and cols start from 0
    def state_2_row_col(self, state):
        col = state % self.COLs
        row = (state - col) / 7
        return int(row) , int(col)
    #=======================================================================





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