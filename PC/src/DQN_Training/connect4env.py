#!/usr/bin/env python3

################################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN       ##
################################################
'''
This file has two classes:-
- Connect4 Class:
    This class defines the connect 4 game rules, game status, player status and GUI of the game board

- Connect4Env Class:
    This class inherits the Connect4 class and provides a bridge between Deep Q Network and the Connect4 class.
    This class is written similar to the OpenAI Gym Environments, so that it can be used with the Gym Library.    
'''
################################################
##                    PC                      ##
################################################


import numpy as np
import pygame
import time

class Connect4:
    #___constructor
    def __init__(self, rows = 6, cols = 7): # initiate with the size of the board (default : 42 states)
        self._ROWs = rows
        self._COLs = cols
        self._total_n_of_states = self.row_col_2_state(rows-1,cols-1) + 1
    #==================================================================================================



    #___setup new game         
    def start_new_game(self):
        self.reset_game()    
    #==================================================================================================



    #___reset all member variables
    def reset_game(self):
        self._board = np.zeros(self._total_n_of_states, dtype = int)
        self._num_of_rows_used = np.zeros(self._COLs)
        self._red_has_won = False
        self._yellow_has_won = False
        self._draw = False
        self._done = False
        self._total_moves_played = 0
    #==================================================================================================



    #___convert coordinates to states, index for rows and cols start from 0
    def row_col_2_state(self, rows, cols):
        return (rows * self._COLs) + cols
    #==================================================================================================



    #___convert states to coordinates, index for rows and cols start from 0
    def state_2_row_col(self, state):
        col = state % self._COLs
        row = (state - col) / 7
        return int(row) , int(col)
    #==================================================================================================



    #___updates board state against action given, returns false if action given is invalid 
    def state_transition(self,is_red,action): # params : bool is_red (true when red's turn), int action (between 1 and number of columns on the board)
        
        # limit action between 1 and number of columns on the board
        if (action < 1):
            action = 1
        elif (action > self._COLs):
            action = self._COLs

        # create list of states in action column e.g. action 1 / column 0 : [0, 7, 14, 21, 28, 35] 
        slots_in_action_col = [(action - 1 + (self._COLs * r)) for r in range(self._ROWs)]
        
        valid = False
        check_max_row = 0
        state_num = 0
        for r in slots_in_action_col:                   # iterate through all states in action column   
            check_max_row += 1                          # increment counter
            if (check_max_row > self._ROWs):            # reading back, this is not required, to be looked at, at a later date 
                self._num_of_rows_used[action-1] += 1
                break
            elif (not self._board[r]):                  # enter only if the state is empty
                self._board[r] = 1 if is_red else 2     # if this is red's turn mark state as 2 if its yellows turn mark it as 1
                self._num_of_rows_used[action-1] += 1   # mark slot as used , (this tracks used slots in all columns)
                state_num = r                           # holds the state for the current action
                valid = True                            # mark current action to be valid 
                break                                   # break out of for loop
        
        # enter if there was an available slot in the action column  
        if (valid):
            self._total_moves_played += 1       # tracks total moves played in a game
            self.__game_won(is_red,state_num)   # lets member function know the action taken and by whom, the function checks and marks if the game was won
        
        return valid # returns true if the action required was valid
    #==================================================================================================



    #___check if a player has won the game
    def __game_won(self,is_red,state):
        
        state_row, state_col = self.state_2_row_col(state) # converts state to row and column
        #winFlag = False             # will be set to true if the a player has won the game      
        
        self._red_has_won = False
        self._yellow_has_won = False
        self._draw = False
        self._done = False

       
        #___holds horizontal, vertical and diagonal
        #   states, above and below the action state 
        states_to_check_S = []      
        states_to_check_SE1 = []
        states_to_check_SE2 = []
        states_to_check_SE3 = []
        states_to_check_SE4 = []
        states_to_check_SW1 = []
        states_to_check_SW2 = []
        states_to_check_SW3 = []
        states_to_check_SW4 = []
        states_to_check_E1 = []
        states_to_check_E2 = []
        states_to_check_W1 = []
        states_to_check_W2 = []

        #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/#
        #/\ Check if there are any discs in the slots that would make the current move a winning move.     /\#
        #\/ This is a 2-part process. In the first part, all states around the state of the current move,  \/#
        #/\ that would result in a win and are not empty, are extracted (with the board's edges in mind).  /\#
        #\/ In the second part the extracted states are checked for the appropriate colour and quantity    \/#
        #/\ for it to be a winning move.                                                                   /\#
        #\/                                                                                                \/#
        #/\ The locations, NOT including the disc played, that would result in a win are horizontal        /\#
        #\/ states to the left or right, 3 vertical states below the current move state, diagonal states   \/#
        #/\ to the left (above and below current move state), and diagonal states to the right (above and  /\#
        #\/ below current move state).                                                                     \/#
        #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\#    
        
      #___EXTRACT___________________________________________________________________________________________
        #SOUTH (expected size 3)
        states_to_check_S  = [state - (r * self._COLs) for r in range(1,4) if (state - (r * self._COLs)) >= 0]

        for i in range(1,4):
            #SOUTH WEST 1 (expected size 3)
            if ((state_row-i >= 0) and (state_col-i >= 0)):
                states_to_check_SW1.append(self.row_col_2_state(state_row-i,state_col-i))
            

            #SOUTH EAST 1 (expected size 3)
            if ((state_row-i >= 0) and (state_col+i < self._COLs)):
                states_to_check_SE1.append(self.row_col_2_state(state_row-i,state_col+i))
            

        for i in range(4):
            #SOUTH WEST 2 (expected size 3)
            if ((state_row-i+1 >= 0) and (state_col-i+1 >= 0) and 
                (state_row-i+1 <= self._ROWs - 1) and (state_col-i+1 <= self._COLs - 1)):
                states_to_check_SW2.append(self.row_col_2_state(state_row-i+1,state_col-i+1))
            #SOUTH WEST 3 (expected size 3)
            if ((state_row-i+2 >= 0) and (state_col-i+2 >= 0) and 
                (state_row-i+2 <= self._ROWs - 1) and (state_col-i+2 <= self._COLs - 1)):
                states_to_check_SW3.append(self.row_col_2_state(state_row-i+2,state_col-i+2))
            #SOUTH WEST 2 (expected size 3)
            if ((state_row-i+3 >= 0) and (state_col-i+3 >= 0) and 
                (state_row-i+3 <= self._ROWs - 1) and (state_col-i+3 <= self._COLs - 1)):
                states_to_check_SW4.append(self.row_col_2_state(state_row-i+3,state_col-i+3))
            
            #SOUTH EAST 2 (expected size 3)
            if ((state_row-i+1 >= 0) and (state_col+i-1 >= 0) and 
                (state_row-i+1 <= self._ROWs - 1) and (state_col+i-1 <= self._COLs - 1)):
                states_to_check_SE2.append(self.row_col_2_state(state_row-i+1,state_col+i-1))
            #SOUTH EAST 3 (expected size 3)
            if ((state_row-i+2 >= 0) and (state_col+i-2 >= 0) and 
                (state_row-i+2 <= self._ROWs - 1) and (state_col+i-2 <= self._COLs - 1)):
                states_to_check_SE3.append(self.row_col_2_state(state_row-i+2,state_col+i-2))
            #SOUTH EAST 4 (expected size 3)
            if ((state_row-i+3 >= 0) and (state_col+i-3 >= 0) and 
                (state_row-i+3 <= self._ROWs - 1) and (state_col+i-3 <= self._COLs - 1)):
                states_to_check_SE4.append(self.row_col_2_state(state_row-i+3,state_col+i-3))


            #WEST (expected size 4)
            if (state_col - i >= 0):
                states_to_check_W1.append(state - i)
            if ((state_col + 1 - i >= 0) and (state_col + 1 < self._COLs)):
                states_to_check_W2.append(state + 1 - i)
            
            #EAST (expected size 4)
            if (state_col + i < self._COLs):
                states_to_check_E1.append(state + i)
            if ((state_col - 1 + i < self._COLs) and (state_col - 1 >= 0)):
                states_to_check_E2.append(state - 1 + i)
        
        #___DEBUG_______________________________________________#
        # print("action = {}".format(state_col))                #
        # print("south = {}".format(states_to_check_S))         #
        # print("south east1 = {}".format(states_to_check_SE1)) #
        # print("south east2 = {}".format(states_to_check_SE2)) #
        # print("south east3 = {}".format(states_to_check_SE3)) #
        # print("south east4 = {}".format(states_to_check_SE4)) #
        # print("south west1 = {}".format(states_to_check_SW1)) #
        # print("south west2 = {}".format(states_to_check_SW2)) #
        # print("south west3 = {}".format(states_to_check_SW3)) #
        # print("south west4 = {}".format(states_to_check_SW4)) #
        # print("east 1 = {}".format(states_to_check_E1))       #
        # print("east 2 = {}".format(states_to_check_E2))       #
        # print("west 1 = {}".format(states_to_check_W1))       #
        # print("west 2 = {}".format(states_to_check_W2))       #
        #-------------------------------------------------------#



        #___CHECK EXTRACTED STATES__________________________________________
        #check SOUTH
        sum = 0
        if (len(states_to_check_S) >= 3):                   # enter if the number of discs found are enough for a win
            for i in states_to_check_S:                     # iterate through the extracted states
                if (self._board[state] == self._board[i]):  # check if the extracted state colour matches the current move state
                    sum += 1                                # indicates the number of discs found with the correct colour and location
            if (sum >= 3 and is_red):                       # RED player : if this condition is satisfied red player has won the game
                self._red_has_won = True                    
                self._done = True
            elif (sum >= 3 and not is_red):                 # YELLOW player : if this condition is satisfied yellow player has won the game
                self._yellow_has_won = True        
                self._done = True
               
        #check SOUTH EAST 1
        sum = 0
        if (len(states_to_check_SE1) >= 3):
            for i in states_to_check_SE1:
                # print("SE : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 3 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 3 and not is_red):
                self._yellow_has_won = True        
                self._done = True

        #check SOUTH EAST 2
        sum = 0
        if (len(states_to_check_SE2) >= 4):
            for i in states_to_check_SE2:
                # print("SE : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True

        #check SOUTH EAST 3
        sum = 0
        if (len(states_to_check_SE3) >= 4):
            for i in states_to_check_SE3:
                # print("SE : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True
        
        #check SOUTH EAST 4
        sum = 0
        if (len(states_to_check_SE4) >= 4):
            for i in states_to_check_SE4:
                # print("SE : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True


        #check SOUTH WEST 1
        sum = 0
        if (len(states_to_check_SW1) >= 3):
            for i in states_to_check_SW1:
                # print("SW : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 3 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 3 and not is_red):
                self._yellow_has_won = True        
                self._done = True
        
        #check SOUTH WEST 2
        sum = 0
        if (len(states_to_check_SW2) >= 4):
            for i in states_to_check_SW2:
                # print("SW : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True

        #check SOUTH WEST 3
        sum = 0
        if (len(states_to_check_SW3) >= 4):
            for i in states_to_check_SW3:
                # print("SW : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True

        #check SOUTH WEST 4
        sum = 0
        if (len(states_to_check_SW4) >= 4):
            for i in states_to_check_SW4:
                # print("SW : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True


        #check EAST 1
        sum = 0
        if (len(states_to_check_E1) >= 4):
            for i in states_to_check_E1:
                # print("E1 : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True
        
        #check EAST 2
        sum = 0
        if (len(states_to_check_E2) >= 4):
            for i in states_to_check_E2:
                # print("E2 : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True

        #check WEST 1
        sum = 0
        if (len(states_to_check_W1) >= 4):
            for i in states_to_check_W1:
                # print("W1 : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True
            
        #check WEST 2
        sum = 0
        if (len(states_to_check_W2) >= 4):
            for i in states_to_check_W2:
                # print("W2 : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 4 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 4 and not is_red):
                self._yellow_has_won = True        
                self._done = True

        # Finally check if its a draw_________________________________#
        if ((self._total_moves_played >= self._total_n_of_states) and # 
             not self._red_has_won and not self._yellow_has_won):     #
            self._draw = True                                         #
            self._done = True                                         #
        #-------------------------------------------------------------#
    #==================================================================================================




    def check_if_red_has_won(self):
        return self._red_has_won
    #==================================================================================================



    def check_if_yellow_has_won(self):
        return self._yellow_has_won
    #==================================================================================================



    def check_if_draw(self):
        return self._draw
    #==================================================================================================



    def check_if_game_finished(self):
        return self._done
    #==================================================================================================



    def get_board(self):
        return self._board # current board state : 0 empty, 1 red, 2 yellow 
    #==================================================================================================



    def get_max_row_count(self):
        return self._num_of_rows_used # an array of arrays that holds all used states as 1 and empty states as 0 in row column format
    #==================================================================================================


    
    #___Creates a GUI for the connect 4 game board with pygame
    #   this method is set to grab the board state internally
    #   (from member variables) and does not require any params  
    def print_board(self):
        SQUARESIZE = 100                     # square size of one slot
        RADIUS = int(SQUARESIZE/2 - 5)       # radius that defines one slot
        BLUE = (0,0,255)                     # board colour in BGR
        BLACK = (0,0,0)                      # background colour in BGR
        RED = (255,0,0)                      # red disc colour in BGR
        YELLOW = (255,255,0)                 # yellow disc colour in BGR
        width = self._COLs * SQUARESIZE      # board width
        height = (self._ROWs+1) * SQUARESIZE # board height
        size = (width, height)               # board size 

        screen = pygame.display.set_mode(size)  # create image window

        #___Iterate through all columns and rows of the board
        for c in range(self._COLs):
    	    for r in range(self._ROWs):
                #___draw board
                pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                #___draw empty slots
                pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        #___Iterate through all columns and rows of the board
        for c in range(self._COLs):
            for r in range(self._ROWs):
                stateIN = self.row_col_2_state(r,c) # convert coordinates (row,column) to state (1 to 42)	
                if self._board[stateIN] == 1:       # if state is one draw red disc
                    pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif self._board[stateIN] == 2:     # if state is 2 draw yellow disc
                    pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()                     # update display 
    #==================================================================================================



    

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # END OF CONNECT4 CLASS # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



'''                     _____________________________________________________
                        |                                                   |
                        |            CONNECT 4 GAME ENVIRONMENT             |
                        |___________________________________________________|
'''
class Connect4Env(Connect4):            # inherit Connect4 game class
    def __init__(self, rows=6, cols=7): # initiate with rows and columns from the user or the default values and pass it to Connect4 class            
        Connect4.__init__(self, rows=rows, cols=cols)
        self.ROWs = rows    
        self.COLs = cols
        self.total_n_of_states = Connect4.row_col_2_state(self,rows-1,cols-1) + 1
    #==================================================================================================
        
    
    
    #___sets up a new game
    def new_game(self):
        self.reset()                    # resets all member variables
        Connect4.start_new_game(self)   # setup new game
        return Connect4.get_board(self) # return clean board states
    #==================================================================================================
        
    
    
    #___resets member variables and calls reset method of Connect4 class
    def reset(self):
        Connect4.reset_game(self)                   # reset Connect4 class 
        self.row_max_count = np.zeros(self.COLs);   # reset discs counted in all columns
    #==================================================================================================



    #___Play move for the given action
    def step(self, action):
        #_local variables_#
        valid = False     # checked by Connect4 class and is true if the given action is valid
        done = False      # true when the game has finished
        reward = 0        # currently set as follows: 0 during game, -200 lost, 50 draw, 100 win 
        #-----------------#

        # limit action between 1 and number of columns on the board
        if (action < 1):
            action = 1
        elif (action > self.COLs):
            action = self.COLs    

        #___play move, it checks if the move is valid, updates board states,
        #   updates win status and returns true if the move was valid  
        valid = Connect4.state_transition(self, True, action)
        
        # double check if the column had available slots for the move played 
        if (self.row_max_count[action - 1] > self.ROWs): 
            valid = False
        elif(valid):    # if slot was available and the move came as valid from state_transition                     
            self.row_max_count[action - 1] += 1   # function then increment count 
        
        # assign reward if the game has finished 
        if (Connect4.check_if_game_finished(self)):
            done = True    
            try:
                if(Connect4.check_if_red_has_won(self)):
                    reward = 100
                elif(Connect4.check_if_yellow_has_won(self)):
                    reward = -200
                elif (Connect4.check_if_draw(self)):
                    reward = 50
            except ValueError:
                print("Error: Connect4Env::step() Game finished but win or draw flags were not set")
        
        # return new_state, reward, true if the
        # game is finished (done) and true if the
        # move played was valid (valid) 
        return Connect4.get_board(self), reward, done, valid 
        #==================================================================================================


    #___Same as step but with reduced checks and for yellow discs
    def opponent_step(self,action):
        valid = Connect4.state_transition(self, False, action) 
        done = Connect4.check_if_game_finished(self)
        new_state = Connect4.get_board(self)
        reward = 0
        if (done):   
            try:
                if(Connect4.check_if_red_has_won(self)):
                    reward = 100
                elif(Connect4.check_if_yellow_has_won(self)):
                    reward = -200
                elif (Connect4.check_if_draw(self)):
                    reward = 50
            except ValueError:
                print("Error: Connect4Env::step() Game finished but win or draw flags were not set")
        return  new_state, reward, done, valid
    #==================================================================================================

 

 
    def render(self):
        Connect4.print_board(self)
    #==================================================================================================



    #___used by dqn for NN input size
    def get_dqn_input_size(self):
        return self.total_n_of_states
    #==================================================================================================



    #___used by dqn for NN output size
    def get_dqn_output_size(self):
        return self.COLs
    #==================================================================================================



    #___check if the opponent can win in the next move, if so return a counter move (only used by robocon's opponent during training)
    def counter_move(self,counter_against):
        sum = 0
        # get current state of the board  
        board = Connect4.get_board(self)
        
        #><--><--><--><--><--><--><--><--><
        #--->> HORIZONTAL
        board_i = 0 
        for r in range(1,self.ROWs+1):      # iterate through all rows and columns starting at 1 not 0
            for c in range(1,self.COLs+1):
                sum = 0
                if (board[board_i] == counter_against): # check if the board state is the same colour disc as counter_against      
                    for i in range(0,4):    # pan left and right for every opponent's disc found and add 1 to sum for every disc found  
                        _,col = Connect4.state_2_row_col(self,board_i-i)
                        if ((board[board_i-i] == counter_against) and (col >= 0) and (board_i-i >= 0)):
                            sum+=1

                    if (sum >= 3):          # if condition is satisfied, a counter move is available
                        if ((board_i < 41) and (board_i >= 2)):
                            for i in range(0,4):
                                _,col = Connect4.state_2_row_col(self,board_i-i)
                                if ((board[board_i-i] == 0) and (col >= 0) and (board_i-i >= 0)): # find empty slot in the middle
                                    return col+1,True

                            _,col = Connect4.state_2_row_col(self,board_i+1)
                            if ((board[board_i+1] == 0) and (col >= 0) and (board_i+1 <= 41)): # then check the corner slots
                                return col+1,True 
                else:
                    sum = 0
                board_i += 1   
                
                

        #><--><--><--><--><--><--><--><--><
        #--->> VERTICAL
        board_i = 0
        for c in range(1,self.COLs+1):
            sum = 0
            for r in range(1,self.ROWs+1):
                board_i = Connect4.row_col_2_state(self,r-1,c-1)
                if (board[board_i] == counter_against): # check if the board state is the same colour disc as counter_against   
                    sum += 1 
                else:
                    sum = 0
                    
                if (sum >= 3):                          # if condition is satisfied, a counter move is available 
                    if (r < self.ROWs):
                        if (board[board_i+7] == 0):
                            return c, True

        #><--><--><--><--><--><--><--><--><
        #--->> NORTH EAST & NORTH WEST
        board_i = 0
        board_NE_i = 0
        sum = 0
        for r in range(self.ROWs):
            for c in range(self.COLs):
                if (board[board_i] == counter_against):
                    #NORTH EAST
                    sum=0
                    for i in range(0,4):
                        if ((r+i < self.ROWs) and (c+i < self.COLs)):
                            board_NE_i = Connect4.row_col_2_state(self,r+i,c+i)
                            if (board[board_NE_i] == counter_against): # check if the board state is the same colour disc as counter_against 
                                sum+=1

                            if (sum >= 3):          # if condition is satisfied, a counter move is available
                                for i in range(1,4):
                                    board_NE_i = Connect4.row_col_2_state(self,r-1+i,c-1+i)
                                    if (board[board_NE_i] == 0):
                                        return c+1+i, True
                    
                    #NORTH WEST
                    sum=0
                    for i in range(0,4):
                        if ((r+i < self.ROWs) and (c-i >= 0)):
                            board_NE_i = Connect4.row_col_2_state(self,r+i,c-i)
                            if (board[board_NE_i] == counter_against): # check if the board state is the same colour disc as counter_against 
                                sum+=1
                                                        
                            if (sum >= 3):          # if condition is satisfied, a counter move is available
                                for i in range(1,4):        
                                    board_NE_i = Connect4.row_col_2_state(self,r-1+i,c-1-i)
                                    if (board[board_NE_i] == 0):
                                        return c+1-i, True
                    sum = 0

                sum = 0                
                board_i+=1    
                                       
        #><--><--><--><--><--><--><--><--><        
        return 0, False # if this point is reached there were no counter moves available
        #><--><--><--><--><--><--><--><--><
        


################################################################################
##                                 end of class                               ##
################################################################################