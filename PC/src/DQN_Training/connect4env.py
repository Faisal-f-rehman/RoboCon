#!/usr/bin/env python3
import numpy as np
import pygame
import time

class Connect4:
    #___constructor
    def __init__(self, rows = 6, cols = 7):
        self._ROWs = rows
        self._COLs = cols
        self._total_n_of_states = self.row_col_2_state(rows-1,cols-1) + 1

    #___setup new game         
    def start_new_game(self):
        self.reset_game()    

    #___reset all member variables
    def reset_game(self):
        self._board = np.zeros(self._total_n_of_states, dtype = int)
        self._num_of_rows_used = np.zeros(self._COLs)
        self._red_has_won = False
        self._yellow_has_won = False
        self._draw = False
        self._done = False
        self._total_moves_played = 0

    #___convert coordinates to states, index for rows and cols start from 0
    def row_col_2_state(self, rows, cols):
        return (rows * self._COLs) + cols


    #___convert states to coordinates, index for rows and cols start from 0
    def state_2_row_col(self, state):
        col = state % self._COLs
        row = (state - col) / 7
        return int(row) , int(col)


    #___updates _board state against action given, returns false if action given is invalid 
    def state_transition(self,is_red,action):
        
        if (action < 1):
            action = 1
        elif (action > self._COLs):
            action = self._COLs

        slots_in_action_col = [(action - 1 + (self._COLs * r)) for r in range(self._ROWs)]
        
        # print(f"action in state_transition: {action}")
        valid = False
        check_max_row = 0
        state_num = 0
        for r in slots_in_action_col:    
            check_max_row += 1
            if (check_max_row > self._ROWs):
                self._num_of_rows_used[action-1] += 1
                break
            elif (not self._board[r]):
                self._board[r] = 1 if is_red else 2
                self._num_of_rows_used[action-1] += 1                    
                state_num = r
                valid = True
                break

        if (valid):
            self._total_moves_played += 1
            self.__game_won(is_red,state_num)
        
        # print(f"action in state_transition: {action}")
        return valid


    #___check is a player has won the game
    def __game_won(self,is_red,state):
        # state_row, state_col = self.state_2_row_col(state)
        # print("action = {}".format(state_col))
        # if (state <= 0):
        #     return
        self._red_has_won = False
        self._yellow_has_won = False
        self._draw = False
        self._done = False
        # winFlag = False
        state_row, state_col = self.state_2_row_col(state)

        # print("state = {} row = {} col = {}".format(state,state_row,state_col))
        
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
        
        # print("action = {}".format(state_col))
        # print("south = {}".format(states_to_check_S))
        # print("south east1 = {}".format(states_to_check_SE1))
        # print("south east2 = {}".format(states_to_check_SE2))
        # print("south east3 = {}".format(states_to_check_SE3))
        # print("south east4 = {}".format(states_to_check_SE4))
        # print("south west1 = {}".format(states_to_check_SW1))
        # print("south west2 = {}".format(states_to_check_SW2))
        # print("south west3 = {}".format(states_to_check_SW3))
        # print("south west4 = {}".format(states_to_check_SW4))
        # print("east 1 = {}".format(states_to_check_E1))
        # print("east 2 = {}".format(states_to_check_E2))
        # print("west 1 = {}".format(states_to_check_W1))
        # print("west 2 = {}".format(states_to_check_W2))

        #check SOUTH
        sum = 0
        if (len(states_to_check_S) >= 3):
            for i in states_to_check_S:
                # print("S : {}".format(i))
                if (self._board[state] == self._board[i]):
                    sum += 1
            if (sum >= 3 and is_red):
                self._red_has_won = True        
                self._done = True
            elif (sum >= 3 and not is_red):
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

        if ((self._total_moves_played >= self._total_n_of_states) and 
             not self._red_has_won and not self._yellow_has_won):
            self._draw = True
            self._done = True

        


    def check_if_red_has_won(self):
        return self._red_has_won
    
    def check_if_yellow_has_won(self):
        return self._yellow_has_won
    
    def check_if_draw(self):
        return self._draw
    
    def check_if_game_finished(self):
        return self._done

    def get_board(self):
        return self._board
    
    def print_board(self):
        # print("Board : {}".format(self._board))
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)
        BLUE = (0,0,255)
        BLACK = (0,0,0)
        RED = (255,0,0)
        YELLOW = (255,255,0)
        width = self._COLs * SQUARESIZE
        height = (self._ROWs+1) * SQUARESIZE
        size = (width, height)

        screen = pygame.display.set_mode(size)
        
        for c in range(self._COLs):
    	    for r in range(self._ROWs):
                pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
        for c in range(self._COLs):
            for r in range(self._ROWs):
                stateIN = self.row_col_2_state(r,c)	
                if self._board[stateIN] == 1:
                    pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif self._board[stateIN] == 2: 
                    pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()

    def get_max_row_count(self):
        return self._num_of_rows_used


 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Connect4Env(Connect4):
    def __init__(self, rows=6, cols=7):
        Connect4.__init__(self, rows=rows, cols=cols)
        self.ROWs = rows
        self.COLs = cols
        self.total_n_of_states = Connect4.row_col_2_state(self,rows-1,cols-1) + 1
        # print("Total Num of states = {}".format(self.total_n_of_states))
        
    
    def new_game(self):
        self.reset()
        Connect4.start_new_game(self)
        return Connect4.get_board(self)
        
        
    def reset(self):
        Connect4.reset_game(self)
        self.row_max_count = np.zeros(self.COLs);


    def step(self, action):
        valid = False
        done = False
        reward = 0;

        if (action < 1):
            action = 1
        elif (action > self.COLs):
            action = self.COLs    

        valid = Connect4.state_transition(self, True, action)
        
        # print("ROW MAX COUNT : {}".format(Connect4.get_max_row_count(self)))
        if (self.row_max_count[action - 1] > self.ROWs): 
            valid = False
        elif(valid):
            self.row_max_count[action - 1] += 1
        

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

        return Connect4.get_board(self), reward, done, valid


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

 
    def render(self):
        Connect4.print_board(self)
    
 
    def get_dqn_input_size(self):
        return self.total_n_of_states


    def get_dqn_output_size(self):
        return self.COLs


    def counter_move(self,counter_against):
        sum = 0

        board = Connect4.get_board(self)
        
        #><--><--><--><--><--><--><--><--><
        #--->> HORIZONTAL
        board_i = 0 
        for r in range(1,self.ROWs+1):
            for c in range(1,self.COLs+1):
                sum = 0
                if (board[board_i] == counter_against):       
                    for i in range(0,4):
                        _,col = Connect4.state_2_row_col(self,board_i-i)
                        if ((board[board_i-i] == counter_against) and (col >= 0) and (board_i-i >= 0)):
                            sum+=1

                    if (sum >= 3):
                        if ((board_i < 41) and (board_i >= 2)):
                            for i in range(0,4):#check middle ones first
                                _,col = Connect4.state_2_row_col(self,board_i-i)
                                if ((board[board_i-i] == 0) and (col >= 0) and (board_i-i >= 0)):
                                    return col+1,True

                            _,col = Connect4.state_2_row_col(self,board_i+1)
                            if ((board[board_i+1] == 0) and (col >= 0) and (board_i+1 <= 41)):#then check the corner
                                return col+1,True 
                else:
                    sum = 0
                board_i += 1
                        # print("and I was here")
                        # if (board[board_i+1] == 0):
                        #     # print("NOT here")
                        #     _,col = Connect4.state_2_row_col(self,board_i+1)
                        #     return col+1, True
                        # elif (board[board_i-2] == 0):
                        #     # print("OR here")
                        #     print("board_i = {}".format(board_i-2))
                        #     _,col = Connect4.state_2_row_col(self,board_i-2)
                        #     return col+1, True                         
                
                

        #><--><--><--><--><--><--><--><--><
        #--->> VERTICAL
        board_i = 0
        for c in range(1,self.COLs+1):
            sum = 0
            for r in range(1,self.ROWs+1):
                board_i = Connect4.row_col_2_state(self,r-1,c-1)
                if (board[board_i] == counter_against):
                    sum += 1 
                    # print("vertical -> counter against : {}, board_i : {}, board[board_i] : {}, r & c : ({},{}), sum : {}"
                    #         .format(counter_against,board_i,board[board_i],r,c,sum))
                else:
                    sum = 0
                    
                if (sum >= 3):
                    if (r < self.ROWs):
                        # print("board[board_i+7], board_i+7 : {}, {}".format(board[board_i+7],board_i+7))
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
                            if (board[board_NE_i] == counter_against):
                                sum+=1
                                # print("NORTH EAST -> board_NE_i : {}, r+i & c+i : ({},{}), sum : {}"
                                #     .format(board_NE_i,r+i,c+i,sum))

                            if (sum >= 3):
                                for i in range(1,4):
                                    board_NE_i = Connect4.row_col_2_state(self,r-1+i,c-1+i)
                                    if (board[board_NE_i] == 0):
                                        return c+1+i, True
                    
                    #NORTH WEST
                    sum=0
                    for i in range(0,4):
                        if ((r+i < self.ROWs) and (c-i >= 0)):
                            board_NE_i = Connect4.row_col_2_state(self,r+i,c-i)
                            if (board[board_NE_i] == counter_against):
                                sum+=1
                                                        
                            if (sum >= 3):
                                for i in range(1,4):
                                    board_NE_i = Connect4.row_col_2_state(self,r-1+i,c-1-i)
                                    if (board[board_NE_i] == 0):
                                        return c+1-i, True
                    sum = 0

                    
                    # print("horizontal -> counter against : {}, board_i : {}, board[board_i] : {}, r & c : ({},{}), sum : {}"
                    #         .format(counter_against,board_i,board[board_i],r,c,sum))
                sum = 0                
                board_i+=1    
                                       
        #><--><--><--><--><--><--><--><--><        
        return 0, False
        #><--><--><--><--><--><--><--><--><
        


################################################################################
##                                 end of class                               ##
################################################################################


# c4 = Connect4Env()
# for episode in range(10):
#     c4.new_game()
#     done = False

#     while (not done):

#         valid = False
#         while (not valid  and not done):
#             RoboMove = np.random.randint(1, c4.COLs+1)
#             cState,reward,done,valid = c4.step(RoboMove)
            
#             if (not valid):
#                 print("Invalid action RED : {}".format(RoboMove))

#             if (done):
#                 if (done):
#                     if(c4.check_if_red_has_won()):
#                         print("--->>> RED WINS <<<---")
#                     elif(c4.check_if_yellow_has_won()):
#                         print("--->>> YELLOW WINS <<<---")
#                     elif(c4.check_if_draw()):
#                         print("--->>> ITS A DRAW <<<---")
#         valid = False

#         while (not valid and not done):
#             opMove = np.random.randint(1, c4.COLs+1)
#             valid, done = c4.opponent_step(opMove)
            
#             if (not valid):
#                 print("Invalid action YELLOW : {}".format(opMove))

#             if (done):
#                 if(c4.check_if_red_has_won()):
#                     print("--->>> RED WINS <<<---")
#                 elif(c4.check_if_yellow_has_won()):
#                     print("--->>> YELLOW WINS <<<---")
#                 elif(c4.check_if_draw()):
#                     print("--->>> ITS A DRAW <<<---")

#         c4.render()
#         time.sleep(1)
#     # time.sleep(1)