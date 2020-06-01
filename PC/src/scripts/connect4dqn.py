#!/usr/bin/env python3
 
################################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN       ##
################################################
'''
Deep Q Network Class and Thread Routine

Provides a Deep Q Network script which grabs a trained model and uses it to generate actions.
This script is only intended to be used with the rest of RoboCon software to play games and not for training.
It is developed using Keras with Tensorflow and is trained seperately. Please see DQN_Training directory.

This file uses Connect4 and Connect4Env classes for the game API.

Note: This script was developed with help from the tutorials on https://pythonprogramming.net/

'''
################################################
##                    PC                      ##
################################################

import os
import random
import sys
import time
import datetime
from collections import deque
from copy import deepcopy
import queue

from keras import initializers 
from keras.models import load_model
import keras.backend.tensorflow_backend as backend
import numpy as np
from connect4env import Connect4Env
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import (Activation, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from tqdm import tqdm
from sklearn import preprocessing


# print(sys.path)          # if include path error occurs, uncomment this to get the path 
SHUTDOWN_RPI_CODE = 147258 # This number is used through out the RoboCon software which tells each script to shutdown     
##############################
TRAINING = False            ## 
EPISODES = 50000            ## Number of episodes for the training session
#---------------------------##
HIDDEN_LAYERS = 100         ## Number of neurons in a layer (layer width)
LEARNING_RATE = 0.001       ## DQN learning rate, (how fast to learn)
REWARD_PLUS_VAL = 1         ## Reward added for every counter move learnt by DQN
DISCOUNT = 0.99             ## Gamma 
DELAY_PER_EPISODE = 0.01    ## used to slow down training to protect the PC 
TRAIN_OPPONENT = False      #====# If True DQN's opponent will use DQN's Q-table for decisions 
OPPONENT_DIFFICULTY_HARD = False # If true opponent will use counter moves

#___Load previously trained model__________________________________________________________________________________________________# 
MODEL_PATH = "./dqn_models/" # Path to where the DQN models are saved                                                              #
LOAD_MODEL = MODEL_PATH + "2020-04-21" + "/" + "2020-04-21 23:10:51.644057__Connect4_DQN__ 100.00max_  17.50avg_-200.00min.model"  #
#----------------------------------------------------------------------------------------------------------------------------------#



REPLAY_MEMORY_SIZE = 100000     # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 2000   # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 200            # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5         # Terminal states (end of episodes)
MODEL_NAME = 'Connect4_DQN'     # used for saving models, logs and videos
MIN_AVG_REWARD = -200           # used while saving logs
AGGREGATE_STATS_EVERY = 200     # number of episodes to skip between saving logs and models
ep_rewards = [-200]             # stores rewards for all episodes


# Exploration settings  (equation for decay : epsilon * EPSILON_DECAY ^ EPSILON_DECAY_POWER)
epsilon = 1 
EPSILON_DECAY = 0.099975
MIN_EPSILON = 0.001


SHOW_PREVIEW = False # if true display GUI

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

# Memory fraction, used mostly when training multiple agents
# MEMORY_FRACTION = 0.20
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.isdir('models'):
    os.makedirs('models')

#####################################################################
#####################################################################
# NOTE:
# The ModifiedTensorBoard class was taken from 
# https://pythonprogramming.net/. I have used this
# without making any adjustments to it and therefore
# has no input from me at all.

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

#####################################################################
#####################################################################




#__instance of game environment class__#
env = Connect4Env()                    #
#--------------------------------------#



##################################################################################################################
# ------------------------------------ > > > > DEEP Q NETWORK CLASS < < < < ------------------------------------ # 
##################################################################################################################
class Deep_Q_Network:

    def __init__(self):

        # Main model
        self.model = self.create_model()
        # model.initializers.he_normal(seed=None)

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, datetime.datetime.now()))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    #=============================================================================



    # Loads a pre-trained data for a given path and model name 
    # stored in TRAINED_MODEL in the beginning if this file
    def load_trained_model(self):
        print("Loading {}".format(LOAD_MODEL))
        model = load_model(LOAD_MODEL)
        print("Model {} loaded!".format(LOAD_MODEL))
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ USING LOADED MODEL ^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        return model # return model
    #=============================================================================



    # Create NN model or if required load existing one
    def create_model(self):
        if (LOAD_MODEL is not None): # if LOAD_MODEL (top of this file) is not set to None, a pre-trained model is used
            model = self.load_trained_model()
            
        else: # if pre-trained model is not required create a new one
            
            print("FIRING NEW MODEL ---->>")
            
            model = Sequential()    
            
            # Dense (fully connected layer), input size from Connect4Env class, HE initialised weights
            model.add(Dense(HIDDEN_LAYERS, input_shape=(env.get_dqn_input_size(),),kernel_initializer='he_normal'))
            model.add(Activation('relu'))   # output activation function used ReLU (Rectified Linear Unit)
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            # Dense (fully connected layer), output size from Connect4Env class, HE initialised weights and linear output 
            model.add(Dense(env.get_dqn_output_size(), kernel_initializer='he_normal', activation='linear'))  
            model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        
        return model # return pre-trained or new model
    #=============================================================================



    # Adds step's data to a memory replay (memory pool) array
    # (current state, action, reward, new state, done, valid)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    #=============================================================================



    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        
        # standardize the data attributes
        current_states = preprocessing.scale(current_states)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        
        # standardize the data attributes
        new_current_states = preprocessing.scale(new_current_states)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)


        # Fit on all samples (back propagation) as one batch, log only on terminal state
        self.model.fit(np.array(preprocessing.scale(X)), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    #=============================================================================



    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state): 
        # # standardize the data attributes
        tempArray = preprocessing.scale(state)
        # return action for the current state
        return self.model.predict(np.array(tempArray).reshape(-1, *state.shape))[0]
    #=============================================================================


    # Prints connect 4 game board on terminal
    def print_board(self,board):
        b = board
        print("\n")
        idx = 0

        rows = env.ROWs
        cols = env.COLs

        
        for r in range((rows - 1),-1,-1):
            for c in range(0,cols):
                idx = c + (r * cols)

                if (board[idx] <= 0):
                    print("  .  ", end='')
                elif (board[idx] == 1):
                    print(" (R) ", end='')        
                elif (board[idx] >= 2):
                    print(" (Y) ", end='')
            print("")
        print("")
    #=============================================================================



#######################################################################
##                              Training                             ##
#######################################################################
# Connect 4 Deep Q Network Player Thread Routine
def connect4player(visionBoardQue,instructionQue,data2rpiQue,rpiTaskCompleteQue):
    agent = Deep_Q_Network()

    if (TRAINING):
        # Iterate over episodes
        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            time.sleep(DELAY_PER_EPISODE)

            # Restarting episode - reset variables and game environment
            episode_reward = 0
            step = 1
            current_state = env.new_game()
            done = False
            reward_plus = 0

            # randomise first turn
            switcher = np.random.randint(0, 2)
            
            # New Game
            while (not done):
                whoWonTheGame = 3        # reset win status
                switcher = not switcher # switch turn
                
                if (switcher):#----------------------RoboCon----------------------#
                
                    counter_move_flag = False
                    # action, counter_move_flag = env.counter_move(2) # uncomment to use counter moves 
                    
                    if (not counter_move_flag): 
                        
                        if (np.random.random() > epsilon):  # use Q-Table or explore 
                            
                            # Get action from Q table
                            # python passes arrays by reference and we dont want to change current state
                            # hence a deepcopy used to bypass. 
                            cur_state_temp = deepcopy(current_state)        
                            action = np.argmax(agent.get_qs(current_state))
                            current_state = deepcopy(cur_state_temp)

                        else:
                            # Get random action
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                        
                        reward_plus = 0 

                    elif (counter_move_flag):   # if counter move is on and available
                        reward_plus = REWARD_PLUS_VAL
                    
                    # use action generated to make a move and get new_state, reward,
                    # whether the game is finished and whether the generated action was valid
                    valid = False
                    new_state, reward, done, valid = env.step(action)
                    
                    # if the generated action was invalid generate a random one
                    while (not valid):
                        reward_plus = 0
                        action = np.random.randint(1, env.get_dqn_output_size()+1)
                        new_state, reward, done, valid = env.step(action)
                    valid = False

                    # Turns GUI on
                    if (SHOW_PREVIEW):
                        env.render()
                        time.sleep(0.5)
                        if (done):
                            input()  

                elif (not switcher):#----------------------Opponent----------------------#    
                    # if true the opponent will use Q-Table for decisions
                    if (TRAIN_OPPONENT):
                        if (OPPONENT_DIFFICULTY_HARD):  # if true opponent will use counter moves
                            counter_move_flag = False   
                            action, counter_move_flag = env.counter_move(1) # if counter move is available it will return true and counter action
                        else:
                            counter_move_flag = False

                        # if counter move was not used or found use Q-Table or 
                        # random move depending on epsilon decay
                        if (not counter_move_flag): 
                            if (np.random.random() > epsilon):
                                # Get action from Q-table
                                cur_state_temp = deepcopy(current_state)
                                action = np.argmax(agent.get_qs(current_state))
                                current_state = deepcopy(cur_state_temp)
                                
                            else:
                                # Get random action
                                action = np.random.randint(1, env.get_dqn_output_size()+1)

                        # take action 
                        valid = False
                        new_state, reward, done, valid = env.opponent_step(action)
                        
                        # if generated action was invalid generate and use a random action 
                        while (not valid):
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                            new_state, reward, done, valid = env.opponent_step(action)
                        valid = False
                    
                    # if opponent is not allowed to use Q-Table
                    elif (not TRAIN_OPPONENT):

                        # if true opponent will use counter moves
                        if (OPPONENT_DIFFICULTY_HARD):  
                            counter_move_flag = False
                            action, counter_move_flag = env.counter_move(1) # if counter move is available it will return true and counter action
                            
                            # if counter move was not used or found use random action
                            if (not counter_move_flag): 
                                action = np.random.randint(1, env.get_dqn_output_size()+1)

                            # take action
                            valid = False
                            new_state, reward, done, valid = env.opponent_step(action)
                        
                        # if generated action was invalid generate and use a random action 
                        while (not valid):
                            valid = False
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                            new_state, reward, done, valid = env.opponent_step(action)
                    valid = False
                    

                    # If true displays GUI
                    if (SHOW_PREVIEW):
                        env.render()
                        time.sleep(0.5)
                        if (done):
                            input()


                # If the game is finished, check who won the game or if it was a draw
                if (done):
                    if(env.check_if_red_has_won()):
                        # print(f"{episode_number} --->>> RED WINS <<<---")
                        whoWonTheGame = 2
                    elif(env.check_if_yellow_has_won()):
                        # print(f"{episode_number} --->>> YELLOW WINS <<<---")
                        whoWonTheGame = 1
                    elif(env.check_if_draw()):
                        # print(f"{episode_number} --->>> ITS A DRAW <<<---")
                        whoWonTheGame = 0
                
                action = action - 1 # shift action scale e.g. for connect 4 shift from 1-7 to 0-6
    
                ################################################################
                # count reward
                episode_reward += reward + reward_plus
                            
                # Every step we update replay memory and train main network
                # update memory:
                memBlock = (deepcopy(current_state), action, reward+reward_plus, deepcopy(new_state), done, env.check_if_yellow_has_won())
                agent.update_replay_memory(memBlock)
                # Train network:
                agent.train(done, step)
                
                # Transform new continous state to new discrete state
                current_state = deepcopy(new_state)
                
                # update steps
                step += 1

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if average_reward >= MIN_AVG_REWARD:
                    agent.model.save("models/{}__{}__{:7.2f}max_{:7.2f}avg_{:7.2f}min.model"
                    .format(datetime.datetime.now(),MODEL_NAME,max_reward,average_reward,min_reward))

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            
        
        
        #######################################################################
        ##                              Testing                              ##
        #######################################################################
        
    elif (not TRAINING):
        FINISHED_PLAYING = False
        externalBoard = np.zeros(42,int) # represents board states from vision
        internalBoard = np.zeros(42,int) # represents board states from Connect4Env

        # main loop for the thread routine
        while (not FINISHED_PLAYING):

            # block and wait for instructions from client (client.py) 
            client_instruction = instructionQue.get()
            if (client_instruction == 9):   # instruction 9 is for shutdown
                FINISHED_PLAYING = True
                return
            
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.new_game()

            # Reset flag and start iterating until episode ends
            done = False
            switcher = np.random.randint(0, 100)

            # New Game
            while (not done):
                # reset variables 
                done = False
                valid = False
                action = 0
                
                # display board
                env.render()

                if (switcher >= 50): #----------------------RoboCon----------------------#
                    counter_move_flag = False
                    # action, counter_move_flag = env.counter_move(2)

                    # if counter move is not allowed or found
                    # always use Q-table to generate action
                    if (not counter_move_flag):  
                        action = np.argmax(agent.get_qs(current_state)) # get action from Q-table
                    
                    # limit action range, not really required as validity checked by step function
                    if (action < 1):
                        action = 1
                    elif (action > env.COLs):
                        action = env.COLs

                    # use action generated to make a move and get new_state, reward,
                    # whether the game is finished and whether the generated action was valid
                    new_state, reward, done, valid = env.step(action)
                    
                    # if the generated action was invalid generate a random one and use it
                    while (not valid):
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                            new_state, reward, done, valid = env.step(action)
    
                    valid = False
                    
                    
                    
                    internalBoard = list(env.get_board()) # get internal board state
                    
                    #___DEBUG________________________________________#
                    # print internal and external boards on terminal #
                    # expecting external board to have 1 extra state #
                    # from vision capture of last move               #-------------------#
                    #print("Move Number {} - RoboCon_FM Action: {}".format(step,action)) #
                    #print("External Board : ")                      #-------------------#
                    #agent.print_board(externalBoard)                #
                    #print("Internal Board : ")                      #
                    #agent.print_board(internalBoard)                #
                    #------------------------------------------------#

                    env.render() # display board

                    #___RPI CONTROL__________##
                    data2rpiQue.put(action)  ##
                    rpiTaskCompleteQue.get() ##
                    ###########################

                    #----------------------Opponent----------------------#
                    while (not valid and not done): # stay in loop while action is not valid and game has not finished  
                        
                        # block and wait for external board state, available after
                        # human player has played their move
                        exBoardBuff = visionBoardQue.get()
                        
                        # check the board state is iteratable
                        while (not any(exBoardBuff)):
                            exBoardBuff = visionBoardQue.get()

                        # create numpy array of type integer of external board state
                        externalBoard = np.array(exBoardBuff,int)
                        # get internal board state
                        internalBoard = list(env.get_board())

                        # block and wait for instructions from client thread (client.py)
                        done_instruction = instructionQue.get()
                        if ( done_instruction == 2): # instruction 2 means cancel / finish game 
                            done = True
                            FINISHED_PLAYING = False
                            return

                        # get boolean array, expecting external board to have 1 extra state
                        # therefore expected array should only have one 0
                        equal_board = externalBoard == internalBoard
                        
                        # if there is no difference then camera did not pick up human players move 
                        if (equal_board.all()):
                            raise Exception("Error: players move not found, (connect4dqn.py -->> connect4player -->> Testing -->> Opponents turn)")
                        
                        # extract human player's move
                        board_i_ = 0
                        breakFlag = False
                        # loop through all states of the board
                        while ((board_i_ < len(internalBoard)) and not breakFlag):
                            
                            # enter when a external board is not equal to internal board 
                            if (externalBoard[board_i_] != internalBoard[board_i_]):
                                # check the state that is different is equal to 2 on the external board 
                                # and the internal board has an empty slot in that location 
                                if((externalBoard[board_i_] == 2) and (internalBoard[board_i_] == 0)):
                                    # extract column number of the state in question
                                    _ , action = env.state_2_row_col(board_i_)
                                    # the extracted column becomes action (shift from 0-6 to 1-7) 
                                    action+=1
                                    breakFlag = True # exit loop
                            board_i_+=1 # increment counter

                        #___DEBUG________________________________________#
                        # print internal and external boards on terminal #
                        # expecting external board to have 1 extra state #
                        # from vision capture of last move               #-------------------#
                        #print("Move Number {} - RoboCon_FM Action: {}".format(step,action)) #
                        #print("External Board : ")                      #-------------------#
                        #agent.print_board(externalBoard)                #
                        #print("Internal Board : ")                      #
                        #agent.print_board(internalBoard)                #
                        #------------------------------------------------#

                        # use extracted action to make a move and get new_state, reward,
                        # whether the game is finished and whether the generated action was valid
                        new_state, reward, done, valid = env.opponent_step(action)
                        
                        if (not valid):
                            print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                    
                    env.render() # display updated board GUI
            
                elif (switcher < 50):
                    #----------------------Opponent----------------------#
                    valid = False
                    while (not valid):
                        # block and wait for external board state, available after
                        # human player has played their move
                        exBoardBuff = visionBoardQue.get()
                        
                        # check the board state is iteratable
                        while (not any(exBoardBuff)):
                            exBoardBuff = visionBoardQue.get()

                        # create numpy array of type integer of external board state
                        externalBoard = np.array(exBoardBuff,int)
                        # get internal board state
                        internalBoard = list(env.get_board())


                        # block and wait for instructions from client thread (client.py)
                        done_instruction = instructionQue.get()
                        if ( done_instruction == 2): # instruction 2 means cancel / finish game 
                            done = True
                            FINISHED_PLAYING = False
                            return

                        # get boolean array, expecting external board to have 1 extra state
                        # therefore expected array should only have one 0
                        equal_board = externalBoard == internalBoard
                        
                        # if there is no difference then camera did not pick up human players move 
                        if (equal_board.all()):
                            raise Exception("Error: players move not found, (connect4dqn.py -->> connect4player -->> Testing -->> Opponents turn)")
                        
                        # extract human player's move
                        board_i_ = 0
                        breakFlag = False
                        # loop through all states of the board
                        while ((board_i_ < len(internalBoard)) and not breakFlag):
                            
                            # enter when a external board is not equal to internal board 
                            if (externalBoard[board_i_] != internalBoard[board_i_]):
                                # check the state that is different is equal to 2 on the external board 
                                # and the internal board has an empty slot in that location 
                                if((externalBoard[board_i_] == 2) and (internalBoard[board_i_] == 0)):
                                    # extract column number of the state in question
                                    _ , action = env.state_2_row_col(board_i_)
                                    # the extracted column becomes action (shift from 0-6 to 1-7) 
                                    action+=1
                                    breakFlag = True # exit loop
                            board_i_+=1 # increment counter

                    
                        internalBoard = list(env.get_board()) # get internal board state
                    
                        #___DEBUG________________________________________#
                        # print internal and external boards on terminal #
                        # expecting external board to have 1 extra state #
                        # from vision capture of last move               #-------------------#
                        #print("Move Number {} - RoboCon_FM Action: {}".format(step,action)) #
                        #print("External Board : ")                      #-------------------#
                        #agent.print_board(externalBoard)                #
                        #print("Internal Board : ")                      #
                        #agent.print_board(internalBoard)                #
                        #------------------------------------------------#
    
                        # use extracted action to make a move and get new_state, reward,
                        # whether the game is finished and whether the generated action was valid
                        new_state, reward, done, valid = env.opponent_step(action)
                        
                        if (not valid):
                            print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                    
                    env.render() # display updated board GUI

                    #----------------------RoboCon----------------------#
                    if(not done):
                        counter_move_flag = False
                        # action, counter_move_flag = env.counter_move(2)

                        # if counter move is not allowed or found
                        # always use Q-table to generate action
                        if (not counter_move_flag):  
                            action = np.argmax(agent.get_qs(current_state)) # get action from Q-table
                        
                        # limit action range, not really required as validity checked by step function
                        if (action < 1):
                            action = 1
                        elif (action > env.COLs):
                            action = env.COLs

                        # use action generated to make a move and get new_state, reward,
                        # whether the game is finished and whether the generated action was valid
                        new_state, reward, done, valid = env.step(action)
                        
                        # if the generated action was invalid generate a random one and use it
                        while (not valid):
                                action = np.random.randint(1, env.get_dqn_output_size()+1)
                                new_state, reward, done, valid = env.step(action)
        
                        valid = False
                        
                        #___DEBUG________________________________________#
                        # print internal and external boards on terminal #
                        # expecting external board to have 1 extra state #
                        # from vision capture of last move               #-------------------#
                        #print("Move Number {} - RoboCon_FM Action: {}".format(step,action)) #
                        #print("External Board : ")                      #-------------------#
                        #agent.print_board(externalBoard)                #
                        #print("Internal Board : ")                      #
                        #agent.print_board(internalBoard)                #
                        #------------------------------------------------#

                        #___RPI CONTROL__________##
                        data2rpiQue.put(action)  ##
                        rpiTaskCompleteQue.get() ##
                        ###########################
                    
                    env.render() # display updated board GUI

                # If the game is finished, check who won the game or if it was a draw      
                if (done):
                    if(env.check_if_red_has_won()):
                        print("--->>> RED WINS <<<---")
                    elif(env.check_if_yellow_has_won()):
                        print("--->>> YELLOW WINS <<<---")
                    elif(env.check_if_draw()):
                        print("--->>> ITS A DRAW <<<---")
                
                action = action - 1 # shift action scale e.g. for connect 4 shift from 1-7 to 0-6
                
                ################################################################
                
                # count reward
                episode_reward += reward

                # display GUI board
                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    env.render()

                # Every step we update replay memory and train main network
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step)

                # Transform new continous state to new discrete state
                current_state = new_state
                
                # update steps
                step += 1


            # check if user would like to play another game
            play_more = ""
            while (not FINISHED_PLAYING):
                play_more = input("WOULD LIKE TO PLAY ANOTHER GAME? (y/n): ")
                print("WOULD LIKE TO PLAY ANOTHER GAME? (y/n): {}".format(play_more))

                if (play_more == "n"):
                    FINISHED_PLAYING = True
                    break
                elif (play_more == "y"):
                    FINISHED_PLAYING = False
                    break

    #___SHUTDOWN RPI SOCKETS___________##
    data2rpiQue.put(SHUTDOWN_RPI_CODE) ##
    rpiTaskCompleteQue.get()           ##
    #####################################