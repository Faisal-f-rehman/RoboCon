#!/usr/bin/env python3

################################################
##      AUTHOR : FAISAL FAZAL-UR-REHMAN       ##
################################################
'''
Deep Q Network Class and Training & Testing sequence 

Provides a Deep Q Network Training script. Developed using Keras with Tensorflow and provides logs for analysis with Tensorboard.
It also provides a connect 4 training video, if required, which creates and saves a training video in the background without 
effecting the training speed.

This file uses Connect4, Connect4Env and GameRecord classes for the game API and for the game recorder.

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
import signal
import cv2

from keras import initializers 
from keras.models import load_model
import keras.backend.tensorflow_backend as backend
import numpy as np
from connect4env import Connect4Env
from gameRecord import GameRecord

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import (Activation, Conv1D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from tqdm import tqdm
from sklearn import preprocessing


# print(sys.path) # if include path error occurs, uncomment this to get the path 

#===========================#
# Training Controls         #
TRAINING = True             # False if testing, True if training             
EPISODES = 40000            # Number of episodes for the training session
RECORD_TRAINING = True      # True if a video of the training is required
RECORD_EVERY_EPISODES = 50  # Number of games to skip between video recordings
SHOW_PREVIEW = False        # Turns game GUI on during training  
#---------------------------#
# Neural Network Setup      #
HIDDEN_LAYERS = 256         # Number of neurons in a layer (layer width)
LEARNING_RATE = 0.00001     # DQN learning rate, (how fast to learn)
REWARD_PLUS_VAL = 1         # Reward added for every counter move learnt by DQN
DISCOUNT = 0.99             # Gamma 
DELAY_PER_EPISODE = 0.01    # used to slow down training to protect the PC
TRAIN_OPPONENT = False      #====# If True DQN's opponent will use DQN's Q-table for decisions 
OPPONENT_DIFFICULTY_HARD = False # If true opponent will use counter moves
#===========================#====#


#___Load previously trained model_________________________________________________________________________________________# 
LOAD_MODEL = False          # If true, TRAINED_MODEL from the path provided is loaded and used instead of creating one    #
MODEL_PATH = "./use_model/" # Path to where the DQN models are saved                                                      #
TRAINED_MODEL = MODEL_PATH + "2020-04-21/2020-04-21 20:19:37.486389__Connect4_DQN__ 100.00max_-116.25avg_-200.00min.model"#
#-------------------------------------------------------------------------------------------------------------------------#



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
EPSILON_DECAY = 2
EPSILON_DECAY_POWER = -0.03
MIN_EPSILON = 0.001

    

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

# Memory fraction, used mostly when training multiple agents
# MEMORY_FRACTION = 0.20
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


# Create folders
if not os.path.isdir('models'): os.makedirs('models')                           # DQN model folder
if not os.path.isdir(MODEL_PATH+"LAST_RUN"): os.makedirs(MODEL_PATH+"LAST_RUN") # Seperate folder to save DQN model for the last episode of a session 
if not os.path.isdir('TrainingVideos'): os.makedirs('TrainingVideos')           # Folder to save training videos

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

        # Target network
        self.target_model = self.create_model()
        # copies weights from main model
        self.target_model.set_weights(self.model.get_weights()) 

        # Memory Pool (an array with last n steps for training) 
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
                
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, datetime.datetime.now()))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    #=============================================================================



    # Loads a pre-trained data for a given path and model name 
    # stored in TRAINED_MODEL in the beginning if this file
    def load_trained_model(self):
        
        print("\n\nLoading model...\n{}\n".format(TRAINED_MODEL))
        
        model = load_model(TRAINED_MODEL) # loads model
        
        print("\n\nModel {} loaded!\n".format(TRAINED_MODEL))
        print("\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^ USING LOADED MODEL ^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")
        
        return model # return model
    #=============================================================================



    # Create NN model or if required load existing one
    def create_model(self):
        if (LOAD_MODEL): # if LOAD_MODEL (top of this file) is set True, a pre-trained model is used
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
    def update_replay_memory(self, transition,step):
        self.replay_memory.append(transition)
    #=============================================================================




    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)      
        random.shuffle(minibatch)

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

        # time.sleep(100)
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done, OppWin) in enumerate(minibatch):

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
        if ((self.target_update_counter > UPDATE_TARGET_EVERY) or (OppWin)):
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    #=============================================================================



    # Queries main network for Q values given current state (environment state)
    def get_qs(self, state):         
        # standardize data attributes
        tempArray = preprocessing.scale(state)
        # return action for the current state
        return self.model.predict(np.array(tempArray).reshape(-1, *state.shape))[0]
    #=============================================================================



#######################################################################
##                              Training                             ##
#######################################################################

agent = Deep_Q_Network()

if (TRAINING):
    episode_number = 0      # tracks episode number
    whoWonTheGame = 3       # indicates the winner
    recVid = GameRecord()   # training video creater class

    # If the training is cancelled with ctrl+c by user, this triggers
    # the handler function in GameRecord class
    signal.signal(signal.SIGINT, recVid.signal_handler)
    
    store_training_video_flag = True # not constant, used for skipping games as recording every game doesn't make much sense

    # Training begins, tqdm creates progress bar and information on terminal during training
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'): 

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # slows training down for weaker systems
        time.sleep(DELAY_PER_EPISODE)   

        # Restarting episode - reset variables and game environment
        episode_reward = 0
        step = 1
        current_state = env.new_game()
        new_state = []
        done = False
        reward_plus = 0

        # randomise first turn
        switcher = np.random.randint(0, 2)

        # increment episode number       
        episode_number+=1
        
        # New Game
        while (not done):
            whoWonTheGame = 3        # reset win status
            switcher = not switcher  # switch turn
        
            if (switcher): #----------------------RoboCon----------------------#
                
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

                # If RECORD_TRAINING video flag was set TRUE and required number
                # of games have been skipped record current game
                if (store_training_video_flag and RECORD_TRAINING):
                    recState = deepcopy(new_state)
                    recVid.update_board(new_state)
                    new_state = deepcopy(recState)
            

            elif (not switcher): #----------------------Opponent----------------------#    
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


                # If RECORD_TRAINING video flag was set TRUE and required number
                # of games have been skipped record current game
                if (store_training_video_flag and RECORD_TRAINING):
                    recState = deepcopy(new_state)
                    recVid.update_board(new_state)
                    new_state = deepcopy(recState)

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
            agent.update_replay_memory(memBlock,step)
            # Train network:
            agent.train(done, step)
            
            # Transform new continous state to new discrete state
            current_state = deepcopy(new_state)
            
            # update steps
            step += 1


        # add a result image to the image array at the end of each recorded game
        if (store_training_video_flag and RECORD_TRAINING):
            recVid.gameFinished(episode_number,whoWonTheGame)
        store_training_video_flag = False
        
        # permits recording a game after every N games, where N is defined by RECORD_EVERY_EPISODES
        if (not episode % RECORD_EVERY_EPISODES):
            store_training_video_flag = True


        #___SAVE MODELS DURING TRAINING
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if (not episode % AGGREGATE_STATS_EVERY) or (episode == 1):            
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if (average_reward >= MIN_AVG_REWARD):
                agent.model.save("models/{}__{}__{:7.2f}max_{:7.2f}avg_{:7.2f}min.model"
                .format(datetime.datetime.now(),MODEL_NAME,max_reward,average_reward,min_reward))
        
        #___SAVE THE LAST MODEL IN A DIFFERENT FOLDER
        if (episode >= EPISODES):
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if average_reward >= MIN_AVG_REWARD:
                agent.model.save(MODEL_PATH + "LAST_RUN/{}__{}__{:7.2f}max_{:7.2f}avg_{:7.2f}min.model"
                .format(datetime.datetime.now(),MODEL_NAME,max_reward,average_reward,min_reward))

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY**EPSILON_DECAY_POWER
            epsilon = max(MIN_EPSILON, epsilon)

    # After the training session, stitch all images and export training video with a given path and name   
    recVid.save_video("{}__{}__{:7.2f}max_{:7.2f}avg_{:7.2f}min"
                .format(datetime.datetime.now(),MODEL_NAME,max_reward,average_reward,min_reward))
    
    
    
    #######################################################################
    ##                              Testing                              ##
    #######################################################################
    
elif (not TRAINING):
    FINISHED_PLAYING = False
    new_state = 1

    # Iterate over episodes
    while (not FINISHED_PLAYING):
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.new_game()

        # Reset flag and start iterating until episode ends
        done = False
        switcher = np.random.randint(0, 2)

        while (not done):
            
            env.render()
            if (switcher): # RoboCon plays first turn
                #----------------------RoboCon----------------------#
                #--------SEE TRAINING SECTION FOR COMMENTS----------#

                counter_move_flag = False
                # action, counter_move_flag = env.counter_move(2)

                if (not counter_move_flag):
                    cur_state_temp = deepcopy(current_state)
                    action = np.argmax(agent.get_qs(current_state))
            
                valid = False
                new_state, reward, done, valid = env.step(action)
                
                while (not valid):
                        action = np.random.randint(1, env.get_dqn_output_size()+1)
                        new_state, reward, done, valid = env.step(action)
                valid = False
                current_state = deepcopy(cur_state_temp)
                

                env.render()
                #----------------------Opponent----------------------#
                while (not valid and not done):  # stay in loop while action is not valid and game has not finished  
                    while (True):
                        try:
                            # get action from user
                            action = int(input("User's turn, enter move between 1 to 7 or enter 0 to quit the game : "))
                            
                            # limit action within range
                            if (action == 0):
                                action = 0
                                Exception("Game Over")
                            elif (action > 7):
                                action = 7
                            elif (action < 1):
                                action = 1
                            break

                        except ValueError:  # used to make sure user inputs numbers only that are between 0 and 7
                            print("===>>> Error: Please enter a number between 1 and 7 <<<===")

                    # take action
                    valid = False
                    new_state, reward, done, valid = env.opponent_step(action)

                    if (not valid):
                        print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                
                # display GUI
                env.render()

            elif (not switcher): # Opponent plays first turn
                #----------------------Opponent----------------------#
                valid = False
                while (not valid): # stay in loop while action is not valid
                    while (True):
                        try:
                            # get action from user
                            action = int(input("User's turn, enter move between 1 to 7 or enter 0 to quit the game : "))
                            
                            # limit action within range
                            if (action == 0):
                                action = 0
                                Exception("Game Over")
                            elif (action > 7):
                                action = 7
                            elif (action < 1):
                                action = 1
                            break
                        
                        except ValueError:
                            print("===>>> Error: Please enter a number between 1 and 7 <<<===")
                    
                    # take action
                    valid = False
                    new_state, reward, done, valid = env.opponent_step(action)

                    if (not valid):
                        print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                
                # display GUI
                env.render()

                #----------------------RoboCon----------------------#
                #--------SEE TRAINING SECTION FOR COMMENTS----------#

                if(not done):
                    counter_move_flag = False
                    # action, counter_move_flag = env.counter_move(2)
                    
                    if (not counter_move_flag):
                        cur_state_temp = deepcopy(current_state)
                        action = np.argmax(agent.get_qs(current_state))
                
                    valid = False
                    new_state, reward, done, valid = env.step(action)
                    
                    while (not valid):
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                            new_state, reward, done, valid = env.step(action)
                    valid = False
                    current_state = deepcopy(cur_state_temp)

                # display GUI
                env.render()

            # if the game is finished, check who won the game or if it was a draw
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

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done, False),step)
            agent.train(done, step)
            
            # Transform new continous state to new discrete state
            current_state = deepcopy(new_state)

            # update steps
            step += 1



        # check if user would like to play another game
        play_more = ""
        while (True):
            play_more = input("WOULD LIKE TO RECORD_TRAININGPLAY ANOTHER GAME? (y/n): ")
            print("WOULD LIKE TO PLAY ANOTHER GAME? (y/n): {}".format(play_more))

            if (play_more == "n"):
                FINISHED_PLAYING = True
                break
            elif (play_more == "y"):
                FINISHED_PLAYING = False
                break
