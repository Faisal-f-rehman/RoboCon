#!/usr/bin/env python3
 
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


# print(sys.path)

##############################
TRAINING = True
EPISODES = 40000
RECORD_TRAINING = True
RECORD_EVERY_EPISODES = 50
SHOW_PREVIEW = False
#---------------------------##
HIDDEN_LAYERS = 256       
LEARNING_RATE = 0.00001     ##  0.00001, 0.0000625
REWARD_PLUS_VAL = 1
DISCOUNT = 0.99            
DELAY_PER_EPISODE = 0.01   
TRAIN_OPPONENT = False     
OPPONENT_DIFFICULTY_HARD = False
LOAD_MODEL = False
MODEL_PATH = "./use_model/"
TRAINED_MODEL = MODEL_PATH + "2020-04-21/2020-04-21 20:19:37.486389__Connect4_DQN__ 100.00max_-116.25avg_-200.00min.model" 
#"./models_archive/v3/2020-02-28 04:01:28.617140__Connect4_DQN___290.00max__111.30avg_-180.00min__2020-02-28 04:01:28.617164.model" 
#"./models/2020-02-27 02:44:04.558768__Connect4_DQN___100.00max__-14.75avg_-200.00min__2020-02-27 02:44:04.558788.model"
##############################



REPLAY_MEMORY_SIZE = 100000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 2000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 200  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'Connect4_DQN'
MIN_AVG_REWARD = -200  # For model save
# MEMORY_FRACTION = 0.20




# Exploration settings
epsilon = 1  # not a constant, going to be decayed
# EPSILON_DECAY = 0.099975
EPSILON_DECAY = 2
EPSILON_DECAY_POWER = -0.03
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 200  # episodes


# For stats
ep_rewards = [-200]

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir(MODEL_PATH+"LAST_RUN"):
    os.makedirs(MODEL_PATH+"LAST_RUN")
if not os.path.isdir('TrainingVideos'):
    os.makedirs('TrainingVideos')

#####################################################################
#####################################################################


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

env = Connect4Env()

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
    
    def load_trained_model(self):
        print("Loading {}".format(TRAINED_MODEL))
        model = load_model(TRAINED_MODEL)
        print("Model {} loaded!".format(TRAINED_MODEL))
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ USING LOADED MODEL ^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        return model

    def create_model(self):
        if (LOAD_MODEL):
            model = self.load_trained_model()
            
        else:
            # model.add(Dropout(0.2))
            print("FIRING NEW MODEL ---->>")
            model = Sequential()

            model.add(Dense(HIDDEN_LAYERS, input_shape=(env.get_dqn_input_size(),),kernel_initializer='he_normal'))#kernel_initializer='glorot_uniform'))
            model.add(Activation('relu'))
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            
            # model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            # model.add(Activation('relu'))
            
            # model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            # model.add(Activation('relu'))
            
            # model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            # model.add(Activation('relu'))

            # model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            # model.add(Activation('relu'))

            # model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            # model.add(Activation('relu'))

            # model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            # model.add(Activation('relu'))

            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            model.add(Dense(env.get_dqn_output_size(), kernel_initializer='he_normal', activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition,step):
        self.replay_memory.append(transition)


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


        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(preprocessing.scale(X)), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if ((self.target_update_counter > UPDATE_TARGET_EVERY) or (OppWin)):
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):         
        # standardize the data attributes
        tempArray = preprocessing.scale(state)
        return self.model.predict(np.array(tempArray).reshape(-1, *state.shape))[0]



# def DQN_TRAINING_ROUTINE():
#######################################################################
##                              Training                             ##
#######################################################################

agent = Deep_Q_Network()

if (TRAINING):
    episode_number = 0
    whoWonTheGame = 3
    recVid = GameRecord()

    signal.signal(signal.SIGINT, recVid.signal_handler)
    # print('Press Ctrl+C')
    # signal.pause()
    
    store_training_video_flag = True

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        time.sleep(DELAY_PER_EPISODE)

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.new_game()
        new_state = []
        # Reset flag and start iterating until episode ends
        done = False
        switcher = np.random.randint(0, 2)
               
        reward_plus = 0
        episode_number+=1
        # print ("\n\n\n==================>>>>> NEW GAME\n\n")
        while (not done):
            whoWonTheGame = 3
            switcher = not switcher
            # print("switcher : {}".format(switcher))
            if (switcher):
                #----------------------RoboCon----------------------#
                counter_move_flag = False
                # action, counter_move_flag = env.counter_move(2)
                if (not counter_move_flag):
                    # This part stays mostly the same, the change is to query a model for Q values
                    if (np.random.random() > epsilon):
                        # Get action from Q table
                        cur_state_temp = deepcopy(current_state)
                        action = np.argmax(agent.get_qs(current_state))
                        current_state = deepcopy(cur_state_temp)
                    else:
                        # Get random action
                        action = np.random.randint(1, env.get_dqn_output_size()+1)
                    reward_plus = 0

                elif (counter_move_flag):
                    reward_plus = REWARD_PLUS_VAL

                valid = False
                new_state, reward, done, valid = env.step(action)
                
                while (not valid):
                    reward_plus = 0
                    action = np.random.randint(1, env.get_dqn_output_size()+1)
                    new_state, reward, done, valid = env.step(action)
                valid = False

                if (SHOW_PREVIEW):
                    env.render()
                    time.sleep(0.5)
                    if (done):
                        input()    

                if (store_training_video_flag and RECORD_TRAINING):
                    # print(f"RED PLAYED {action}")
                    recState = deepcopy(new_state)
                    recVid.update_board(new_state)
                    new_state = deepcopy(recState)
                # print (f"{step} RoboCon Turn cur == new : {current_state == new_state}")
            elif (not switcher):
                #----------------------Opponent----------------------#
                # This part stays mostly the same, the change is to query a model for Q values
                if (TRAIN_OPPONENT):
                    if (OPPONENT_DIFFICULTY_HARD):
                        counter_move_flag = False
                        action, counter_move_flag = env.counter_move(1)
                    else:
                        counter_move_flag = False

                    if (not counter_move_flag):
                        if (np.random.random() > epsilon):
                            # Get action from Q table
                            cur_state_temp = deepcopy(current_state)
                            action = np.argmax(agent.get_qs(current_state))
                            current_state = deepcopy(cur_state_temp)
                            
                        else:
                            # Get random action
                            action = np.random.randint(1, env.get_dqn_output_size()+1)

                    valid = False
                    new_state, reward, done, valid = env.opponent_step(action)
                    
                    while (not valid):
                        action = np.random.randint(1, env.get_dqn_output_size()+1)
                        new_state, reward, done, valid = env.opponent_step(action)
                    valid = False
                
                elif (not TRAIN_OPPONENT):
                    if (OPPONENT_DIFFICULTY_HARD):
                        counter_move_flag = False
                        action, counter_move_flag = env.counter_move(1)
                        
                        if (not counter_move_flag):
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                        
                        valid = False
                        new_state, reward, done, valid = env.opponent_step(action)
                        
                    while (not valid):
                        valid = False
                        action = np.random.randint(1, env.get_dqn_output_size()+1)
                        new_state, reward, done, valid = env.opponent_step(action)
                valid = False
                
                if (SHOW_PREVIEW):
                    env.render()
                    time.sleep(0.5)
                    if (done):
                        input()

                # print (f"{step} Opponent Turn cur == new : {current_state == new_state}")
                if (store_training_video_flag and RECORD_TRAINING):
                    # print(f"YELLOW PLAYED {action}")
                    recState = deepcopy(new_state)
                    recVid.update_board(new_state)
                    new_state = deepcopy(recState)
            if (done):
                # new_state = deepcopy(current_state)
                # if (store_training_video_flag and RECORD_TRAINING):
                #     recState = deepcopy(new_state)
                #     recVid.update_board(new_state)
                #     new_state = deepcopy(recState)
                if(env.check_if_red_has_won()):
                    # print(f"{episode_number} --->>> RED WINS <<<---")
                    whoWonTheGame = 2
                elif(env.check_if_yellow_has_won()):
                    # print(f"{episode_number} --->>> YELLOW WINS <<<---")
                    whoWonTheGame = 1
                elif(env.check_if_draw()):
                    # print(f"{episode_number} --->>> ITS A DRAW <<<---")
                    whoWonTheGame = 0
            action = action - 1
            # env.render()
            # time.sleep(1)
            ################################################################
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward + reward_plus
            
            # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            #     env.render()
                
            # Every step we update replay memory and train main network
            memBlock = (deepcopy(current_state), action, reward+reward_plus, deepcopy(new_state), done, env.check_if_yellow_has_won())
            agent.update_replay_memory(memBlock,step)
            
            # print (f"{step} cur == new : {current_state == new_state}")
            agent.train(done, step)

            current_state = deepcopy(new_state)
            step += 1


        if (store_training_video_flag and RECORD_TRAINING):
            recVid.gameFinished(episode_number,whoWonTheGame)
        store_training_video_flag = False
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
                #agent.model.save(f'models/{datetime.datetime.now()}__{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
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
                #agent.model.save(f'models/{datetime.datetime.now()}__{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                agent.model.save(MODEL_PATH + "LAST_RUN/{}__{}__{:7.2f}max_{:7.2f}avg_{:7.2f}min.model"
                .format(datetime.datetime.now(),MODEL_NAME,max_reward,average_reward,min_reward))

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY**EPSILON_DECAY_POWER
            epsilon = max(MIN_EPSILON, epsilon)
        
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
            # print("switcher : {}".format(switcher))
            env.render()
            if (switcher):
                #----------------------RoboCon----------------------#
                counter_move_flag = False
                # action, counter_move_flag = env.counter_move(2)
                # print("Counter move flag = {}".format(counter_move_flag))
                # print (f"{step} RoboCon turn cur == new : {current_state == new_state}")
                # print (f"F RoboCon Before, cur : {current_state}\nnew : {new_state}")

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
                # print (f"RoboCon After, cur : {current_state}\nnew : {new_state}")
                

                env.render()
                #----------------------Opponent----------------------#
                while (not valid and not done):    
                    while (True):
                        try:
                            action = int(input("User's turn, enter move between 1 to 7 or enter 0 to quit the game : "))
                            
                            if (action == 0):
                                # return
                                action = 0
                                Exception("Game Over")
                            elif (action > 7):
                                action = 7
                            elif (action < 1):
                                action = 1
                            break

                        except ValueError:
                            print("===>>> Error: Please enter a number between 1 and 7 <<<===")
                        
                    valid = False
                    new_state, reward, done, valid = env.opponent_step(action)

                    if (not valid):
                        print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                env.render()

            elif (not switcher):
                #----------------------Opponent----------------------#
                valid = False
                while (not valid):
                    while (True):
                        try:
                            action = int(input("User's turn, enter move between 1 to 7 or enter 0 to quit the game : "))
                            
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
                    valid = False
                    new_state, reward, done, valid = env.opponent_step(action)

                    if (not valid):
                        print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                env.render()

                #----------------------RoboCon----------------------#
                if(not done):
                    counter_move_flag = False
                    # action, counter_move_flag = env.counter_move(2)
                    # print("Counter move flag = {}".format(counter_move_flag))
                    # print (f"{step} RoboCon turn cur == new : {current_state == new_state}")
                    # print (f"RoboCon Before, cur : {current_state}\nnew : {new_state}")
                    
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
                    # print (f"RoboCon After, cur : {current_state}\nnew : {new_state}")
    
                env.render()    
            if (done):
                if(env.check_if_red_has_won()):
                    print("--->>> RED WINS <<<---")
                elif(env.check_if_yellow_has_won()):
                    print("--->>> YELLOW WINS <<<---")
                elif(env.check_if_draw()):
                    print("--->>> ITS A DRAW <<<---")
            action = action - 1
            
            # time.sleep(1)
            ################################################################
            
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done, False),step)
            agent.train(done, step)
            
            # print (f"{step} cur == new : {current_state == new_state}")
            current_state = deepcopy(new_state)
            step += 1



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

# DQN_TRAINING_ROUTINE()