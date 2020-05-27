#!/usr/bin/env python3
 
import os
import random
import sys
import time
import datetime
from collections import deque
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


# print(sys.path)
SHUTDOWN_RPI_CODE = 147258
##############################
TRAINING = False
EPISODES = 50000     
#---------------------------##
HIDDEN_LAYERS = 100         ##
LEARNING_RATE = 0.001       ##
REWARD_PLUS_VAL = 1
DISCOUNT = 0.99             ##
DELAY_PER_EPISODE = 0.01    ##
TRAIN_OPPONENT = True       ##
MODEL_PATH = "./dqn_models/"
LOAD_MODEL = MODEL_PATH + "2020-04-21" + "/" + "2020-04-21 23:10:51.644057__Connect4_DQN__ 100.00max_  17.50avg_-200.00min.model" 
#"./models_archive/v3/2020-02-28 04:01:28.617140__Connect4_DQN___290.00max__111.30avg_-180.00min__2020-02-28 04:01:28.617164.model" 
#"./models/2020-02-27 02:44:04.558768__Connect4_DQN___100.00max__-14.75avg_-200.00min__2020-02-27 02:44:04.558788.model"
##############################



REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'Connect4_DQN'
MIN_AVG_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20




# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.099975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 200  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.isdir('models'):
    os.makedirs('models')

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
        print("Loading {}".format(LOAD_MODEL))
        model = load_model(LOAD_MODEL)
        print("Model {} loaded!".format(LOAD_MODEL))
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^ USING LOADED MODEL ^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        return model

    def create_model(self):
        if (LOAD_MODEL is not None):
            model = self.load_trained_model()
            
        else:
            # model.add(Dropout(0.2))
            print("FIRING NEW MODEL ---->>")
            model = Sequential()

            model.add(Dense(HIDDEN_LAYERS, input_shape=(env.get_dqn_input_size(),),kernel_initializer='glorot_uniform'))
            model.add(Activation('sigmoid'))
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))
            
            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            model.add(Dense(HIDDEN_LAYERS,kernel_initializer='he_normal'))
            model.add(Activation('relu'))

            model.add(Dense(env.get_dqn_output_size(), kernel_initializer='he_normal', activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


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


        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(preprocessing.scale(X)), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state): 
        # # standardize the data attributes
        tempArray = preprocessing.scale(state)
        return self.model.predict(np.array(tempArray).reshape(-1, *state.shape))[0]


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

#######################################################################
##                              Training                             ##
#######################################################################
def connect4player(visionBoardQue,instructionQue,data2rpiQue,rpiTaskCompleteQue):
    agent = Deep_Q_Network()

    if (TRAINING):
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

            # Reset flag and start iterating until episode ends
            done = False
            switcher = np.random.randint(0, 2)
            reward_plus = 0
        
            while (not done):
                switcher = not switcher
                # print("switcher : {}".format(switcher))
                if (switcher):
                    #----------------------RoboCon----------------------#
                    counter_move_flag = False
                    action, counter_move_flag = env.counter_move(2)
                    if (not counter_move_flag):
                        # This part stays mostly the same, the change is to query a model for Q values
                        if np.random.random() > epsilon:
                            # Get action from Q table
                            action = np.argmax(agent.get_qs(current_state))
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

                elif (not switcher):
                    #----------------------Opponent----------------------#
                    # This part stays mostly the same, the change is to query a model for Q values
                    if (TRAIN_OPPONENT):
                        if np.random.random() > epsilon:
                            # Get action from Q table
                            action = np.argmax(agent.get_qs(current_state))
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
                        valid = False
                        while (not valid):
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                            new_state, reward, done, valid = env.opponent_step(action)
                    valid = False


                # if (done):
                #     if(env.check_if_red_has_won()):
                #         print("--->>> RED WINS <<<---")
                #     elif(env.check_if_yellow_has_won()):
                #         print("--->>> YELLOW WINS <<<---")
                #     elif(env.check_if_draw()):
                #         print("--->>> ITS A DRAW <<<---")
                action = action - 1
                # env.render()
                # time.sleep(1)
                ################################################################
                # Transform new continous state to new discrete state and count reward
                episode_reward += reward + reward_plus

                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    env.render()

                # Every step we update replay memory and train main network
                agent.update_replay_memory((current_state, action, reward+reward_plus, new_state, done))
                agent.train(done, step)

                current_state = new_state
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
                    #agent.model.save(f'models/{datetime.datetime.now()}__{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
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
        externalBoard = np.zeros(42,int) 
        internalBoard = np.zeros(42,int)

        # Iterate over episodes
        while (not FINISHED_PLAYING):
            client_instruction = instructionQue.get()
            if (client_instruction == 9):
                FINISHED_PLAYING = True
                return
            # elif(client_instruction == 1):
                #new game
            


            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.new_game()

            # Reset flag and start iterating until episode ends
            done = False
            switcher = np.random.randint(0, 100)

            
            while (not done):
                # print("switcher : {}".format(switcher))
                done = False
                valid = False
                action = 0
                env.render()
                if (switcher >= 50):
                    #----------------------RoboCon----------------------#
                    action, counter_move_flag = env.counter_move(2)
                    # print("Counter move flag = {}".format(counter_move_flag))
                    # counter_move_flag = False
                    if (not counter_move_flag):
                        action = np.argmax(agent.get_qs(current_state))
                    
                    if (action < 1):
                        action = 1
                    elif (action > env.COLs):
                        action = env.COLs

                    # print("Move Number {} - RoboCon_FM Action: {} - Valid Move Status: {}".format(step,action,valid))
                    # print("Move Number {} - RoboCon_FM Action: {}".format(step,action))
                    new_state, reward, done, valid = env.step(action)
                    
                    while (not valid):
                            action = np.random.randint(1, env.get_dqn_output_size()+1)
                            new_state, reward, done, valid = env.step(action)
                            # print("Move Number {} - RoboCon_FM Action: {} - Valid Move Status: {}".format(step,action,valid))
                    # print("Move Number {} - RoboCon_FM Action: {} - Valid Move Status: {}".format(step,action,valid))
                    valid = False
                    
                    # print("{} - RoboCon_FM Action: {}\n, External Board: {}\n, Internal Board: {}\n".format(step,action,externalBoard,internalBoard))
                    print("Move Number {} - RoboCon_FM Action: {}".format(step,action))
                    internalBoard = list(env.get_board())
                    print("External Board : ")
                    agent.print_board(externalBoard)
                    print("Internal Board : ")
                    agent.print_board(internalBoard)

                    env.render()

                    #___RPI CONTROL__________##
                    data2rpiQue.put(action)  ##
                    rpiTaskCompleteQue.get() ##
                    ###########################

                    #----------------------Opponent----------------------#
                    while (not valid and not done):
                        exBoardBuff = visionBoardQue.get()
                        while (not any(exBoardBuff)):
                            exBoardBuff = visionBoardQue.get()

                        # print(f"exBoardBuff ---------->>> {exBoardBuff}")
                        externalBoard = np.array(exBoardBuff,int)
                        internalBoard = list(env.get_board())

                        # print("External Board : {}".format(externalBoard))
                        done_instruction = instructionQue.get()
                        if ( done_instruction == 2):
                            done = True
                            FINISHED_PLAYING = False
                            return

                        equal_board = externalBoard == internalBoard
                        if (equal_board.all()):
                            raise Exception("Error: players move not found, (connect4dqn.py -->> connect4player -->> Testing -->> Opponents turn)")

                        board_i_ = 0
                        breakFlag = False
                        while ((board_i_ < len(internalBoard)) and not breakFlag):
                            if (externalBoard[board_i_] != internalBoard[board_i_]):
                                if((externalBoard[board_i_] == 2) and (internalBoard[board_i_] == 0)):
                                    _ , action = env.state_2_row_col(board_i_)
                                    action+=1
                                    # internalBoard[board_i_] = 2
                                    breakFlag = True
                            board_i_+=1

                        # equal_board = externalBoard == internalBoard
                        # if (not(equal_board.all())):
                        #     raise Exception("Error: internal and external boards are different, (connect4dqn.py -->> connect4player -->> Testing -->> Opponents turn)")

                        # print("{} - Opponent_SM Action: {}\n, External Board: {}\n, Internal Board: {}\n".format(step,action,externalBoard,internalBoard))
                        print("Move Number {} - Opponent_SM Action: {}".format(step,action))
                        print("External Board : ")
                        agent.print_board(externalBoard)
                        print("Internal Board : ")
                        agent.print_board(internalBoard)

                        new_state, reward, done, valid = env.opponent_step(action)
                        if (not valid):
                            print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                    env.render()
            
                elif (switcher < 50):
                    #----------------------Opponent----------------------#
                    valid = False
                    while (not valid):
                        exBoardBuff = visionBoardQue.get()
                        while (not any(exBoardBuff)):
                            exBoardBuff = visionBoardQue.get()

                        externalBoard = np.array(exBoardBuff,int)
                        # externalBoard = np.array(visionBoardQue.get(),int)
                        internalBoard = list(env.get_board())

                        # print("External Board : {}".format(externalBoard))
                        done_instruction = instructionQue.get()
                        if ( done_instruction == 2):
                            done = True
                            FINISHED_PLAYING = False
                            return

                        equal_board = externalBoard == internalBoard
                        if (equal_board.all()):
                            raise Exception("Error: players move not found, (connect4dqn.py -->> connect4player -->> Testing -->> Opponents turn)")

                        board_i_ = 0
                        breakFlag = False
                        while ((board_i_ < len(internalBoard)) and not breakFlag):
                            if (externalBoard[board_i_] != internalBoard[board_i_]):
                                if((externalBoard[board_i_] == 2) and (internalBoard[board_i_] == 0)):
                                    _ , action = env.state_2_row_col(board_i_)
                                    action+=1
                                    # internalBoard[board_i_] = 2
                                    breakFlag = True
                            board_i_+=1

                        # equal_board = externalBoard == internalBoard
                        # if (not(equal_board.all())):
                        #     raise Exception("Error: internal and external boards are different, (connect4dqn.py -->> connect4player -->> Testing -->> Opponents turn)")
                        
                        # print("{} - Opponent_FM Action: {}\n, External Board: {}\n, Internal Board: {}\n".format(step,action,externalBoard,internalBoard))
                        print("{} - Opponent_FM Action: {}".format(step,action))
                        internalBoard = list(env.get_board())
                        print("External Board : ")
                        agent.print_board(externalBoard)
                        print("Internal Board : ")
                        agent.print_board(internalBoard)
    
                        new_state, reward, done, valid = env.opponent_step(action)
                        if (not valid):
                            print("===>>> INVALID MOVE : {}, PLEASE TRY AGAIN <<===".format(action))
                    env.render()

                    #----------------------RoboCon----------------------#
                    if(not done):
                        action, counter_move_flag = env.counter_move(2)
                        # print("Counter move flag = {}".format(counter_move_flag))
                        # counter_move_flag = False
                        if (not counter_move_flag):
                            action = np.argmax(agent.get_qs(current_state))

                        if (action < 1):
                            action = 1
                        elif (action > env.COLs):
                            action = env.COLs

                        
                        # print("{} - RoboCon_SM Action: {}".format(step,action))
                        
                        new_state, reward, done, valid = env.step(action)
                        
                        while (not valid):
                                action = np.random.randint(1, env.get_dqn_output_size()+1)
                                new_state, reward, done, valid = env.step(action)
                        valid = False
                        
                        # print("{} - RoboCon_SM Action: {}\n, External Board: {}\n, Internal Board: {}\n".format(step,action,externalBoard,internalBoard))
                        print("{} - RoboCon_SM Action: {}".format(step,action))
                        print("External Board : ")
                        agent.print_board(externalBoard)
                        print("Internal Board : ")
                        agent.print_board(internalBoard)

                        #___RPI CONTROL__________##
                        data2rpiQue.put(action)  ##
                        rpiTaskCompleteQue.get() ##
                        ###########################
                    
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
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step)

                current_state = new_state
                step += 1



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