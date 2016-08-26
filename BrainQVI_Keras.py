# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import numpy as np
import random
from collections import deque
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, merge
from keras.layers.core import Flatten, Activation
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.models import Model
import os
import matplotlib.pyplot as plt


# Hyper Parameters:
FRAME_PER_ACTION = 6
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # time steps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.01  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of mini_batch
UPDATE_TIME = 100


class BrainQVICallback(Callback):
    def __init__(self):
        super(BrainQVICallback, self).__init__()
        self.batch_count = 0

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.batch_count += 1
        if self.batch_count == 500:
            self.batch_count = 0
            print 'training loss:%10.4f, accuracy:%2.4f' % (logs.get('loss'), logs.get('acc'))


class BrainQVI:
    def __init__(self, actions, state_dim):

        # init replay memory
        self.replay_memory = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.current_state = []

        # init Q network
        self.model = self.create_q_network(state_dim)
        self.acc_print = BrainQVICallback()
        # self.acc_print = TensorBoard(log_dir='./tf_logs', histogram_freq=0)

        # loading networks weights
        if os.path.isfile('weights.best_simple.hd5'):
            self.model.load_weights('weights.best_simple.hd5')

    def create_q_network(self, state_shape):
        # network weights
        inputs_state = Input(shape=(state_shape,))
        inputs_action = Input(shape=(1,))

        x_all = merge([inputs_state, inputs_action], mode='concat')
        x_all = Dense(8, activation='linear')(x_all)
        for i in range(5):
            # x_all_1 = Dense(8, activation='tanh')(x_all)
            # x_all = merge([x_all_1, x_all], 'sum')
            x_all = Dense(32, activation='tanh')(x_all)
        # outputs_next_state = Dense(8, activation='tanh')(x_all)
        # outputs_next_state = Dense(state_shape)(outputs_next_state)
        outputs_q = Dense(32, activation='linear')(x_all)
        outputs_q = Dense(1)(outputs_q)

        # model = Model(input=[inputs_state, inputs_action], output=[outputs_q, outputs_next_state])
        model = Model(input=[inputs_state, inputs_action], output=[outputs_q])
        model.compile(optimizer='rmsprop',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        model.summary()

        return model

    def train_q_network(self):
        # Step 1: obtain random batch_data from replay memory
        batch_data = random.sample(self.replay_memory, BATCH_SIZE)

        state_batch = [data[0] for data in batch_data]
        action_batch = [data[1] for data in batch_data]
        reward_batch = [data[2] for data in batch_data]
        next_state_batch = [data[3] for data in batch_data]
        terminal_batch = [data[4] for data in batch_data]

        # Step 2: calculate y
        q_batch = []
        for i in range(0, BATCH_SIZE):
            if terminal_batch[i]:
                q_batch.append(reward_batch[i])
            else:
                # q, _ = self.model.predict([np.asarray([next_state_batch[i] for i in range(self.actions)]),
                #                            np.arange(0, self.actions, 1)])
                q = self.model.predict([np.asarray([next_state_batch[i] for i in range(self.actions)]),
                                        np.asarray([[0], [1]])])
                                        # np.arange(0, self.actions, 1)])
                q_batch.append(reward_batch[i] + GAMMA * np.max(q))

        # self.model.fit([np.asarray(state_batch), np.asarray(action_batch)],
        #                [np.asarray(q_batch), np.asarray(next_state_batch)],
        #                batch_size=BATCH_SIZE, verbose=0)  # , callbacks=[self.acc_print])
        self.model.fit([np.asarray(state_batch), np.asarray(action_batch)],
                       [np.asarray(q_batch)],
                       batch_size=BATCH_SIZE, verbose=0, callbacks=[self.acc_print])

        # save network every 10000 iteration
        if self.time_step % 10000 == 0:
            self.model.save_weights('weights.best_simple.hd5')

    def set_perception(self, next_state, action, reward, terminal):
        self.replay_memory.append((self.current_state, action, reward, next_state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time_step > OBSERVE:
            self.train_q_network()

        self.current_state = next_state
        self.time_step += 1

        # print info
        state = ""
        if self.time_step <= OBSERVE:
            state = "observe"
        elif OBSERVE < self.time_step <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.time_step % 100 == 0:
            print "TIME_STEP", self.time_step, "/ STATE", state, \
                "/ EPSILON", self.epsilon

    def get_action(self):
        if self.time_step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
            else:
                # q, _ = self.model.predict([np.asarray([self.current_state for i in range(self.actions)]),
                #                           np.arange(0, self.actions, 1)])
                q = self.model.predict([np.asarray([self.current_state for i in range(self.actions)]),
                                        np.asarray([[0], [1]])])
                                        # np.arange(0, self.actions, 1)])
                print 'q value predict:', q
                action_index = np.argmax(q)
        else:
            # do nothing
            action_index = 0

        # change epsilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action_index

    def set_init_state(self, state):
        self.current_state = state

