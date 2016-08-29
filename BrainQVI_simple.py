# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import numpy as np
import random
from collections import deque
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, merge
from keras.layers.core import Flatten, Activation, Dropout
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.optimizers import *
from keras.models import Model
import os
import matplotlib.pyplot as plt
import laplotter
import h5py

# Hyper Parameters:
FRAME_PER_ACTION = 4
GAMMA = 0.95  # decay rate of past observations
OBSERVE = 200.  # time steps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001
INITIAL_EPSILON = 0.01
weights_file_name = 'weights.qvi.simple.hd5'

class BrainQVISimple:
    def __init__(self):
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = 2
        self.q_space = np.zeros((600, 288, 18, 2))
        self.current_state = []
        self.cum_q_change = 0

        if os.path.isfile(weights_file_name):
            data_file = h5py.File(weights_file_name, 'r')
            self.q_space = data_file['weights'][()]
            data_file.close()

    def set_perception(self, next_state, action, reward, terminal):
        n_state = [next_state[5] - next_state[0] + 300, next_state[3], next_state[1] + 7]
        state_action = self.current_state[:]
        state_action.append(action)
        if terminal:
            new_q = reward
        else:
            new_q = reward + GAMMA * np.max(self.q_space[tuple(n_state)])
        self.cum_q_change += (self.q_space[tuple(state_action)] - new_q)
        self.q_space[tuple(state_action)] = new_q

        self.current_state = n_state
        self.time_step += 1

        # print info
        if self.time_step % 100 == 0:
            print "TIME_STEP: ", self.time_step, "/ EPSILON: ", self.epsilon, \
                "CUM_Q_CHANGE: ", self.cum_q_change
            self.cum_q_change = 0;

            if self.time_step % 10000 == 0:
                data_file = h5py.File(weights_file_name, 'w')
                data_file.create_dataset('weights', data=self.q_space)
                data_file.close()

    def get_action(self):
        if self.time_step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
            else:
                q = self.q_space[tuple(self.current_state)]
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
        self.current_state = [state[5] - state[0] + 300, state[3], state[1] + 7]

