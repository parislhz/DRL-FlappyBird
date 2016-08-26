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
from keras.models import Model
import os

# Hyper Parameters:
FRAME_PER_ACTION = 8
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.01  # 0.001 # final value of epsilon
INITIAL_EPSILON = 1  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100


class BrainDQN:
    def __init__(self, actions):

        # init replay memory
        self.replayMemory = deque()

        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.action_space = np.eye(actions)

        # init Q network
        self.model = self.createQNetwork()

        # loading networks weights
        if os.path.isfile('weights.best_simple.hd5'):
            self.model.load_weights('weights.best_simple.hd5')

    '''
    def createQNetwork(self):
        # network weights
        inputs_state = Input(shape=(4, 80, 80,))
        x = Convolution2D(32, 8, 8, activation='relu', subsample=(4, 4), border_mode='same')(inputs_state)
        x = MaxPooling2D((2, 2))(x)
        x = Convolution2D(64, 4, 4, activation='relu', subsample=(2, 2), border_mode='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Convolution2D(64, 2, 2, activation='relu')(x)
        #x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        #x = Activation(activation='relu')(x)
        x = Dense(self.actions)(x)

        inputs_action = Input(shape=(self.actions,))

        outputs_q = merge([x, inputs_action], mode='dot', dot_axes=1)

        model = Model(input=[inputs_state, inputs_action], output=outputs_q)
        model.compile(optimizer='rmsprop',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        return model
    '''
    def createQNetwork(self):
        # network weights
        inputs_state = Input(shape=(4, 8,))
        x = Flatten()(inputs_state)
        x = Dense(8, activation='relu')(x)
        for i in range(8):
            x1 = Dense(8, activation='relu')(x)
            x = merge([x1, x], 'sum')
        x = Dense(self.actions)(x)

        inputs_action = Input(shape=(self.actions,))

        outputs_q = merge([x, inputs_action], mode='dot', dot_axes=1)

        model = Model(input=[inputs_state, inputs_action], output=outputs_q)
        model.compile(optimizer='rmsprop',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        model.summary()

        return model

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.model.predict([np.asarray(nextState_batch), np.asarray(action_batch)])
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.model.fit([np.asarray(state_batch), np.asarray(action_batch)], np.asarray(y_batch), verbose=0)

        # save network every 10000 iteration
        if self.timeStep % 10000 == 0:
            self.model.save_weights('weights.best_simple.hd5')

    def setPerception(self, nextObservation, action, reward, terminal):
        #newState = np.append(self.currentState[1:, :, :], nextObservation, axis=0)
        newState = np.append(self.currentState[1:, :], np.asarray([nextObservation]), axis=0)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif OBSERVE < self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.timeStep % 100 == 0:
            print "TIMESTEP", self.timeStep, "/ STATE", state, \
                "/ EPSILON", self.epsilon

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        # QValue = self.model.predict([np.asarray(
        #     [self.currentState for i in range(self.actions)]), self.action_space])
        # action_index = np.argmax(QValue)
        # action = self.action_space[action_index]

        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action = self.action_space[action_index]
            else:
                QValue = self.model.predict([np.asarray(
                    [self.currentState for i in range(self.actions)]), self.action_space])
                action_index = np.argmax(QValue)
                action = self.action_space[action_index]
        else:
            action = self.action_space[0]  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)

