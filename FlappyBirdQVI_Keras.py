# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import sys
import numpy as np
import pygame

from BrainQVI_Keras_V2 import BrainQVI
from BrainQVI_simple import BrainQVISimple
sys.path.append("game/")
import FlappyBirdGame as fbg


class FlappyBirdTrainer:
    def __init__(self):
        self.action_space = np.eye(2)
        # self.brain = BrainQVI(2, 6)
        self.brain = BrainQVISimple()
        self.game = fbg.FlappyBirdGame(with_graph=True, fps=30, difficulty=10)
        self.manual_action = False

    def get_action(self, brain=True):
        action_index = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.manual_action = not self.manual_action

        if self.manual_action:
            if keys[pygame.K_SPACE]:
                action_index = 1
        else:
            action_index = self.brain.get_action()

        return action_index

    def play_flappy_bird(self):
        observation, reward, terminal, positions = self.game.frame_step(self.action_space[0])
        self.brain.set_init_state(positions)
        while True:
            action_index = self.get_action()
            observation, reward, terminal, positions = self.game.frame_step(self.action_space[action_index])
            # print positions, reward, terminal, action_index
            self.brain.set_perception(positions, action_index, reward, terminal)


def main():
    fbt = FlappyBirdTrainer()
    fbt.play_flappy_bird()


if __name__ == '__main__':
    main()
