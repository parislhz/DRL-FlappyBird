import random
import pygame
import flappy_bird_utils
from itertools import cycle


class FlappyBirdGame:

    def __init__(self, with_graph=True, fps=30, difficulty=10):

        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.with_graph = with_graph
        self.difficulty = difficulty
        self.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

        pygame.init()
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')

        self.FPS = fps
        self.RES_IMAGES, self.RES_SOUNDS, self.RES_HIT_MASKS = flappy_bird_utils.load()
        self.PIPE_GAP_SIZE = 300 - 20 * difficulty
        self.BASE_Y = self.SCREEN_HEIGHT * 0.79

        self.PLAYER_WIDTH = self.RES_IMAGES['player'][0].get_width()
        self.PLAYER_HEIGHT = self.RES_IMAGES['player'][0].get_height()
        self.PIPE_WIDTH = self.RES_IMAGES['pipe'][0].get_width()
        self.PIPE_HEIGHT = self.RES_IMAGES['pipe'][0].get_height()
        self.BACKGROUND_WIDTH = self.RES_IMAGES['background'].get_width()
        self.BASE_SHIFT = self.RES_IMAGES['base'].get_width() - self.BACKGROUND_WIDTH

        self.score = 0
        self.base_x = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.player_x = int(self.SCREEN_WIDTH * 0.2)
        self.player_y = int((self.SCREEN_HEIGHT - self.PLAYER_HEIGHT) / 2)
        self.pipeVelX = -4
        self.playerVelY = 0         # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10     # max vel along Y, max descend speed
        self.playerMinVelY = -8     # min vel along Y, max ascend speed
        self.playerAccY = 1         # players downward acceleration
        self.playerFlapAcc = -7     # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        self.upper_pipes = []
        self.lower_pipes = []
        self.reset()

    def reset(self):
        self.score = 0
        self.base_x = 0
        self.playerIndex = 0
        self.loopIter = 0
        self.player_x = int(self.SCREEN_WIDTH * 0.2)
        self.player_y = int((self.SCREEN_HEIGHT - self.PLAYER_HEIGHT) / 2)

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.pipeVelX = -4
        self.playerVelY = 0         # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10     # max vel along Y, max descend speed
        self.playerMinVelY = -8     # min vel along Y, max ascend speed
        self.playerAccY = 1         # players downward acceleration
        self.playerFlapAcc = -7     # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        new_pipe1 = self.get_random_pipe()
        new_pipe2 = self.get_random_pipe()
        self.upper_pipes = [
            {'x': self.SCREEN_WIDTH, 'y': new_pipe1[0]['y']},
            {'x': self.SCREEN_WIDTH + (self.SCREEN_WIDTH / 2), 'y': new_pipe2[0]['y']},
        ]
        self.lower_pipes = [
            {'x': self.SCREEN_WIDTH, 'y': new_pipe1[1]['y']},
            {'x': self.SCREEN_WIDTH + (self.SCREEN_WIDTH / 2), 'y': new_pipe2[1]['y']},
        ]

    '''returns a randomly generated pipe'''
    def get_random_pipe(self):
        # y of gap between upper and lower pipe
        gap_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gap_ys) - 1)
        gap_y = gap_ys[index] * self.difficulty / 10

        gap_y += int(self.BASE_Y * 0.2)
        pipe_x = self.SCREEN_WIDTH + 10

        return [
            {'x': pipe_x, 'y': gap_y - self.PIPE_HEIGHT},     # upper pipe
            {'x': pipe_x, 'y': gap_y + self.PIPE_GAP_SIZE},   # lower pipe
        ]

    '''Checks if two objects collide and not just their rects'''
    @staticmethod
    def pixel_collision(rect1, rect2, hit_mask1, hit_mask2):
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hit_mask1[x1 + x][y1 + y] and hit_mask2[x2 + x][y2 + y]:
                    return True
        return False

    '''returns True if player collders with base or pipes.'''
    def check_crash(self):
        # if player crashes into ground
        if self.player_y + self.PLAYER_HEIGHT >= self.BASE_Y - 1:
            return True
        else:

            player_rect = pygame.Rect(self.player_x, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

            for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
                # upper and lower pipe rects
                u_pipe_rect = pygame.Rect(uPipe['x'], uPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)
                l_pipe_rect = pygame.Rect(lPipe['x'], lPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)

                # player and upper/lower pipe hitmasks
                p_hit_mask = self.RES_HIT_MASKS['player'][self.playerIndex]
                u_hit_mask = self.RES_HIT_MASKS['pipe'][0]
                l_hit_mask = self.RES_HIT_MASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                u_collide = self.pixel_collision(player_rect, u_pipe_rect, p_hit_mask, u_hit_mask)
                l_collide = self.pixel_collision(player_rect, l_pipe_rect, p_hit_mask, l_hit_mask)

                if u_collide or l_collide:
                    return True

        return False

    '''displays score in center of screen'''
    def show_score(self):
        score_digits = [int(x) for x in list(str(self.score))]
        total_width = 0  # total width of all numbers to be printed

        for digit in score_digits:
            total_width += self.RES_IMAGES['numbers'][digit].get_width()

        x_offset = (self.SCREEN_WIDTH - total_width) / 2

        for digit in score_digits:
            self.screen.blit(self.RES_IMAGES['numbers'][digit], (x_offset, self.SCREEN_HEIGHT * 0.1))
            x_offset += self.RES_IMAGES['numbers'][digit].get_width()

    def frame_step(self, input_actions):
        if self.with_graph:
            pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.player_y > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                # SOUNDS['wing'].play()

        # check for score
        player_mid_pos = self.player_x + self.PLAYER_WIDTH / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                # SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = self.PLAYER_INDEX_GEN.next()
        self.loopIter = (self.loopIter + 1) % 30
        self.base_x = -((-self.base_x + 100) % self.BASE_SHIFT)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.player_y += min(self.playerVelY, self.BASE_Y - self.player_y - self.PLAYER_HEIGHT)
        if self.player_y < 0:
            self.player_y = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self.get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # remove first pipe if its out of the screen
        if self.upper_pipes[0]['x'] < - self.PIPE_WIDTH:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # check if crash here
        is_crash = self.check_crash()
        if is_crash:
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            terminal = True
            self.reset()
            reward = -1

        # draw sprites
        self.screen.blit(self.RES_IMAGES['background'], (0, 0))

        positions = [self.player_y, self.playerVelY, self.playerAccY, 0, 0, 0]
        pos_idx = 1
        first_pipe = True
        for uPipe, lPipe in zip(self.upper_pipes, self.lower_pipes):
            self.screen.blit(self.RES_IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.screen.blit(self.RES_IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
            if uPipe['x'] > self.player_x and first_pipe:
                positions[3] = uPipe['x'] - self.player_x
                positions[4] = uPipe['y'] + self.PIPE_HEIGHT
                positions[5] = lPipe['y']
                first_pipe = False

        self.screen.blit(self.RES_IMAGES['base'], (self.base_x, self.BASE_Y))
        # print score so player overlaps the score
        self.show_score()
        self.screen.blit(self.RES_IMAGES['player'][self.playerIndex],
                        (self.player_x, self.player_y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        if self.with_graph:
            pygame.display.update()
            if self.FPS:
                self.fps_clock.tick(self.FPS)

        # print self.upper_pipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)

        return image_data, reward, terminal, positions

