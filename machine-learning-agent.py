#! /usr/bin/env -vS python

import torch
import random
import numpy as np
from collections import deque
from snakegame import SnakeGameML, Direction, Point, SIZE_GRID
from model import Linear_QNet, QTrainer
from plotter import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


# #############################################    CLASS DEFINITIONS    ################################################

class SnakeAgentML:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # RANDOMNESS
        self.gamma = 0.9  # DISCOUNT RATE
        self.memory = deque(maxlen=MAX_MEMORY)  # This pops off the left side of the deque. (like a list/array)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # NEIGHBORING CELLS  -  (Not part of state, but used for determining the danger outlook, which IS part of state.)
        cell_l: Point = Point(head.x - SIZE_GRID, head.y)
        cell_r: Point = Point(head.x + SIZE_GRID, head.y)
        cell_u: Point = Point(head.x, head.y - SIZE_GRID)
        cell_d: Point = Point(head.x, head.y + SIZE_GRID)

        # DIRECTION OF MOTION  -  4 inputs  (Part of state AND used to calculate danger outlook.)
        moving_left: bool = game.direction == Direction.LEFT
        moving_right: bool = game.direction == Direction.RIGHT
        moving_up: bool = game.direction == Direction.UP
        moving_down: bool = game.direction == Direction.DOWN

        # DANGER OUTLOOK  -  3 inputs
        danger_straight: bool = (
            (moving_right and game.is_collision(cell_r)) or
            (moving_left and game.is_collision(cell_l)) or
            (moving_up and game.is_collision(cell_u)) or
            (moving_down and game.is_collision(cell_d))
        )
        danger_right: bool = (
            (moving_up and game.is_collision(cell_r)) or
            (moving_down and game.is_collision(cell_l)) or
            (moving_left and game.is_collision(cell_u)) or
            (moving_right and game.is_collision(cell_d))
        )
        danger_left: bool = (
            (moving_down and game.is_collision(cell_r)) or
            (moving_up and game.is_collision(cell_l)) or
            (moving_right and game.is_collision(cell_u)) or
            (moving_left and game.is_collision(cell_d))
        )

        # FOOD RECKONING  -  4 inputs
        food_left: bool = game.food.x < game.head.x
        food_right: bool = game.food.x > game.head.x
        food_up: bool = game.food.y < game.head.y
        food_down: bool = game.food.y > game.head.y

        state = [  # 11 inputs to the model
            # STATE: DANGER OUTLOOK  -  3 inputs
            danger_straight,
            danger_right,
            danger_left,
            # STATE: DIRECTION OF MOTION  -  4 inputs
            moving_left,
            moving_right,
            moving_up,
            moving_down,
            # STATE: FOOD RECKONING  -  4 inputs
            food_left,
            food_right,
            food_up,
            food_down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # RANDOM MOVES: Trade-off exploration vs. exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # PREDICT BEST MOVE  # TODO: Confirm this description?
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


# #############################################    FUNCTION DEFINITIONS    #############################################

def hardware_gpu_check():  # Specific to CUDA/NVidia
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    cuda_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    for device in cuda_devices:
        print(f"CUDA device name: [{torch.cuda.get_device_name(device)}]")
# TODO: Check out this module for testing GPU:
# https://pypi.org/project/test-pytorch-gpu/
# UPDATE: Ran fine on my machine. My GPU is: NVIDIA RTX A5000 Laptop GPU

def gpu_resource_sample():  # Specific to CUDA/NVidia
    memory_allocated = round(torch.cuda.memory_allocated(0)/1024**3,1)
      # Returns the current GPU memory usage by tensors in bytes. (Device 0)
    memory_reserved = round(torch.cuda.memory_reserved(0)/1024**3,1)
    print(f"GPU Memory:    Allocated: {memory_allocated} GB    Reserved: {memory_reserved} GB")
    # OTHER EXPERIMENTAL STATS BELOW HERE:
    max_memory_reserved = round(torch.cuda.max_memory_reserved(0)/1024**3,1)
    print(f"GPU Memory:    Max Reserved: {max_memory_reserved} GB")
      # Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.
# end def gpu_resource_sample()  -  # TODO: Troubleshoot all three memory values always 0.0.
# Othere tools can briefly see a small amount of allocated usage. Perhaps this programs just uses no GPU memory?
#   No I have a feeling maybe the values just exist for a very short amount of time or there is some issue with the
#   monitoring libs/calls. Not sure yet. I do want robust resource monitoring however, especially for other projects.
# NOTE: 'reserved' used to be called 'cached' everywhere in older CUDA stuff.

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = SnakeAgentML()
    game = SnakeGameML()
    while True:
        # CAPTURE CURRENT STATE OF GAME
        state_current = agent.get_state(game)

        # PREDICT NEXT MOVE (WITH epsilon-TUNABLE RANDOMIZATION)  # Randomly, some moves just pick a random move.
        final_move = agent.get_action(state_current)

        # ADVANCE A GAME STEP USING THE NEW MOVE. CAPTURE NEW STATE.
        reward, done, score = game.advance_game_step(final_move)
        state_new = agent.get_state(game)

        # ****  TRAIN SHORT MEMORY  ****
        agent.train_short_memory(state_current, final_move, reward, state_new, done)

        # REMEMBER THE FULL RESULT OF THIS ITERATION
        agent.remember(state_current, final_move, reward, state_new, done)

        if done:
            gpu_resource_sample()
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()  # ****  TRAIN LONG MEMORY  ****

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.n_games}    Score: {score}    Record: {record}")

            # UPDATE PLOT OF TRAINING PROGRESS
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    hardware_gpu_check()
    gpu_resource_sample()
    train()


##
#

# Unrelated but noticed this paper. Looks interesting. Read this later:
# Deep Residual Learning for Image Recognition

# https://arxiv.org/pdf/1512.03385

# In understanding Convolutional Neural Networks like PyTorch, one will benefit from understanding the mathematical
# concept of convolution.
# Wolfrom MathWorld - Convolution:
# https://mathworld.wolfram.com/Convolution.html
# * This article has a great animation. Visualization is very helpful in understanding complex mathematics!

# GPUSTAT - Python GPU (NVidia) Monitoring tool. (I'm looking at how it works for other monitoring interests.)
# https://pypi.org/project/gpustat/
# This shows that in my RTX A5000 this project never hits over 5%-10% GPU usage if that and increases the temp about 10 degrees.
#   So my GPU easily handles this project. This is good because I would like to add more inputs and see if we can make
#   it even better at playing the snake game. I think we can if we add some other kinds of 'danger outlook'. There are
#   a few simply things we could try. TODO: GOAL: Add USEFUL new senses which are COMPUTATIONALLY CHEAP.
#   TODO: Idea: Look out farther for danger, even one more cell would add a new capability. Could look out full distance!?

# CHECK THIS OUT! NVTOP - CUDA GPU MONITORING (In the console, but beautifully graphical!)
# https://github.com/Syllo/nvtop


##
#
