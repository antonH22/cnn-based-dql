import os
import pickle
import random
import math
import numpy as np
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

import torch
import torch.nn as nn

class BombermanCNN(nn.Module):
    def __init__(self):
        super(BombermanCNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 17 * 17, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, len(ACTIONS))

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def setup(self):
    self.coordinate_history = deque([], 20)
    self.action_history = deque([],3)
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Initialize the CNN model
        self.model = BombermanCNN()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

EPS_START = 0.9
EPS_MIN = 0.01
EPS_DECAY = 1000

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # Explore random actions with probability epsilon
    rounds_done = game_state['round']
    eps_threshold = EPS_MIN + (EPS_START - EPS_MIN) * math.exp(-1. * rounds_done / EPS_DECAY)

    if self.train and random.random() <= eps_threshold:
        self.logger.debug(f"Choosing action purely at random. Prob: {eps_threshold * 100:.2f} %")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return action
    
    self.logger.debug("Querying model for action.")
    features = state_to_features(self, game_state)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        action_logits = self.model(features_tensor)
        action_idx = torch.argmax(action_logits, dim=-1).item()
    action = ACTIONS[action_idx]
    return action


# Convert the game state into a multi-channel feature representation.
def state_to_features(self, game_state: dict) -> np.ndarray:
    if game_state is None:
        return None
    
    field_channel = np.copy(game_state['field']) 
    bomb_map = np.zeros_like(game_state['field'])
    for (x, y), t in game_state['bombs']:
        bomb_map[x, y] = t

    explosion_map = np.copy(game_state['explosion_map'])
    coin_map = np.zeros_like(game_state['field'])
    for (x, y) in game_state['coins']:
        coin_map[x, y] = 1

    self_pos_channel = np.zeros_like(game_state['field'])
    self_x, self_y = game_state['self'][3]
    self_pos_channel[self_x, self_y] = 1

    opp_pos_channel = np.zeros_like(game_state['field'])
    for opponent in game_state['others']:
        opp_x, opp_y = opponent[3]
        opp_pos_channel[opp_x, opp_y] = 1

    can_bomb_channel = np.ones_like(game_state['field']) * int(game_state['self'][2])

    opp_can_bomb_channel = np.zeros_like(game_state['field'])
    for opponent in game_state['others']:
        opp_x, opp_y = opponent[3]
        opp_can_bomb_channel[opp_x, opp_y] = int(opponent[2])

    multi_channel_grid = np.stack((
        field_channel, bomb_map, explosion_map, coin_map, self_pos_channel,
        opp_pos_channel, can_bomb_channel, opp_can_bomb_channel
    ), axis=-1)

    return multi_channel_grid