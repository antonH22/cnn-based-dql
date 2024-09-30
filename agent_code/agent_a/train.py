from collections import namedtuple, deque

import pickle
import random
from typing import List
import matplotlib.pyplot as plt 
from IPython import display

import events as e
import torch
import torch.nn as nn
import torch.optim as optim
from .callbacks import BombermanCNN, state_to_features
from .custom_events import *


# Used for plotting the scores during training
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Define the Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000  # Maximum number of transitions stored
BATCH_SIZE = 128  # Size of the mini-batch

TARGET_UPDATE_FREQUENCY = 100

# Hyperparameters
ALPHA = 0.001  # Learning rate
GAMMA = 0.95   # Discount factor

plot_maxlen = 100
plt.style.use('ggplot')

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after setup in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Set up the transition history buffer
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Setup device: Use GPU if available, otherwise use CPU
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.logger.info(f"Using device: {self.device}")

    # Setup the neural network and optimizer
    self.model = BombermanCNN()
    self.target_model = BombermanCNN()  # Use a target network for stability in learning
    self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
    self.criterion = nn.MSELoss()  # We'll use Mean Squared Error loss for Q-learning

    # Initialize learning rate scheduler
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Copy the model weights to the target model initially
    self.target_model.load_state_dict(self.model.state_dict())

    # Use for plotting
    self.recent_scores = deque(maxlen=plot_maxlen)
    self.plot_scores = []
    self.plot_mean_scores = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    This is one of the places where you could update your agent.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Add custom events here

    if self_action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        self.action_history.append(self_action)

    # Print the current action history
    self.logger.debug(f'Action history: {list(self.action_history)}')

    # Check for a loop in the action history
    if any(event.startswith('MOVE') for event in events):
        if len(self.action_history) == 3 and self.action_history[0] == self.action_history[2] and self.action_history[0] != self.action_history[1]:
            events.append(LOOP)
        else:
            events.append(NO_LOOP)

    # Convert state and next state to features
    old_state_features = state_to_features(self, old_game_state)
    new_state_features = state_to_features(self, new_game_state)
    
    # Compute the reward from event
    reward = reward_from_events(self, events)

    # Append the transition to the deque
    self.transitions.append(Transition(old_state_features, self_action, new_state_features, reward))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This is where you can update your model using the transitions collected.
    """

    # Add custom events here

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    score = last_game_state['self'][1]

    # Convert last game state to features and get reward
    last_state_features = state_to_features(self, last_game_state)
    reward = reward_from_events(self, events)
    reward += score

    self.logger.debug(f'Round score:{reward}')
    
    # Add final transition
    self.transitions.append(Transition(last_state_features, last_action, None, reward))

    # Perform model update using the transitions
    update_model(self)

    # Update target model periodically
    if last_game_state['round'] % TARGET_UPDATE_FREQUENCY == 0:
        update_target_model(self)

    # Save the model at the end of each round
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # plot results
    self.plot_scores.append(score)
    self.recent_scores.append(score)
    recent_mean_scores = sum(self.recent_scores) / len(self.recent_scores)
    self.plot_mean_scores.append(recent_mean_scores)
    plt.ion()
    plot(self.plot_scores, self.plot_mean_scores)


def update_model(self):
    """
    Updates the model based on a random mini-batch of transitions.
    """
    if len(self.transitions) < BATCH_SIZE:
        return  # Not enough transitions to train yet

    # Sample a random mini-batch of transitions
    mini_batch = random.sample(self.transitions, BATCH_SIZE)
    
    states, actions, next_states, rewards = zip(*mini_batch)

    # Convert the batch to tensors
    state_batch = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2)
    action_batch = torch.tensor([ACTIONS.index(action) for action in actions], dtype=torch.long)
    reward_batch = torch.tensor(rewards, dtype=torch.float32)

    # Filter out None values from next_states
    non_final_mask = torch.tensor([ns is not None for ns in next_states], dtype=torch.bool)
    non_final_next_states = torch.tensor([ns for ns in next_states if ns is not None], dtype=torch.float32).permute(0, 3, 1, 2)

    # Forward pass through the model
    q_values = self.model(state_batch)

    # Get Q-values for the actions that were actually taken
    state_action_values = q_values.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

    # Compute target Q-values using the target network
    next_q_values = torch.zeros(BATCH_SIZE, device=self.device)
    with torch.no_grad():
        next_q_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
    expected_q_values = reward_batch + (GAMMA * next_q_values)

    # Compute loss and backpropagate
    loss = self.criterion(state_action_values, expected_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

def update_target_model(self):
    """
    Copies the weights from the model to the target model.
    """
    self.target_model.load_state_dict(self.model.state_dict())
    self.logger.info("Updated target model weights.")


def reward_from_events(self, events: List[str]) -> float:
    """
    Compute the reward based on game events.
    """
    "scenario coin-heaven"
    "try to navigate the field and collect coins"
    game_rewards = {
        e.COIN_COLLECTED: +2,
        e.INVALID_ACTION: -1,
        e.WAITED: -1,
        e.BOMB_DROPPED: -1,
        LOOP: -1,
        NO_LOOP: +0.5
    }
    reward_sum = 0.0
    for event in events:
        reward_sum += game_rewards.get(event, 0)  # Default to 0 if event not in dict
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
