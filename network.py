import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class CNN(nn.Module):
    """
    Convolutional Neural Network used for approximating Q-values in the Snake game.
    The network processes the board state and outputs Q-values for each possible action.
    """
    def __init__(self, lr):
        """
        Initializes the CNN model architecture and optimizer.

        Args:
            lr (float): Learning rate for the optimizer.
        """
        super(CNN, self).__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Model architecture of the CNN consisting of 3 convolutional layers followed by 2 linear layers
        self.cl1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=8, padding=1)
        self.cl2 = nn.Conv2d(kernel_size=3, in_channels=8, out_channels=16, padding=1)
        self.cl3 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32, padding=1)
        self.l1 = nn.Linear(in_features=3200, out_features=500)
        self.l2 = nn.Linear(in_features=500, out_features=4)

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass of the CNN.

        Args:
            state (torch.Tensor): Input tensor representing the game state (shape: [batch_size, 1, 10, 10]).

        Returns:
            torch.Tensor: Output tensor with Q-values for each of the 4 actions.
        """
        x = F.relu(self.cl1(state))
        x = F.relu(self.cl2(x))
        x = F.relu(self.cl3(x))
        if x.ndim == 4: # Check if input is a batch
            x = x.view(x.size(0), -1)  # Flatten keeping batch dimension
        else: # if input is a single state (no batch) flatten completely
            x = T.flatten(x)
        x = F.relu(self.l1(x))
        output = self.l2(x)
        return output


class CNNAgent():
    """
    Deep Q-Learning agent using a convolutional neural network to play the Snake game.
    Handles experience replay, target network updates, and epsilon-greedy action selection.
    """
    def __init__(self, gamma, epsilon, lr, batch_size, max_mem, eps_dec=0.99999, eps_min = 0.0):
        """
        Initializes the CNNAgent with hyperparameters and neural networks.

        Args:
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial epsilon value for exploration.
            lr (float): Learning rate.
            batch_size (int): Size of minibatches sampled from memory.
            max_mem (int): Maximum size of the replay buffer.
            eps_dec (float): Epsilon decay rate.
            eps_min (float): Minimum epsilon value.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_mem = max_mem
        self.lr = lr
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.DeepQ = CNN(lr=self.lr)
        self.targetQ = CNN(lr=self.lr)
        self.targetQ.load_state_dict(self.DeepQ.state_dict())
        self.targetQ.eval()
        self.memory = deque(maxlen=self.max_mem)
    
    def store_memory(self, state, action, next_state, reward, done):
        """
        Stores a single transition in the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            next_state (np.ndarray): Next state after action.
            reward (float): Reward received.
            done (bool): Whether the episode ended.
        """
        self.memory.append((state, action, next_state, reward, done))
    
    def choose_action(self, state, eps):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state as a (1, 10, 10) array.
            eps (bool): If True, allow exploration.

        Returns:
            int: Chosen action (0: up, 1: down, 2: left, 3: right).
        """
        # Take random action with probability epsilon
        if eps and random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        
        # Take action with highest Q-value from network output with probability (1 - epsilon)
        state = T.from_numpy(state)
        state = T.tensor(state, dtype=T.float).to(self.DeepQ.device)
        with T.no_grad():
            pred_output = self.DeepQ.forward(state)
            action = int(T.argmax(pred_output))

        return action
    
    def learn(self):
        """
        Samples a batch from memory and performs a learning step.
        Uses target network to compute target Q-values.

        Returns:
            float: The computed loss for the learning step. Returns 0 if not enough memory.
        """

        if len(self.memory) < self.batch_size:
            return 0

        # Collect a random sample from networks memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        # Convert to tensors
        state_batch = T.from_numpy(np.array(states)).to(self.DeepQ.device) # shape: (batch_size, 1, 10, 10)
        next_state_batch = T.from_numpy(np.array(next_states)).to(self.DeepQ.device) # shape: (batch_size, 1, 10, 10)
        done_batch = T.tensor(dones, dtype=T.bool).to(self.DeepQ.device) # shape: (batch_size)
        reward_batch = T.tensor(rewards, dtype=T.float).to(self.DeepQ.device) # shape: (batch_size)

        # Process state batch through network and collect Q-values for each state
        Q_s = self.DeepQ(state_batch) # shape: (batch_size, 4)

        # Gather the Q values which correspond to the chosen action
        Q_s_a = T.gather(Q_s, 1, T.tensor(actions).unsqueeze(1).to(self.DeepQ.device)).squeeze(1) # shape: (batch_size)
        
        # Calculate the target Q-values
        with T.no_grad():
            Q_s_next = self.targetQ(next_state_batch) # shape: (batch_size, 4)
            Q_s_next_max = T.max(Q_s_next, dim=1).values # shape: (batch_size)
            Q_s_next_max[done_batch] = 0 # if next state is terminal set Q value to 0
            target = reward_batch + self.gamma * Q_s_next_max # target tensor for computing the loss
        
        # Reset gradient, calculate the loss (between the networks output Q-values and target values) and update weights
        self.DeepQ.optimizer.zero_grad()
        loss = self.DeepQ.criterion(Q_s_a, target)
        loss.backward()
        self.DeepQ.optimizer.step()

        # Update epsilon
        self.epsilon = (self.epsilon-self.eps_min)*self.eps_dec + self.eps_min
        return loss.item()
    
    def update_target_network(self):
        """
        Updates the target network with the weights of the current network.
        """
        self.targetQ.load_state_dict(self.DeepQ.state_dict())
        self.targetQ.eval()
    
    def save(self):
        """
        Saves the current DeepQ model weights to a file.
        """
        T.save(self.DeepQ.state_dict(), "pretrained_models/trained_model.pth")