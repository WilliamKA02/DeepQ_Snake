# 🐍 Deep Q-Learning Snake Game

This project implements a version of the classic Snake game with an AI agent trained using **Deep Q-Learning** and a **Convolutional Neural Network (CNN)**. The agent learns to play the game by interacting with the environment, storing experiences, and improving over time. Specifically the agent learns the optimal Q-value for a given state-action pair using Bellman's optimality equation:

$$Q(s,a) = R(s,a) + \gamma \max_{a'}(Q(s', a))$$


## 🧠 Key Features

- Deep Q-Learning agent with a CNN architecture
- Custom 10x10 Snake environment built with Pygame
- Experience replay memory
- Target network for stable learning
- Epsilon-greedy action selection with decay

---

## 🗂️ Project Structure

```bash
CNN_Snake/
├── pretrained_models/
    └──CNNmodel.pth
├── main.py # Script containing game code
├── network.py # CNN model and reinforcement learning agent
├── play_game.py # Play snake game manually
├── README.md
├── requirements.txt # Python dependencies
├── test.py # Load pretrained models and test / train them further
└── train.py # Script to train agent
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/WilliamKA02/deepq-cnn-snake
cd deepq-cnn-snake
```

### 2. Install dependencies
python -m venv venv
```bash
conda create -n deepq_snake python=3.11
conda activate deepq_snake
pip install -r requirements.txt
```

## ⚙ Train / Test agent

### Training new model from scratch
In the train.py file, you can specify parameters for your agent and start training. In the network.py file, you can specify the path to the model name, where you want to save your trained model.

### Testing or finetuning pretrained model
A pretrained model is already available in "pretrained_models/CNN_model.pth". In the test.py file, you can sepcify the path to the location of the pretrained model you wish to load, and then test it or train it further (controlled by "train" bool). 