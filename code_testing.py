import numpy as np
import torch

board = np.zeros((20,20))
apple = [5,5]
snake_head = [10, 10]
snake_pos = [[i, 10] for i in range(5,11)]

for pos in snake_pos[:-1]:
    board[pos[0], pos[1]] = 1
board[snake_head[0], snake_head[1]] = 5
board[apple[0], apple[1]] = 10


a = np.array([0, 1, 0, 1], dtype=bool)
b = np.array([9, 9, 9, 9])
b[a] = 0
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

A = np.array([[0, 2], [0, 1], [5, -1]])
print(np.max(A, axis=1, keepdims=True))

board = np.zeros((10,10))
apple = [2,2]
snake_head = [5, 5]
snake_pos = [[i, 5] for i in range(1,6)]

for pos in snake_pos[:-1]:
    board[pos[0], pos[1]] = -1
board[snake_head[0], snake_head[1]] = 2
board[apple[0], apple[1]] = 5
state_ = np.array(board, dtype=np.float32)
state_ = np.reshape(state_, (1, 10, 10))
state_normalized = state_ / 5
print(state_normalized)
print(state_normalized == 0)
print(snake_pos)
available_positions = [[i, j] for i in range(10) for j in range(10) if [i, j] not in snake_pos]
# for i in range(10):
#     for j in range(10):
#         if [i, j] not in snake_pos:
#             available_positions.append([i, j])
print(available_positions)
apple_pos = available_positions[np.random.choice(len(available_positions))]
print(apple_pos)
