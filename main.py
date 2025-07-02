import pygame
import numpy as np

class Snake():
    """
    Class representing the Snake in the Snake game.
    Handles the snake's position, movement, and growth.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        Resets the snake to its initial position and direction.
        """
        self.head = [5, 5]
        self.tail = [1, 5]
        self.pos = [[i, 5] for i in range(1, 6)]
        self.direction = "right"
        self.length = 5
    
    def step(self, action):
        """
        Moves the snake one step in the given direction.

        Args:
            action (str): "up", "down", "left", or "right".
        """
        head = (self.head).copy()
        if action == "up":
            if self.direction != "down":
                head[1] -= 1
                self.direction = action
            else:
                head[1] += 1
        elif action == "down":
            if self.direction != "up":
                head[1] += 1
                self.direction = action
            else:
                head[1] -= 1
        elif action == "right":
            if self.direction != "left":
                head[0] += 1
                self.direction = action
            else:
                head[0] -= 1
        elif action == "left":
            if self.direction != "right":
                head[0] -= 1
                self.direction = action
            else:
                head[0] += 1
        self.pos.pop(0)
        self.pos.append(head)
        self.tail, self.head = self.pos[0], self.pos[-1]

    def extend(self, current_tail):
        """
        Extends the snake's body by adding a new segment at the tail.

        Args:
            current_tail (list): The tail position to reattach as the new segment.
        """
        self.tail = current_tail
        new_pos = [self.tail] + self.pos
        self.pos = new_pos
        self.length += 1



class Game():
    """
    Class managing the overall Snake game environment, including game logic,
    rendering, state updates, and interaction with the Snake object.
    """
    def __init__(self, tick_rate):
        """
        Initializes the game environment and sets the tick rate.

        Args:
            tick_rate (int): Frame rate for game rendering.
        """
        pygame.init()
        self.snake = Snake()
        self.clock = pygame.time.Clock()
        self.fps = tick_rate
        self.reset()
    
    def reset(self):
        """
        Resets the game state, snake position, apple location, and board layout.
        """

        self.game_over = False
        self.won = False
        self.rendering = False
        self.snake.reset()
        self.apple = [2, 2]
        self.board = np.zeros((10,10))
        for pos in self.snake.pos[:-1]:
            self.board[pos[1], pos[0]] = -0.2
        self.board[5, 5] = 0.4
        self.board[2,2] = 1
    
    def step(self, action):
        """
        Executes one step of the game logic based on the provided action.

        Args:
            action (str): "up", "down", "left", or "right".

        Returns:
            tuple: A tuple containing:
                - snake position (list of lists)
                - reward (float)
                - game over flag (bool)
                - apple position (list)
                - board state (np.ndarray)
                - current snake direction (str)
        """

        reward = 0
        current_tail = self.snake.tail.copy()

        if not self.game_over:
            self.snake.step(action) # Updates the Snakes position (self.snake.pos)
        
        if self.snake.head == self.apple: # Check if the Snake eats an apple
            self.snake.extend(current_tail) # Adds the tail, that the Snake had before taking a step
            reward += 5
            if self.snake.length == 100: # Check if game is over
                self.game_over = True
                self.won = True
                reward += 100
            else:
                available_positions = [[i, j] for i in range(10) for j in range(10) if [i, j] not in self.snake.pos] # Finds all available positions on the board (no snake)
                self.apple = available_positions[np.random.choice(len(available_positions))] # Selects random available position
        elif (self.snake.head in self.snake.pos[:-1]) or (self.snake.head[0] > 9 or self.snake.head[0] < 0) or (self.snake.head[1] > 9 or self.snake.head[1] < 0): # Checks if snake head hits wall or body
            self.game_over = True
            reward -= 1
        
        snake_pos = self.snake.pos[:-1].copy()
        head = self.snake.head.copy()
        apple = self.apple.copy()
        if not self.game_over:
            board = np.zeros((10,10))
            for pos in snake_pos:
                board[pos[1], pos[0]] = -0.2
            board[head[1], head[0]] = 0.4
            board[apple[1], apple[0]] = 1
            self.board = board
        
        return self.snake.pos, reward, self.game_over, self.apple, self.board, self.snake.direction
    
    def init_render(self):
        """
        Initializes the rendering window using Pygame.
        """
        self.screen = pygame.display.set_mode([1000, 1000])
        pygame.display.set_caption("Snake game")
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True

    def render(self):
        """
        Renders the current game state to the screen using Pygame.
        """
        if not self.rendering:
            self.init_render()
        
        self.clock.tick(self.fps)

        self.screen.fill((0,0,0))

        for pos in self.snake.pos:
            pygame.draw.rect(self.screen, "white", [pos[0]*100, pos[1]*100, 100, 100])
        
        pygame.draw.rect(self.screen, "red", [self.apple[0]*100, self.apple[1]*100, 100, 100])

        pygame.display.flip()

    def close(self):
        pygame.quit()