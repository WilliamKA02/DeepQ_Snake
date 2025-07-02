from main import *
from network import *

env = Game(tick_rate=20)
env.reset()
actions = {0: "up", 1: "down", 2: "right", 3: "left"}
directions = {"up": 0, "down": 1, "right": 2, "left": 3}
agent = CNNAgent(gamma=0.99, epsilon=0.25, lr=5e-5, batch_size=32, max_mem=10000)
run = True
render = True
action = "right"
done = False
eps = True
steps = 0
games = 0

train = True # this bool controls whether the loaded model gets trained further or not

state_dict = T.load("pretrained_models/CNNmodel.pth")
agent.DeepQ.load_state_dict(state_dict)
agent.update_target_network()

if train:
    agent.DeepQ.train()
if not train:
    agent.DeepQ.eval()

while run:
    steps += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                render = not render
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                run = False
                print(f"Total steps: {steps} \n Total games: {games}")
                if train:
                    agent.save()
            if event.key == pygame.K_e:
                eps = not eps   
    if render:
        env.render()
    board = env.board.copy()
    state = np.array(board, dtype=np.float32)
    state_reshaped = np.reshape(state, (1, 10, 10))
    action = agent.choose_action(state=state_reshaped, eps=eps)
    snake_pos, reward, done, apple, new_board, direction = env.step(actions[int(action)])
    
    if train:
        next_state = np.array(new_board, dtype=np.float32)
        next_state_reshaped = np.reshape(next_state, (1, 10, 10))
        agent.store_memory(state=state_reshaped, action=action, next_state=next_state_reshaped, reward=reward, done=done)
        training_loss = agent.learn()
        if steps % 5000 == 0:
            print(f"Loss: {training_loss}, Epsilon:{agent.epsilon}")
            agent.update_target_network()
    else:
        if steps % 5000 == 0:
            print(f"Epsilon:{agent.epsilon}")
    if done:
        games += 1
        env.reset()
        continue

env.close()