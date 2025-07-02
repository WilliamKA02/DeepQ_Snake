from main import *

env = Game(tick_rate=5)
env.reset()
run = True
action = "right"
done = False

while run and not done:
    env.render()
    snake_pos, reward, done, apple, board, direction = env.step(action)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                run = False
            if event.key == pygame.K_UP:
                action = "up"
            if event.key == pygame.K_DOWN:
                action = "down"
            if event.key == pygame.K_RIGHT:
                action = "right"
            if event.key == pygame.K_LEFT:
                action = "left"
            if event.key == pygame.K_r:
                action = "right"
                env.reset()

env.close()