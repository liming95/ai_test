# # Run `pip install "gymnasium[classic-control]"` for this example.
# import gymnasium as gym

# # Create our training environment - a cart with a pole that needs balancing
# env = gym.make("CartPole-v1", render_mode="human")

# # Reset environment to start a new episode
# observation, info = env.reset()
# # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# # info: extra debugging information (usually not needed for basic learning)

# print(f"Starting observation: {observation}")
# # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

# episode_over = False
# total_reward = 0

# while not episode_over:
#     # Choose an action: 0 = push cart left, 1 = push cart right
#     action = env.action_space.sample()  # Random action for now - real agents will be smarter!

#     # Take the action and see what happens
#     observation, reward, terminated, truncated, info = env.step(action)

#     # reward: +1 for each step the pole stays upright
#     # terminated: True if pole falls too far (agent failed)
#     # truncated: True if we hit the time limit (500 steps)

#     total_reward += reward
#     episode_over = terminated or truncated

# print(f"Episode finished! Total reward: {total_reward}")
# env.close()

import pygame
import numpy as np
import random
import time

GRID_SIZE = 5
CELL_SIZE = 80
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
N_COINS = 2  # 减少金币数量，让状态空间小一点
MAX_STEPS = 50
ACTIONS = [0, 1, 2, 3]

class GridCollectEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Collect Coins")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        # 固定金币数量
        self.coins = []
        while len(self.coins) < N_COINS:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if pos != tuple(self.agent_pos) and pos not in self.coins:
                self.coins.append(pos)
        self.steps = 0
        return self.get_state()

    def get_state(self):
        # 状态 = agent 坐标 + 每个金币是否被收集 (0/1)
        state = [self.agent_pos[0], self.agent_pos[1]]
        for coin in self.coins:
            state += [coin[0], coin[1]]
        return tuple(state)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < GRID_SIZE-1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < GRID_SIZE-1: x += 1
        self.agent_pos = [x, y]

        reward = 0
        new_coins = []
        for coin in self.coins:
            if tuple(self.agent_pos) == coin:
                reward += 1
            else:
                new_coins.append(coin)
        self.coins = new_coins

        self.steps += 1
        done = len(self.coins) == 0 or self.steps >= MAX_STEPS
        return self.get_state(), reward, done

    def render(self):
        self.screen.fill((30, 30, 30))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (50,50,50), rect, 1)
        for (cx, cy) in self.coins:
            pygame.draw.circle(self.screen, (255, 215, 0),
                               (cx*CELL_SIZE + CELL_SIZE//2, cy*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//4)
        ax, ay = self.agent_pos
        pygame.draw.rect(self.screen, (0, 200, 0),
                         (ax*CELL_SIZE+10, ay*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20))
        pygame.display.flip()
        self.clock.tick(5)

# -----------------------------
# Q-learning 表格
# -----------------------------
env = GridCollectEnv()
q_table = {}

alpha = 0.1
gamma = 0.99
epsilon = 0.3
EPISODES = 2000

def get_q(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    return q_table[state]

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(get_q(state))
        next_state, reward, done = env.step(action)
        q = get_q(state)
        q_next = get_q(next_state)
        q[action] = q[action] + alpha * (reward + gamma * np.max(q_next) - q[action])
        state = next_state
        total_reward += reward
    if episode % 200 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        epsilon = max(0.05, epsilon*0.99)  # epsilon 衰减

# -----------------------------
# 测试智能体
# -----------------------------
state = env.reset()
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    env.render()
    action = np.argmax(get_q(state))
    state, reward, done = env.step(action)

env.render()
time.sleep(2)
pygame.quit()
print("游戏结束！")

