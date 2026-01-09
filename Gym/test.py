import pygame
import numpy as np
import random
import pickle
import time

# -----------------------------
# 测试参数（比训练复杂）
# -----------------------------
GRID_SIZE = 7          # 比训练大
CELL_SIZE = 80
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
N_COINS = 4            # 比训练多
MAX_STEPS = 80
ACTIONS = [0, 1, 2, 3]  # 上 下 左 右

# -----------------------------
# 测试环境（与训练几乎一致）
# -----------------------------
class GridCollectEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("TEST: Collect Coins")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.coins = []
        while len(self.coins) < N_COINS:
            pos = (random.randint(0, GRID_SIZE-1),
                   random.randint(0, GRID_SIZE-1))
            if pos != tuple(self.agent_pos) and pos not in self.coins:
                self.coins.append(pos)
        self.steps = 0
        return self.get_state()

    def get_state(self):
        if self.coins:
            closest = min(
                self.coins,
                key=lambda c: abs(c[0]-self.agent_pos[0]) + abs(c[1]-self.agent_pos[1])
            )
            dx = closest[0] - self.agent_pos[0]
            dy = closest[1] - self.agent_pos[1]
        else:
            dx, dy = 0, 0
        return (dx, dy)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < GRID_SIZE-1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < GRID_SIZE-1: x += 1
        self.agent_pos = [x, y]

        reward = -0.01
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
                rect = pygame.Rect(i*CELL_SIZE, j*CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (50,50,50), rect, 1)

        for (cx, cy) in self.coins:
            pygame.draw.circle(
                self.screen, (255, 215, 0),
                (cx*CELL_SIZE + CELL_SIZE//2,
                 cy*CELL_SIZE + CELL_SIZE//2),
                CELL_SIZE//4
            )

        ax, ay = self.agent_pos
        pygame.draw.rect(
            self.screen, (0, 200, 0),
            (ax*CELL_SIZE+10, ay*CELL_SIZE+10,
             CELL_SIZE-20, CELL_SIZE-20)
        )

        pygame.display.flip()
        self.clock.tick(8)

# -----------------------------
# 测试智能体（无探索）
# -----------------------------
class QAgentTest:
    def __init__(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        if state not in self.q_table:
            return random.choice(ACTIONS)
        return int(np.argmax(self.q_table[state]))

# -----------------------------
# 测试主循环
# -----------------------------
def test(agent, env, episodes=10):
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        print(f"\n=== Test Episode {ep+1} ===")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward

            env.render()

        print(f"Episode {ep+1} finished | Total reward = {total_reward:.2f}")
        time.sleep(1)

# -----------------------------
# 入口
# -----------------------------
if __name__ == "__main__":
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    env = GridCollectEnv()
    agent = QAgentTest(q_table)

    test(agent, env, episodes=10)
