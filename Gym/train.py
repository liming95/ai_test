import pygame
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import threading
import queue

# -----------------------------
# 游戏参数
# -----------------------------
GRID_SIZE = 5
CELL_SIZE = 80
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
N_COINS = 2
MAX_STEPS = 50
ACTIONS = [0, 1, 2, 3]  # 上，下，左，右

# -----------------------------
# 游戏环境
# -----------------------------
class GridCollectEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Collect Coins")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.coins = []
        while len(self.coins) < N_COINS:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if pos != tuple(self.agent_pos) and pos not in self.coins:
                self.coins.append(pos)
        self.steps = 0
        return self.get_state()

    def get_state(self):
        if self.coins:
            closest = min(self.coins, key=lambda c: abs(c[0]-self.agent_pos[0])+abs(c[1]-self.agent_pos[1]))
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

        if self.coins:
            closest = min(self.coins, key=lambda c: abs(c[0]-x)+abs(c[1]-y))
            dist = abs(closest[0]-x) + abs(closest[1]-y)
            reward += 0.1 / (dist + 1)

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
        self.clock.tick(10)

# -----------------------------
# Q-learning智能体
# -----------------------------
class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.8):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            return int(np.argmax(self.get_q(state)))

    def update(self, state, action, reward, next_state):
        q = self.get_q(state)
        q_next = self.get_q(next_state)
        q[action] += self.alpha * (reward + self.gamma * np.max(q_next) - q[action])

    def save(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

# -----------------------------
# 收敛曲线线程
# -----------------------------
def plot_rewards(reward_queue):
    plt.ion()
    fig, ax = plt.subplots()
    rewards_history = []
    line, = ax.plot([], [], 'r-')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Convergence")
    plt.show()

    while True:
        try:
            reward = reward_queue.get(timeout=0.1)
            if reward is None:
                break
            rewards_history.append(reward)

            line.set_xdata(range(len(rewards_history)))
            line.set_ydata(rewards_history)

            # 横轴动态扩展
            ax.set_xlim(0, len(rewards_history)+1)
            ax.relim()
            ax.autoscale_view(True, True, True)

            fig.canvas.draw()
            fig.canvas.flush_events()
        except queue.Empty:
            continue
    plt.ioff()
    plt.show()

# -----------------------------
# 训练过程
# -----------------------------
def train(env, agent, episodes=2000):
    reward_queue = queue.Queue()

    # 启动绘图线程
    plot_thread = threading.Thread(target=plot_rewards, args=(reward_queue,), daemon=True)
    plot_thread.start()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    reward_queue.put(None)
                    exit()

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            env.render()

        reward_queue.put(total_reward)
        agent.epsilon = max(0.05, agent.epsilon*0.995)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    reward_queue.put(None)
    plot_thread.join()
    return

# -----------------------------
# 主程序
# -----------------------------
env = GridCollectEnv()
agent = QLearningAgent()
train(env, agent, episodes=2000)
agent.save("q_table.pkl")
print("训练完成并已保存 Q-table！")
