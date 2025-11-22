import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.means = np.zeros(k)
        self.time_step = 0

    def step(self):
        self.means += np.random.normal(0, 0.01, self.k)
        self.time_step += 1

    def pull(self, action):
        return np.random.normal(self.means[action], 1.0)

def visualize_random_walk(steps=1000):
    bandit = NonStationaryBandit(k=10)
    history = np.zeros((steps, 10))
    
    for i in range(steps):
        history[i] = bandit.means
        bandit.step()
        
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(history[:, i], label=f'Arm {i}')
        
    plt.title('Random Walk of 10-Armed Bandit Means')
    plt.xlabel('Steps')
    plt.ylabel('Mean Reward')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig('../docs/bandit_means.png')
    print("Plot saved to ../docs/bandit_means.png")

if __name__ == "__main__":
    visualize_random_walk(steps=2000)
