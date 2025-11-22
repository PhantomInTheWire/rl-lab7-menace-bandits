import numpy as np
import matplotlib.pyplot as plt
from ex7_3 import NonStationaryBandit

class StandardAgent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_values = np.zeros(k)
        self.action_counts = np.zeros(k)

    def get_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (1.0 / self.action_counts[action]) * (reward - self.q_values[action])

class ModifiedAgent:
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(k)

    def get_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

def run_comparison(steps=10000, runs=10):
    avg_rewards_std = np.zeros(steps)
    avg_rewards_mod = np.zeros(steps)
    optimal_action_std = np.zeros(steps)
    optimal_action_mod = np.zeros(steps)
    
    print(f"Running comparison for {steps} steps over {runs} runs...")
    
    for r in range(runs):
        bandit = NonStationaryBandit(k=10)
        std_agent = StandardAgent(k=10, epsilon=0.1)
        mod_agent = ModifiedAgent(k=10, epsilon=0.1, alpha=0.1)
        
        for i in range(steps):
            optimal_act = np.argmax(bandit.means)
            
            act_std = std_agent.get_action()
            rew_std = bandit.pull(act_std)
            std_agent.update(act_std, rew_std)
            
            act_mod = mod_agent.get_action()
            rew_mod = bandit.pull(act_mod)
            mod_agent.update(act_mod, rew_mod)
            
            avg_rewards_std[i] += rew_std
            avg_rewards_mod[i] += rew_mod
            if act_std == optimal_act:
                optimal_action_std[i] += 1
            if act_mod == optimal_act:
                optimal_action_mod[i] += 1
            
            bandit.step()
            
    avg_rewards_std /= runs
    avg_rewards_mod /= runs
    optimal_action_std /= runs
    optimal_action_mod /= runs
    
    return avg_rewards_std, avg_rewards_mod, optimal_action_std, optimal_action_mod

def plot_comparison(r_std, r_mod, o_std, o_mod):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(r_std, label='Standard Agent (Sample Avg)', alpha=0.7)
    plt.plot(r_mod, label='Modified Agent (Alpha=0.1)', alpha=0.7)
    plt.title('Average Reward over Time')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(o_std, label='Standard Agent', alpha=0.7)
    plt.plot(o_mod, label='Modified Agent', alpha=0.7)
    plt.title('% Optimal Action over Time')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/non_stationary_comparison.png')
    print("Plot saved to ../results/non_stationary_comparison.png")

if __name__ == "__main__":
    r_std, r_mod, o_std, o_mod = run_comparison(steps=10000, runs=20)
    plot_comparison(r_std, r_mod, o_std, o_mod)
