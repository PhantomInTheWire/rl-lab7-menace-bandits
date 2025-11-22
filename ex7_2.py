import numpy as np
import matplotlib.pyplot as plt

class BinaryBandit:
    def __init__(self, p):
        self.p = p # Probability of success (reward=1)

    def pull(self):
        return 1 if np.random.random() < self.p else 0

class EpsilonGreedyAgent:
    def __init__(self, epsilon, n_actions):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)

    def get_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Break ties randomly
            max_q = np.max(self.q_values)
            actions_with_max_q = [a for a in range(self.n_actions) if self.q_values[a] == max_q]
            return np.random.choice(actions_with_max_q)

    def update(self, action, reward):
        self.action_counts[action] += 1
        # Sample average update: Q(a) <- Q(a) + 1/N(a) * (R - Q(a))
        self.q_values[action] += (1.0 / self.action_counts[action]) * (reward - self.q_values[action])

def run_experiment(steps=2000, epsilon=0.1):
    # Two bandits: A and B
    # Let's assume p_A = 0.7, p_B = 0.3 (or similar distinct values)
    bandit_a = BinaryBandit(p=0.7)
    bandit_b = BinaryBandit(p=0.3)
    bandits = [bandit_a, bandit_b]
    
    agent = EpsilonGreedyAgent(epsilon=epsilon, n_actions=2)
    
    rewards = []
    optimal_action_counts = []
    
    optimal_action = 0 # Since 0.7 > 0.3
    
    for i in range(steps):
        action = agent.get_action()
        reward = bandits[action].pull()
        agent.update(action, reward)
        
        rewards.append(reward)
        optimal_action_counts.append(1 if action == optimal_action else 0)
        
    return rewards, optimal_action_counts

def plot_results(rewards, optimal_action_counts, epsilon):
    # Moving average for rewards
    window = 50
    avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    avg_optimal = np.convolve(optimal_action_counts, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards)
    plt.title(f'Average Reward (Moving Avg {window}) - Epsilon={epsilon}')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_optimal)
    plt.title(f'% Optimal Action (Moving Avg {window})')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('binary_bandit_results.png')
    print("Plot saved to binary_bandit_results.png")

if __name__ == "__main__":
    print("Running Binary Bandit Experiment...")
    rewards, optimal_action_counts = run_experiment(steps=2000, epsilon=0.1)
    plot_results(rewards, optimal_action_counts, epsilon=0.1)
    print("Done.")
