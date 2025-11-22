import random
import matplotlib.pyplot as plt
import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3 : (row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

    def get_state(self):
        return "".join(self.board)

    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None

class MenaceAgent:
    def __init__(self, letter='X', initial_beads=10):
        self.letter = letter
        self.matchboxes = {}
        self.initial_beads = initial_beads
        self.history = []

    def get_action(self, env):
        state = env.get_state()
        available_moves = env.available_moves()
        
        if state not in self.matchboxes:
            self.matchboxes[state] = []
            for move in available_moves:
                self.matchboxes[state].extend([move] * self.initial_beads)
        
        beads = self.matchboxes[state]
        
        if not beads:
             return random.choice(available_moves)

        action = random.choice(beads)
        self.history.append((state, action))
        return action

    def update(self, result):
        if result == 'win':
            delta = 3
        elif result == 'draw':
            delta = 1
        else:
            delta = -1
            
        for state, action in self.history:
            if state in self.matchboxes:
                if delta > 0:
                    self.matchboxes[state].extend([action] * delta)
                elif delta < 0:
                    if action in self.matchboxes[state]:
                        self.matchboxes[state].remove(action)
        
        self.history = []

class RandomAgent:
    def __init__(self, letter='O'):
        self.letter = letter

    def get_action(self, env):
        return random.choice(env.available_moves())

def play_game(p1, p2, env):
    env.reset()
    current_letter = 'X'
    
    while env.empty_squares():
        if current_letter == 'X':
            square = p1.get_action(env)
        else:
            square = p2.get_action(env)
            
        if env.make_move(square, current_letter):
            if env.current_winner:
                return current_letter 
            current_letter = 'O' if current_letter == 'X' else 'X'
    
    return 'Draw'

def train_menace(episodes=1000):
    env = TicTacToe()
    menace = MenaceAgent(letter='X')
    opponent = RandomAgent(letter='O')
    
    results = {'win': 0, 'loss': 0, 'draw': 0}
    history_win_rate = []
    block_size = 100
    block_wins = 0
    
    print(f"Training MENACE for {episodes} episodes...")
    
    for i in range(1, episodes + 1):
        winner = play_game(menace, opponent, env)
        
        if winner == 'X':
            menace.update('win')
            results['win'] += 1
            block_wins += 1
        elif winner == 'O':
            menace.update('loss')
            results['loss'] += 1
        else:
            menace.update('draw')
            results['draw'] += 1
            
        if i % block_size == 0:
            win_rate = block_wins / block_size
            history_win_rate.append(win_rate)
            print(f"Episode {i}: Win Rate (last {block_size}) = {win_rate:.2f}")
            block_wins = 0

    return history_win_rate

def plot_results(win_rates):
    plt.figure(figsize=(10, 6))
    plt.plot(range(100, (len(win_rates)+1)*100, 100), win_rates, marker='o')
    plt.title('MENACE Win Rate vs Random Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (per 100 games)')
    plt.grid(True)
    plt.savefig('../docs/menace_results.png')
    print("Plot saved to ../docs/menace_results.png")

if __name__ == "__main__":
    win_rates = train_menace(episodes=2000)
    plot_results(win_rates)
