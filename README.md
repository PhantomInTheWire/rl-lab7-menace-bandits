# RL Lab 7: MENACE and Multi-Armed Bandits

This repository contains implementations for Lab Assignment 7 on Reinforcement Learning, covering MENACE (Tic-Tac-Toe) and Multi-Armed Bandit problems.

## Repository
🔗 **GitHub**: [https://github.com/PhantomInTheWire/rl-lab7-menace-bandits](https://github.com/PhantomInTheWire/rl-lab7-menace-bandits)

## Authors
- **Pawan Meena** - 202351102@iiitvadodara.ac.in
- **Solanki Kuldipkumar Kishorbhai** - 202351136@iiitvadodara.ac.in
- **Karan Haresh Lokchandani** - 202351055@iiitvadodara.ac.in

*Computer Science and Engineering, IIIT Vadodara*

## Files

### Implementations
- **ex7_1.py** - MENACE (Matchbox Educable Naughts and Crosses Engine) for Tic-Tac-Toe
- **ex7_2.py** - Binary Bandit with Epsilon-Greedy algorithm
- **ex7_3.py** - 10-Armed Non-Stationary Bandit setup
- **ex7_4.py** - Comparison of Standard vs Modified Epsilon-Greedy agents

### Visualizations
- **menace_results.png** - MENACE win rate over episodes
- **binary_bandit_results.png** - Binary bandit performance metrics
- **bandit_means.png** - Random walk of 10-armed bandit means
- **non_stationary_comparison.png** - Agent comparison results

### Documentation
- **report.tex** - IEEE conference paper format report

## Running the Code

```bash
# Install dependencies
pip3 install matplotlib numpy

# Run experiments
python3 ex7_1.py  # MENACE
python3 ex7_2.py  # Binary Bandit
python3 ex7_3.py  # Non-Stationary Bandit
python3 ex7_4.py  # Agent Comparison

# Compile LaTeX report
pdflatex report.tex
```

## Results Summary

- **MENACE**: Successfully learns to play Tic-Tac-Toe, achieving ~60% win rate against random opponent
- **Binary Bandit**: Converges to optimal arm with 90% selection rate
- **Non-Stationary Bandit**: Modified agent (constant step-size) significantly outperforms standard agent in tracking changing rewards
