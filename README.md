# Assignment 2 — Tabular Q-Learning on Taxi-v3
CSCN8020 — Reinforcement Learning

Student: Abdala
Environment: Taxi-v3
Algorithm: Tabular Q-Learning

# Project Overview

This project implements Tabular Q-Learning to solve the Taxi-v3 reinforcement learning environment using the Gymnasium library.

The Taxi agent must:

Navigate a 5×5 grid

Pick up a passenger

Drop the passenger at the correct destination

Minimize penalties and unnecessary movements

The implementation includes:

Baseline training configuration

Hyperparameter experiments (α and ε)

Automatic best-model selection

Policy re-run validation

Moving average plots

Action logging (CSV)

Rendered episode trace export (CSV)

# Q-Learning Background

The Q-Learning update rule used is:

    𝑄(𝑠,𝑎) ← 𝑄(𝑠,𝑎) + 𝛼 [𝑟 + 𝛾max𝑎′𝑄(𝑠′𝑎′)−𝑄(𝑠,𝑎)]

Where:
α (alpha) → Learning rate

γ (gamma) → Discount factor

ε (epsilon) → Exploration rate

An ε-greedy policy is used for exploration.

# Project Structure

assignment2_taxi_qlearning.py   # Main experiment runner
assignment2_utils.py            # Environment utilities & rendering
assignment2_outputs/            # Generated plots
README.md  

# Installation
1. Create Virtual Environment (Recommended)
python -m venv .venv
.venv\Scripts\activate
2. Install Dependencies
pip install gymnasium
pip install matplotlib
pip install numpy

# Running the Project

Default Run (All Experiments)

python assignment2_taxi_qlearning.py

This will:
* Train baseline configuration
* Run alpha experiments
* Run epsilon experiments
* Select best configuration
* Re-run best configuration
* Save plots to assignment2_outputs/

# Output Plots
Generated in:

assignment2_outputs/
* returns_moving_avg.png
* steps_moving_avg.png

These show:
* Moving average return per episode
* Moving average steps per episode
* Convergence behavior
* Hyperparameter sensitivity

# Hyperparameter Experiments
Baseline
α = 0.1
ε = 0.1
γ = 0.9

Alpha Experiments
α = 0.01
α = 0.001
α = 0.2

Epsilon Experiments
ε = 0.2
ε = 0.3

The best configuration is selected based on:
Highest average return
Lowest average steps

# Render Best Learned Policy
python assignment2_taxi_qlearning.py --render-best

# Render multiple episodes:
python assignment2_taxi_qlearning.py --render-best --render-episodes 3

# Log Training Actions (Detailed Q-Update Logging)

To record state-action updates during training:
python assignment2_taxi_qlearning.py --log-actions
Optional controls:

--log-actions-episodes 2

--log-actions-max-steps 200

This generates CSV files containing:
State, Action, Reward, Q-value before update, Q-value after update, TD target, TD error and Epsilon used
Useful for debugging and analysis.

# Key Observations

Larger α (0.2) converges faster than smaller values.
Extremely small α slows learning dramatically.
High ε increases randomness and degrades performance.
Balanced exploration (ε = 0.1) yields stable convergence.
Best configuration converges to ~19 steps per episode.