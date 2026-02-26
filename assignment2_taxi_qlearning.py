#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from assignment2_utils import describe_env, simulate_episodes


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class RunResult:
    name: str
    alpha: float
    epsilon: float
    gamma: float
    episodes: int
    steps_per_episode: List[int]
    returns_per_episode: List[float]
    q_table: np.ndarray

    @property
    def avg_return(self) -> float:
        return float(np.mean(self.returns_per_episode))

    @property
    def avg_steps(self) -> float:
        return float(np.mean(self.steps_per_episode))


# ----------------------------
# Helpers
# ----------------------------
def moving_average(x: List[float], window: int = 100) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window, dtype=np.float32) / window, mode="valid")


def save_plots(results: List[RunResult], out_dir: str, window: int = 100) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Returns plot
    plt.figure()
    for r in results:
        plt.plot(moving_average(r.returns_per_episode, window), label=f"{r.name} (avgR={r.avg_return:.2f})")
    plt.title(f"Taxi-v3 Q-Learning: Moving Avg Return (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "returns_moving_avg.png"), dpi=200)
    plt.close()

    # Steps plot
    plt.figure()
    for r in results:
        plt.plot(moving_average(r.steps_per_episode, window), label=f"{r.name} (avgSteps={r.avg_steps:.1f})")
    plt.title(f"Taxi-v3 Q-Learning: Moving Avg Steps/Episode (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "steps_moving_avg.png"), dpi=200)
    plt.close()


def pick_best(results: List[RunResult]) -> RunResult:
    # Best = highest avg_return, tie-breaker lowest avg_steps
    return sorted(results, key=lambda r: (-r.avg_return, r.avg_steps))[0]


def make_actions_log_path(out_dir: str, run_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = run_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    return os.path.join(out_dir, f"actions_{safe}_{ts}.csv")


# ----------------------------
# Q-Learning Agent
# ----------------------------
class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float, epsilon: float, seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = random.Random(seed)

    def select_action(self, state: int, epsilon_override: Optional[float] = None) -> int:
        eps = self.epsilon if epsilon_override is None else float(epsilon_override)
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        return int(np.argmax(self.q[state]))

    def train(
        self,
        env: gym.Env,
        episodes: int,
        max_steps: int,
        epsilon_decay: bool,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        show_params_every: int = 0,
        # Action logging
        log_actions: bool = False,
        log_path: str = "",
        log_actions_episodes: int = 1,
        log_actions_max_steps: int = 200,
    ) -> Tuple[List[int], List[float]]:
        steps_hist: List[int] = []
        returns_hist: List[float] = []

        writer = None
        csv_f = None
        if log_actions:
            if not log_path:
                raise ValueError("log_actions=True requires log_path")
            csv_f = open(log_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(csv_f)
            writer.writerow([
                "episode", "t", "state", "action", "reward", "next_state", "done",
                "epsilon", "q_before", "q_after", "td_target", "td_error"
            ])

        for ep in range(1, episodes + 1):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            # current epsilon (if using decay)
            if epsilon_decay:
                eps_cur = max(eps_end, eps_start * (eps_decay ** (ep - 1)))
            else:
                eps_cur = self.epsilon

            if show_params_every > 0 and ep % show_params_every == 0:
                print(f"[episode {ep:5d}] gamma={self.gamma:.3f} alpha={self.alpha:.3f} eps={eps_cur:.3f}")

            while not done and steps < max_steps:
                action = self.select_action(state, epsilon_override=eps_cur)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Q-learning update
                q_before = float(self.q[state, action])
                best_next = float(np.max(self.q[next_state]))
                td_target = float(reward) + (0.0 if done else self.gamma * best_next)
                td_error = td_target - q_before
                self.q[state, action] = q_before + self.alpha * td_error
                q_after = float(self.q[state, action])

                # Log actions for first N episodes (to avoid huge files)
                if writer is not None and ep <= log_actions_episodes and steps < log_actions_max_steps:
                    writer.writerow([
                        ep, steps, state, action, float(reward), next_state, int(done),
                        float(eps_cur), q_before, q_after, float(td_target), float(td_error)
                    ])

                state = next_state
                total_reward += float(reward)
                steps += 1

            steps_hist.append(steps)
            returns_hist.append(total_reward)

        if csv_f is not None:
            csv_f.close()

        return steps_hist, returns_hist


# ----------------------------
# Experiment runner
# ----------------------------
def run_one(
    name: str,
    episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    max_steps: int,
    seed: int,
    epsilon_decay: bool,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    show_params_every: int,
    # Logging config
    log_actions: bool,
    log_actions_episodes: int,
    log_actions_max_steps: int,
    out_dir: str,
) -> RunResult:
    env = gym.make("Taxi-v3")
    n_states, n_actions = describe_env(env)

    agent = QLearningAgent(n_states=n_states, n_actions=n_actions, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)

    log_path = make_actions_log_path(out_dir, name) if log_actions else ""

    steps_hist, returns_hist = agent.train(
        env=env,
        episodes=episodes,
        max_steps=max_steps,
        epsilon_decay=epsilon_decay,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        show_params_every=show_params_every,
        log_actions=log_actions,
        log_path=log_path,
        log_actions_episodes=log_actions_episodes,
        log_actions_max_steps=log_actions_max_steps,
    )

    if log_actions:
        print(f"[log] Saved training action log: {log_path}")

    env.close()

    return RunResult(
        name=name,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        episodes=episodes,
        steps_per_episode=steps_hist,
        returns_per_episode=returns_hist,
        q_table=agent.q.copy(),
    )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Assignment 2: Tabular Q-Learning on Taxi-v3 (with logs + traces)")

    parser.add_argument("--episodes", type=int, default=10000, help="Training episodes per run")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Baseline hyperparameters (as per assignment)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (baseline)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate (baseline)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (baseline)")

    # Optional epsilon decay (off by default)
    parser.add_argument("--epsilon-decay", action="store_true", help="Enable epsilon decay during training")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon if decay enabled")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Minimum epsilon if decay enabled")
    parser.add_argument("--eps-decay", type=float, default=0.995, help="Geometric decay rate if decay enabled")

    parser.add_argument("--out-dir", type=str, default="assignment2_outputs", help="Where to save plots/logs")
    parser.add_argument("--ma-window", type=int, default=100, help="Moving average window for plots")
    parser.add_argument("--show-params-every", type=int, default=0,
                        help="If >0, print gamma/alpha/epsilon used every N episodes")

    # Render best policy (greedy)
    parser.add_argument("--render-best", action="store_true",
                        help="Render greedy episodes using the best trained Q-table")
    parser.add_argument("--render-episodes", type=int, default=3, help="How many episodes to render")
    parser.add_argument("--render-sleep", type=float, default=0.15, help="Seconds to sleep between rendered steps")

    # Render trace CSV (table file)
    parser.add_argument("--render-trace", action="store_true",
                        help="Save a CSV trace for the rendered episodes (step-by-step table)")

    # Training action logs
    parser.add_argument("--log-actions", action="store_true",
                        help="Save a CSV action log during training (first N episodes only)")
    parser.add_argument("--log-actions-episodes", type=int, default=1,
                        help="How many initial episodes to log actions for (default=1)")
    parser.add_argument("--log-actions-max-steps", type=int, default=200,
                        help="Max steps per logged episode (default=200)")

    args = parser.parse_args()

    # ----- Required runs per assignment -----
    baseline = run_one(
        name="baseline",
        episodes=args.episodes,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        max_steps=args.max_steps,
        seed=args.seed,
        epsilon_decay=args.epsilon_decay,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        show_params_every=args.show_params_every,
        log_actions=args.log_actions,
        log_actions_episodes=args.log_actions_episodes,
        log_actions_max_steps=args.log_actions_max_steps,
        out_dir=args.out_dir,
    )

    # Alpha experiments (change alpha only)
    alpha_runs = [
        run_one("alpha=0.01", args.episodes, 0.01, args.epsilon, args.gamma, args.max_steps, args.seed,
                args.epsilon_decay, args.eps_start, args.eps_end, args.eps_decay, args.show_params_every,
                args.log_actions, args.log_actions_episodes, args.log_actions_max_steps, args.out_dir),
        run_one("alpha=0.001", args.episodes, 0.001, args.epsilon, args.gamma, args.max_steps, args.seed,
                args.epsilon_decay, args.eps_start, args.eps_end, args.eps_decay, args.show_params_every,
                args.log_actions, args.log_actions_episodes, args.log_actions_max_steps, args.out_dir),
        run_one("alpha=0.2", args.episodes, 0.2, args.epsilon, args.gamma, args.max_steps, args.seed,
                args.epsilon_decay, args.eps_start, args.eps_end, args.eps_decay, args.show_params_every,
                args.log_actions, args.log_actions_episodes, args.log_actions_max_steps, args.out_dir),
    ]

    # Exploration experiments (change epsilon only)
    epsilon_runs = [
        run_one("eps=0.2", args.episodes, args.alpha, 0.2, args.gamma, args.max_steps, args.seed,
                args.epsilon_decay, args.eps_start, args.eps_end, args.eps_decay, args.show_params_every,
                args.log_actions, args.log_actions_episodes, args.log_actions_max_steps, args.out_dir),
        run_one("eps=0.3", args.episodes, args.alpha, 0.3, args.gamma, args.max_steps, args.seed,
                args.epsilon_decay, args.eps_start, args.eps_end, args.eps_decay, args.show_params_every,
                args.log_actions, args.log_actions_episodes, args.log_actions_max_steps, args.out_dir),
    ]

    all_results = [baseline] + alpha_runs + epsilon_runs

    # ----- Summary -----
    print("\n=== Summary (training metrics) ===")
    for r in all_results:
        print(
            f"{r.name:10s} | alpha={r.alpha:<7} eps={r.epsilon:<5} gamma={r.gamma:<4} "
            f"| avg_return={r.avg_return:7.2f} | avg_steps={r.avg_steps:6.1f}"
        )

    # ----- Choose best and rerun -----
    best = pick_best(all_results)
    print(
        f"\nBest config (by avg_return then avg_steps): {best.name} "
        f"(alpha={best.alpha}, eps={best.epsilon}, gamma={best.gamma})"
    )

    best_rerun = run_one(
        name=f"best_rerun({best.name})",
        episodes=args.episodes,
        alpha=best.alpha,
        epsilon=best.epsilon,
        gamma=best.gamma,
        max_steps=args.max_steps,
        seed=args.seed + 1,
        epsilon_decay=args.epsilon_decay,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        show_params_every=args.show_params_every,
        log_actions=args.log_actions,
        log_actions_episodes=args.log_actions_episodes,
        log_actions_max_steps=args.log_actions_max_steps,
        out_dir=args.out_dir,
    )

    results_for_plots = all_results + [best_rerun]
    save_plots(results_for_plots, out_dir=args.out_dir, window=args.ma_window)

    print(f"\nSaved plots to: {args.out_dir}")
    print(" - returns_moving_avg.png")
    print(" - steps_moving_avg.png")

    # ----- Render best greedy policy + optional trace table -----
    if args.render_best:
        print("\nRendering greedy policy from best_rerun (epsilon forced to 0.0)...")
        env_render = gym.make("Taxi-v3", render_mode="human")

        class GreedyAgent:
            def __init__(self, q: np.ndarray):
                self.q = q

            def select_action(self, state: int) -> int:
                return int(np.argmax(self.q[state]))

        greedy_agent = GreedyAgent(best_rerun.q_table)

        trace_path = None
        if args.render_trace:
            os.makedirs(args.out_dir, exist_ok=True)
            trace_path = os.path.join(args.out_dir, "rendered_episodes_trace.csv")

        simulate_episodes(
            env_render,
            greedy_agent,
            num_episodes=args.render_episodes,
            out_path=trace_path,
            sleep=args.render_sleep,
            max_steps=args.max_steps,
        )

        if trace_path:
            print(f"[trace] Saved rendered episode trace: {trace_path}")

        env_render.close()


if __name__ == "__main__":
    main()