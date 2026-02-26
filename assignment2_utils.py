from __future__ import annotations

import csv
import os
import time
from typing import Tuple, Optional

import gymnasium as gym


def describe_env(env) -> Tuple[int, int]:
    """
    Prints details about the Taxi-v3 environment and returns:
    (num_states, num_actions)
    """
    print("Observation space: ", env.observation_space)
    print("Observation space size: ", env.observation_space.n)

    # Gymnasium wrappers may hide reward_range; keep safe fallback
    reward_range = getattr(env, "reward_range", None)
    if reward_range is None:
        reward_range = getattr(env.unwrapped, "reward_range", ("?", "?"))
    print("Reward Range: ", reward_range)

    print("Number of actions: ", env.action_space.n)

    # Taxi-v3 has fixed actions; hardcode for compatibility across versions
    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger",
    }
    print("Action description: ", action_desc)

    return env.observation_space.n, env.action_space.n


def simulate_episodes(
    env,
    agent,
    num_episodes: int = 3,
    out_path: Optional[str] = None,
    sleep: float = 0.15,
    max_steps: int = 200,
) -> None:
    """
    Simulates 'num_episodes' episodes using the given agent.

    - env should be created with render_mode="human" for visualization.
    - agent must implement: select_action(state) -> action int
    - If out_path is provided, writes a step-by-step CSV trace (table) for the rendered episodes.
    """
    writer = None
    csv_f = None

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        csv_f = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)
        writer.writerow([
            "episode", "t",
            "state", "action", "reward", "next_state", "done",
            "taxi_row", "taxi_col", "passenger_loc", "destination"
        ])

    # For readable prints
    loc_names = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue", 4: "In taxi"}

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        t = 0

        # Decode initial state (Taxi-v3 supports decode)
        tr, tc, pl, dest = env.unwrapped.decode(state)
        print(
            f"Passenger is at: {loc_names.get(pl, pl)}, wants to go to {loc_names.get(dest, dest)}. "
            f"Taxi currently at ({tr}, {tc})"
        )

        while not done and t < max_steps:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Action names
            action_desc = {
                0: "Move south",
                1: "Move north",
                2: "Move east",
                3: "Move west",
                4: "Pickup",
                5: "Dropoff",
            }

            # Decode current state
            tr, tc, pl, dest = env.unwrapped.decode(state)

            print(
                f"[Episode {ep} | Step {t}] "
                f"State={state} | Taxi=({tr},{tc}) | "
                f"Action={action} ({action_desc[action]}) | "
                f"Reward={reward}"
            )

            env.render()
            time.sleep(sleep)

            if writer is not None:
                writer.writerow([
                    ep, t,
                    state, action, float(reward), next_state, int(done),
                    tr, tc, pl, dest
                ])

            state = next_state
            t += 1

    if csv_f is not None:
        csv_f.close()