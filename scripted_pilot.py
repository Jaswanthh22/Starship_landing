"""Hand-written suicide-burn controller. Not RL: it proves the task is solvable
and gives you a score to compare your trained agent against.

    python scripted_pilot.py            # success rate at a few difficulties
    python scripted_pilot.py --render   # watch it fly
"""
# argparse exposes options for watching the controller and choosing episode count.
import argparse
import math

import numpy as np
import gymnasium as gym

import catch_env
from catch_env import ALPHA_RCS, FUEL0, G, M_DRY, T_MAX

# Desired braking acceleration used to calculate target downward speed by altitude.
A_REF = 9.0


def pilot(obs):
    # Convert observations to physical units and estimate the booster mass.
    dx, y, vx, vy = obs[0] * 100, obs[1] * 500, obs[2] * 20, obs[3] * 50
    th, om, fuel = obs[4], obs[5], obs[6] * FUEL0
    mass = M_DRY + fuel

    # Use a braking-distance profile for vertical speed, then add feedback to throttle.
    v_ref = -(math.sqrt(2 * A_REF * max(y, 0.0)) + 0.5)       # speed we want at this height
    a_up = G + A_REF + 5.0 * (v_ref - vy)                  # feedforward + feedback on speed error
    throttle = np.clip(mass * a_up / (T_MAX * max(math.cos(th), 0.5)), 0.0, 1.0)

    # Proportional/derivative feedback steers sideways error and damps tilt/rotation.
    ax_des = -(0.05 * dx + 0.35 * vx)                          # sideways accel to null dx, vx
    th_des = np.clip(ax_des / G, -0.25, 0.25)
    alpha_des = -(2.0 * (th - th_des) + 2.8 * om)
    rcs = np.clip(alpha_des / ALPHA_RCS, -1.0, 1.0)
    return np.array([2 * throttle - 1, 0.0, rcs], dtype=np.float32)


def main():
    # Select difficulties and episode count; rendering mode uses a few visible flights.
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    # Run the hand-written pilot and count successful catches at each selected difficulty.
    for d in ([0.5] if args.render else [0.0, 0.5, 1.0]):
        env = gym.make("StarshipCatch-v0", difficulty=d,
                       render_mode="human" if args.render else None)
        caught = 0
        for ep in range(args.episodes if not args.render else 5):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(pilot(obs))
                done = term or trunc
            caught += info["caught"]
        env.close()
        n = args.episodes if not args.render else 5
        print(f"difficulty {d:.1f}: caught {caught}/{n}")


if __name__ == "__main__":
    main()
