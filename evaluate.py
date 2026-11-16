# argparse handles the model name and evaluation settings.
import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import catch_env  # noqa: F401


def main():
    # Parse options and create the unwrapped environment for measuring true task reward.
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="profile")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    env = gym.make("StarshipCatch-v0", difficulty=args.difficulty)  # no wrapper: shaping can't flatter the score
    # Load the trained policy and test it on a repeatable set of evaluation seeds.
    try:
        model = PPO.load(f"ppo_{args.name}")
        caught, rewards, fuel = 0, [], []
        # Run full episodes and collect catches, rewards, and remaining fuel.
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=10_000 + ep)  # fixed seeds, never seen in training order
            done, total = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                total += r
                done = term or trunc
            caught += info["caught"]
            rewards.append(total)
            fuel.append(info["fuel_left"])
        print(f"{args.name} @ difficulty {args.difficulty}: caught {caught}/{args.episodes} "
              f"({100 * caught / args.episodes:.0f}%), mean reward {np.mean(rewards):.1f}, "
              f"mean fuel left {100 * np.mean(fuel):.0f}%")
    finally:
        env.close()


if __name__ == "__main__":
    # Usage: python evaluate.py profile --difficulty 1.0
    main()
