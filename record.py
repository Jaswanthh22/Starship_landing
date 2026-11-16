# argparse handles the model name, difficulty, and number of episodes to record.
import argparse

import gymnasium as gym
from stable_baselines3 import PPO

import catch_env  # noqa: F401


def main():
    # Build a pixel-rendering environment and wrap it to save every episode as video.
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="profile")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = gym.make("StarshipCatch-v0", render_mode="rgb_array", difficulty=args.difficulty)
    env = gym.wrappers.RecordVideo(env, f"videos_{args.name}", episode_trigger=lambda i: True)
    # Load the requested model and run deterministic actions through each episode.
    try:
        model = PPO.load(f"ppo_{args.name}")
        for _ in range(args.episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
    finally:
        env.close()
    print(f"Videos saved in videos_{args.name}/")


if __name__ == "__main__":
    # Usage: python record.py profile --difficulty 0.5
    main()
