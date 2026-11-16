# Gymnasium provides the environment interface and random action sampler.
import gymnasium as gym

import catch_env  # noqa: F401


def main():
    # Open the simulation window and reset the environment to a moderate difficulty.
    env = gym.make("StarshipCatch-v0", render_mode="human", difficulty=0.5)
    try:
        obs, _ = env.reset()
        # Apply random controls for up to 500 frames, restarting after each episode ends.
        for _ in range(500):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            print(obs)  # dx, y, vx, vy, theta, omega, fuel
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
