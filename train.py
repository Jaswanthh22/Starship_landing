# argparse reads the model variant and training options from the command line.
import argparse
# deque stores a bounded window of recent catch results for curriculum decisions.
from collections import deque

# PPO is the reinforcement-learning algorithm used to train the policy.
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import catch_env  # noqa: F401  (registers StarshipCatch-v0)
from wrappers import Fixed, HoverHack, Profile, TimeCost

# Map selectable experiment names to reward wrappers; None means the raw environment.
variants = {"baseline": None, "timecost": TimeCost, "hover": HoverHack,
            "fixed": Fixed, "profile": Profile}


class Curriculum(BaseCallback):
    """Start with a wide start spread and generous arms; tighten once the agent
    catches `threshold` of its last `window` episodes."""

    def __init__(self, start=0.0, step=0.1, window=100, threshold=0.6):
        # Keep curriculum settings and recent outcomes for checking progress.
        super().__init__()
        self.difficulty, self.step_size = start, step
        self.threshold = threshold
        self.results = deque(maxlen=window)

    def _on_training_start(self):
        # Begin every parallel environment at the curriculum's starting difficulty.
        self.training_env.env_method("set_difficulty", self.difficulty)

    def _on_step(self):
        # Record finished episodes, raise difficulty after enough recent catches, and log it.
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.results.append(bool(info.get("caught", False)))
        full = len(self.results) == self.results.maxlen
        if full and sum(self.results) / len(self.results) >= self.threshold and self.difficulty < 1.0:
            self.difficulty = min(1.0, self.difficulty + self.step_size)
            self.training_env.env_method("set_difficulty", self.difficulty)
            self.results.clear()
        self.logger.record("curriculum/difficulty", self.difficulty)
        return True


def main():
    # Define the command-line interface and build eight seeded parallel environments.
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="profile", choices=list(variants))
    parser.add_argument("--steps", type=int, default=2_000_000)
    # Use multiple CPU worker processes to simulate independent episodes in parallel.
    parser.add_argument("--envs", type=int, default=8, help="number of parallel environments (try 8 on this laptop)")
    # Stable-Baselines3 selects CUDA automatically when a CUDA-enabled PyTorch is installed.
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="training device; auto uses the GPU when available")
    parser.add_argument("--no-curriculum", action="store_true", help="train at difficulty 1.0 from the start")
    args = parser.parse_args()

    # Subprocess workers let the CPU advance several simulation instances at once.
    # Spawn is the supported safe process-start method on Windows.
    env = SubprocVecEnv(
        [lambda rank=rank: make_env(rank, args.name) for rank in range(args.envs)],
        start_method="spawn",
    )
    # Train PPO, optionally using the curriculum callback, and save the resulting model.
    try:
        # Put PPO's neural network and optimizer on the selected device.
        model = PPO("MlpPolicy", env, verbose=1, seed=1, device=args.device)
        callback = None if args.no_curriculum else Curriculum()
        if args.no_curriculum:
            env.env_method("set_difficulty", 1.0)
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(f"ppo_{args.name}")
        print(f"Saved ppo_{args.name}.zip")
    finally:
        env.close()


def make_env(rank, name):
    # Factory called inside each worker: independently register and create one simulator.
    import gymnasium as gym
    import catch_env  # noqa: F401
    from stable_baselines3.common.monitor import Monitor

    env = gym.make("StarshipCatch-v0")
    wrapper = variants[name]
    if wrapper is not None:
        env = wrapper(env)
    return Monitor(env)


if __name__ == "__main__":
    # Usage: python train.py profile --steps 100000 --envs 8 --device auto
    main()
