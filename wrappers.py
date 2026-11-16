# Math is used to build the altitude-dependent target speed in Profile.
import math

import gymnasium as gym

# Observation layout of StarshipCatch-v0 (same slots your LunarLander wrappers used):
#   0 dx/100   1 y/500   2 vx/20   3 vy/50   4 theta   5 omega   6 fuel fraction


class TimeCost(gym.Wrapper):
    """Small charge every frame so waiting costs something."""

    def step(self, action):
        # Pass the action through and subtract a tiny per-frame reward.
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r - 0.01, term, trunc, info


class HoverHack(gym.Wrapper):
    """The broken reward: +1 per frame for being upright and slow.
    Hovering out-earns catching, so training may learn to hover until time runs out."""

    def step(self, action):
        # Apply the base transition, then add reward for being upright and moving slowly.
        obs, r, term, trunc, info = self.env.step(action)
        if abs(obs[4]) < 0.2 and abs(obs[3]) < 0.3:  # upright and slow
            r += 1.0
        return obs, r, term, trunc, info


class Fixed(gym.Wrapper):
    """Your original fix: pay for improvement (change between frames), plus a
    small fee every frame."""

    def reset(self, **kwargs):
        # Reset the underlying environment and remember the initial shaping score.
        obs, info = self.env.reset(**kwargs)
        self.prev = self._phi(obs)
        return obs, info

    def _phi(self, obs):
        # Potential score favors being near the target, low, slow vertically, and upright.
        # "how good is this position": closer, lower, slower, more upright
        return -abs(obs[0]) - abs(obs[1]) - abs(obs[3]) - abs(obs[4])

    def step(self, action):
        # Reward improvement in potential since the previous frame and charge a small time cost.
        obs, r, term, trunc, info = self.env.step(action)
        phi = self._phi(obs)
        r += 5.0 * (phi - self.prev) - 0.02
        self.prev = phi
        return obs, r, term, trunc, info


class Profile(Fixed):
    """Same improvement-based shaping, but 'good speed' depends on altitude.
    Fixed rewards going slow at ANY height, which still tempts the agent to hover.
    Here the target is a braking profile: fast when high, ~0.5 m/s at the arms,
    so the cheapest way to score is a real suicide burn."""

    A_REF = 9.0  # m/s^2 braking the profile assumes

    def _phi(self, obs):
        # Convert normalized observations back to units and define a braking speed for altitude.
        dx, y, vx, vy = obs[0] * 100, obs[1] * 500, obs[2] * 20, obs[3] * 50
        v_ref = -(math.sqrt(2 * self.A_REF * max(y, 0.0)) + 0.5)
        vx_ref = max(-10.0, min(10.0, -0.15 * dx))
        return -(abs(dx) / 100 + abs(y) / 500 + abs(vy - v_ref) / 50
                 + abs(vx - vx_ref) / 20 + abs(obs[4]) + 0.5 * abs(obs[5]))
