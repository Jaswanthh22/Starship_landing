"""2D Super Heavy booster catch environment.

A booster falls toward a tower with two "chopsticks" arms. The agent controls
the engine throttle, the engine gimbal and a small attitude-control torque
(grid fins / cold-gas thrusters). The episode ends when the booster's catch
height reaches the arms (y <= 0): it is a catch only if position, speed and
tilt are all inside the tolerances.

Simplifications (on purpose, to make it learnable): flat 2D world, rigid rod
body, one lumped engine, no engine relight delay, simple drag. The catch pins
are modelled at the centre of mass.

Observation (7 floats, roughly in [-1, 1] range):
    0 dx/100   1 y/500   2 vx/20   3 vy/50   4 theta (rad)   5 omega (rad/s)   6 fuel fraction
    (theta > 0 means the booster tilts toward +x)

Action (3 floats in [-1, 1]):
    0 throttle  (-1 -> 0 %, +1 -> 100 %)
    1 gimbal    (+-MAX_GIMBAL rad)
    2 rcs       (attitude torque, +1 spins theta upward)

`difficulty` in [0, 1] scales both the spread of start states and how tight
the catch tolerances are (0 = wide start spread + generous arms,
1 = real-ish tolerances). The curriculum in train.py raises it automatically.
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---- physical constants ----------------------------------------------------
# Gravity, time step, vehicle mass/fuel, engine limits, and simple drag model.
G = 9.81
G0 = 9.80665
DT = 0.05                  # seconds per step (20 Hz)
M_DRY = 200e3              # kg, booster + landing propellant reserve
FUEL0 = 30e3               # kg of usable fuel at the start of the descent
ISP = 330.0                # s
T_MAX = 6.0e6              # N, all engines at 100 %
LENGTH = 70.0              # m
MAX_GIMBAL = 0.2           # rad (~11 degrees)
ALPHA_RCS = 0.4            # rad/s^2 from attitude control at |rcs| = 1
DRAG = 1e-4                # quadratic drag coefficient (1/m)

# catch tolerances at difficulty 0 (easy) and 1 (hard)
TOL = {
    "x": (12.0, 3.0),        # m
    "vx": (5.0, 1.5),        # m/s
    "vy": (8.0, 2.0),        # m/s
    "theta": (0.20, 0.04),   # rad
    "omega": (0.40, 0.10),   # rad/s
}


class StarshipCatchEnv(gym.Env):
    # Gymnasium metadata describes supported render modes and frame rate.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": int(1 / DT)}

    def __init__(self, render_mode=None, difficulty=1.0, wind=0.0, max_steps=600):
        # Store configuration, declare the agent's input/output ranges, and prepare renderer state.
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = float(difficulty)
        self.wind = float(wind)          # max sideways wind acceleration, m/s^2
        self.max_steps = max_steps

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self._screen = None
        self._clock = None
        self._font = None
        self._last_controls = (0.0, 0.0)
        # Artwork is loaded only when rendering starts, keeping training instances light.
        self._art = None

    # ---- helpers ------------------------------------------------------------
    def set_difficulty(self, d):
        """Used by the curriculum callback (works through VecEnv.env_method)."""
        # Clamp the curriculum setting so it stays between the easy and hard extremes.
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def tolerances(self):
        # Linearly interpolate catch limits for the current difficulty.
        d = self.difficulty
        return {k: lo + (hi - lo) * d for k, (lo, hi) in TOL.items()}

    def _obs(self):
        # Return the seven scaled measurements the policy uses to choose its controls.
        return np.array(
            [self.x / 100.0, self.y / 500.0, self.vx / 20.0, self.vy / 50.0,
             self.theta, self.omega, self.fuel / FUEL0],
            dtype=np.float32,
        )

    # ---- gym API ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # Start a reproducible episode and randomize the booster state within difficulty limits.
        super().reset(seed=seed)
        if options and "difficulty" in options:
            self.set_difficulty(options["difficulty"])
        d, rng = self.difficulty, self.np_random

        self.x = rng.uniform(-1, 1) * (20 + 60 * d)
        self.y = rng.uniform(500, 700)
        self.vx = rng.uniform(-1, 1) * (3 + 5 * d)
        self.vy = rng.uniform(-90, -60)
        self.theta = rng.uniform(-1, 1) * (0.03 + 0.07 * d)
        self.omega = 0.0
        self.fuel = FUEL0
        self.wind_acc = rng.uniform(-1, 1) * self.wind * d
        self.steps = 0

        # Draw the starting state immediately when a visible window is requested.
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), {"caught": False}

    def step(self, action):
        # Convert three normalized policy outputs into throttle, gimbal angle, and RCS torque.
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        throttle = (a[0] + 1.0) / 2.0
        gimbal = a[1] * MAX_GIMBAL
        rcs = a[2]
        self._last_controls = (throttle, gimbal)

        # Calculate fuel use and thrust, limiting the final burn to fuel still available.
        mass = M_DRY + self.fuel
        thrust = throttle * T_MAX
        burned = thrust / (ISP * G0) * DT
        if burned > self.fuel:                    # not enough fuel for the full step
            thrust *= self.fuel / burned
            burned = self.fuel
        self.fuel -= burned

        # Resolve engine thrust into horizontal and vertical forces, including wind and drag.
        ang = self.theta + gimbal
        speed = math.hypot(self.vx, self.vy)
        ax = thrust * math.sin(ang) / mass + self.wind_acc - DRAG * speed * self.vx
        ay = thrust * math.cos(ang) / mass - G - DRAG * speed * self.vy
        # Estimate rotational acceleration from gimbal force and attitude-control torque.
        inertia = mass * LENGTH ** 2 / 12.0
        alpha = -(LENGTH / 2.0) * thrust * math.sin(gimbal) / inertia + rcs * ALPHA_RCS

        # Advance the simulated state by one small time step (Euler integration).
        self.vx += ax * DT
        self.vy += ay * DT
        self.omega += alpha * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.theta += self.omega * DT
        self.steps += 1

        # Initialize Gymnasium episode flags, reward, and diagnostic information.
        terminated = truncated = False
        reward = 0.0
        info = {"caught": False, "fuel_left": self.fuel / FUEL0}

        # At arm height, check horizontal position, both speeds, tilt, and rotation rate.
        if self.y <= 0.0:                         # reached the arms
            terminated = True
            tol = self.tolerances()
            ratios = np.array([
                abs(self.x) / tol["x"], abs(self.vx) / tol["vx"], abs(self.vy) / tol["vy"],
                abs(self.theta) / tol["theta"], abs(self.omega) / tol["omega"],
            ])
            # Reward a catch; otherwise penalize the miss according to its normalized size.
            if np.all(ratios <= 1.0):
                info["caught"] = True
                reward = 100.0
            else:
                # graded miss: a near miss hurts less than a crater
                err = float(np.sqrt(np.mean(ratios ** 2)))
                reward = -(10.0 + 40.0 * min(1.0, err / 10.0))
            info["miss_ratios"] = ratios
        # Stop episodes for escape/tip-over, or truncate them at the step limit.
        elif abs(self.x) > 400 or self.y > 1000 or abs(self.theta) > 1.2:
            terminated = True                     # flew away / tipped over
            reward = -50.0
        elif self.steps >= self.max_steps:
            truncated = True

        # Refresh an interactive display if enabled, then return observation and outcome data.
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), reward, terminated, truncated, info

    # ---- rendering ----------------------------------------------------------
    def render(self):
        # Gymnasium's public rendering hook returns pixels for video mode.
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Load pygame lazily so ordinary training does not need a graphical display.
        import pygame
        from pathlib import Path

        # Create a visible window or an off-screen canvas on the first frame.
        W, H = 600, 800
        if self._screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Starship catch (2D)")
            else:
                self._screen = pygame.Surface((W, H))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 22)
            self._small_font = pygame.font.Font(None, 18)

        # Load the supplied image assets once per environment instance.
        if self._art is None:
            asset_dir = Path(__file__).with_name("assets")
            self._art = {
                name: pygame.image.load(str(asset_dir / filename))
                for name, filename in {
                    "booster": "booster.png",
                    "flame": "engine_flame.png",
                    "tower_left": "tower_left.png",
                    "tower_right": "tower_right.png",
                    "ground": "ground.png",
                }.items()
            }
            self._art["background"] = pygame.image.load(str(asset_dir / "launch_background.jpg"))

        # Convert world coordinates in meters to screen pixels.
        scale = (H - 120) / 760.0

        def S(px, py):
            return (W / 2 + px * scale, H - 80 - py * scale)

        def rect(x0, y0, x1, y1, color):
            (sx0, sy0), (sx1, sy1) = S(x0, y1), S(x1, y0)
            pygame.draw.rect(self._screen, color, (sx0, sy0, sx1 - sx0, sy1 - sy0))

        # Paint a dusky sky and fill the scene with the supplied launch-site photograph.
        surf = self._screen
        for y in range(H):
            blend = y / H
            sky = (int(5 + 8 * blend), int(9 + 10 * blend), int(20 + 13 * blend))
            pygame.draw.line(surf, sky, (0, y), (W, y))

        # Keep the photo's aspect ratio while cropping its sides to cover the portrait view.
        background = self._art["background"]
        bg_w = round(H * background.get_width() / background.get_height())
        bg = pygame.transform.smoothscale(background, (bg_w, H))
        surf.blit(bg, ((W - bg_w) // 2, 0))

        # Draw the metal deck, then the catch towers behind the booster.
        deck_h = 96
        deck = pygame.transform.smoothscale(self._art["ground"], (W, deck_h))
        surf.blit(deck, (0, H - deck_h))
        tower_h, tower_w = 225, 78
        left_tower = pygame.transform.smoothscale(self._art["tower_left"], (tower_w, tower_h))
        right_tower = pygame.transform.smoothscale(self._art["tower_right"], (tower_w, tower_h))
        tower_bottom = round(S(0, -40)[1])
        surf.blit(left_tower, (230, tower_bottom - tower_h))
        surf.blit(right_tower, (292, tower_bottom - tower_h))

        # Mark the catch height and highlight the acceptable horizontal landing zone.
        tol_x = self.tolerances()["x"]
        catch_y = round(S(0, 0)[1])
        zone_left, zone_right = round(S(-tol_x, 0)[0]), round(S(tol_x, 0)[0])
        pygame.draw.rect(surf, (50, 190, 150), (zone_left, catch_y - 2, zone_right - zone_left, 4))
        pygame.draw.line(surf, (225, 190, 105), (0, catch_y + 3), (W, catch_y + 3), 1)

        # Add a throttle-sized engine plume aligned with the direction of thrust.
        throttle, gimbal = self._last_controls
        if throttle > 0.02 and self.fuel > 0:
            angle = self.theta + gimbal
            base_x = self.x - LENGTH / 2 * math.sin(self.theta)
            base_y = self.y - LENGTH / 2 * math.cos(self.theta)
            flame_h = max(12, round(86 * throttle))
            flame_w = max(8, round(30 * throttle))
            flame = pygame.transform.smoothscale(self._art["flame"], (flame_w, flame_h))
            flame = pygame.transform.rotate(flame, -math.degrees(angle))
            flame_center = S(base_x - math.sin(angle) * flame_h / (2 * scale),
                             base_y - math.cos(angle) * flame_h / (2 * scale))
            surf.blit(flame, flame.get_rect(center=flame_center))

        # Rotate the detailed booster around its center and align its nozzle with physics.
        sprite_h = 92
        sprite_w = round(sprite_h * self._art["booster"].get_width() / self._art["booster"].get_height())
        booster = pygame.transform.smoothscale(self._art["booster"], (sprite_w, sprite_h))
        booster = pygame.transform.rotate(booster, -math.degrees(self.theta))
        center_x, center_y = S(self.x, self.y)
        center_y -= (sprite_h - LENGTH * scale) / 2
        surf.blit(booster, booster.get_rect(center=(center_x, center_y)))

        # Draw a compact, translucent instrument bar with the important flight values.
        hud_panel = pygame.Surface((W, 54), pygame.SRCALPHA)
        hud_panel.fill((3, 8, 18, 205))
        surf.blit(hud_panel, (0, 0))
        pygame.draw.line(surf, (75, 115, 145), (0, 53), (W, 53), 1)
        values = [
            ("ALT", f"{self.y:,.0f} m"),
            ("DX", f"{self.x:+.1f} m"),
            ("VX", f"{self.vx:+.1f} m/s"),
            ("VY", f"{self.vy:+.1f} m/s"),
            ("TILT", f"{math.degrees(self.theta):+.1f}°"),
            ("FUEL", f"{100 * self.fuel / FUEL0:.0f}%"),
        ]
        col_w = W // len(values)
        for i, (label, value) in enumerate(values):
            x = i * col_w + 6
            surf.blit(self._small_font.render(label, True, (125, 170, 200)), (x, 5))
            surf.blit(self._small_font.render(value, True, (238, 242, 248)), (x, 27))
        difficulty_text = self._small_font.render(
            f"CATCH DIFFICULTY {self.difficulty:.1f}", True, (230, 205, 150)
        )
        surf.blit(difficulty_text, difficulty_text.get_rect(topright=(W - 12, 58)))

        # Flip and rate-limit the visible window, or return a NumPy RGB image for recording.
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    def close(self):
        # Shut down pygame and clear its screen handle when the environment closes.
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


# Register the environment name used by gym.make and the training scripts.
gym.register(id="StarshipCatch-v0", entry_point=StarshipCatchEnv)
