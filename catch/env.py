"""Catch environment implementation using the Python standard library."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CatchState:
    """State of the Catch environment."""

    ball_x: float
    ball_y: float
    paddle_x: float
    velocity_y: float

    def as_array(self) -> List[float]:
        return [self.ball_x, self.ball_y, self.paddle_x, self.velocity_y]


class CatchEnv:
    """A simple differentiable catch environment.

    The environment models a small paddle moving on the bottom of the screen
    trying to catch a falling ball. The state is continuous and normalized to
    the range [0, 1] where possible, making it friendly for neural networks.
    """

    ACTIONS = (-1, 0, 1)

    def __init__(
        self,
        paddle_width: float = 0.2,
        paddle_speed: float = 0.05,
        min_fall_speed: float = 0.02,
        max_fall_speed: float = 0.04,
        gravity: float = 0.0,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        if not 0 < paddle_width < 1:
            raise ValueError("paddle_width must be in the interval (0, 1)")
        self.paddle_width = paddle_width
        self.paddle_speed = paddle_speed
        self.min_fall_speed = min_fall_speed
        self.max_fall_speed = max_fall_speed
        self.gravity = gravity
        self.max_steps = max_steps
        self._seed = seed
        self._rng = self._create_rng(seed)

        self.state: Optional[CatchState] = None
        self.steps: int = 0

    @staticmethod
    def _create_rng(seed: Optional[int]):
        import random

        rng = random.Random()
        rng.seed(seed)
        return rng

    @property
    def action_space(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> List[float]:
        ball_x = float(self._rng.uniform(0.05, 0.95))
        ball_y = 0.0
        paddle_x = 0.5
        velocity_y = float(self._rng.uniform(self.min_fall_speed, self.max_fall_speed))
        self.state = CatchState(ball_x, ball_y, paddle_x, velocity_y)
        self.steps = 0
        return self.state.as_array()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, float]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if not 0 <= action < self.action_space:
            raise ValueError(f"Action must be in [0, {self.action_space - 1}]")

        direction = self.ACTIONS[action]
        half_width = self.paddle_width / 2.0
        paddle_x = self.state.paddle_x + direction * self.paddle_speed
        paddle_x = max(half_width, min(1 - half_width, paddle_x))

        velocity_y = self.state.velocity_y + self.gravity
        ball_y = self.state.ball_y + velocity_y
        ball_x = self.state.ball_x
        self.steps += 1

        done = False
        reward = -0.01
        caught_flag = 0.0

        if ball_y >= 1.0 or self.steps >= self.max_steps:
            done = True
            ball_y = 1.0
            caught = abs(ball_x - paddle_x) <= half_width
            reward = 1.0 if caught else -1.0
            caught_flag = 1.0 if caught else 0.0

        self.state = CatchState(ball_x, ball_y, paddle_x, velocity_y)
        return self.state.as_array(), reward, done, {"caught": caught_flag}

    def render(self, width: int = 40, height: int = 20) -> str:
        if self.state is None:
            raise RuntimeError("Environment must be reset before rendering.")

        grid = [[" " for _ in range(width)] for _ in range(height)]
        ball_x = int(self.state.ball_x * (width - 1))
        ball_y = min(height - 1, int(self.state.ball_y * (height - 1)))
        grid[ball_y][ball_x] = "o"

        half = max(1, int(self.paddle_width * width / 2))
        paddle_center = int(self.state.paddle_x * (width - 1))
        left = max(0, paddle_center - half)
        right = min(width, paddle_center + half + 1)
        for idx in range(left, right):
            grid[-1][idx] = "="

        return "\n".join("".join(row) for row in grid)

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        self._rng = self._create_rng(seed)

    def sample_action(self) -> int:
        return self._rng.randrange(self.action_space)
