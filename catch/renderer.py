"""Tkinter-based renderer for the Catch environment."""
from __future__ import annotations

import time
import tkinter as tk
from .env import CatchEnv, CatchState


class CatchRenderer:
    """Render the Catch environment using Tkinter."""

    def __init__(self, width: int = 400, height: int = 400, title: str = "Catch Training") -> None:
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.pack()

        self.ball_radius = max(4, int(min(width, height) * 0.04))
        self.paddle_height = max(6, int(height * 0.05))

        self._ball_item = self.canvas.create_oval(0, 0, 0, 0, fill="#ff5c5c", outline="")
        self._paddle_item = self.canvas.create_rectangle(0, 0, 0, 0, fill="#4c6ef5", outline="")

        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _handle_close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _draw(self, state: CatchState, paddle_width: float) -> None:
        ball_x = state.ball_x * self.width
        ball_y = state.ball_y * self.height
        r = self.ball_radius
        self.canvas.coords(self._ball_item, ball_x - r, ball_y - r, ball_x + r, ball_y + r)

        paddle_center = state.paddle_x * self.width
        half_width = (paddle_width * self.width) / 2.0
        paddle_top = self.height - self.paddle_height - 4
        paddle_bottom = self.height - 4
        self.canvas.coords(
            self._paddle_item,
            paddle_center - half_width,
            paddle_top,
            paddle_center + half_width,
            paddle_bottom,
        )

    def render_state(self, state: CatchState, paddle_width: float, delay: float = 0.0) -> None:
        if self.closed:
            return

        self._draw(state, paddle_width)
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            return

        if delay > 0.0:
            time.sleep(delay)

    def render_env(self, env: CatchEnv, delay: float = 0.0) -> None:
        if env.state is None:
            return
        self.render_state(env.state, env.paddle_width, delay=delay)

    def close(self) -> None:
        if self.closed:
            return
        self._handle_close()
