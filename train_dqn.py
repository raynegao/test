"""Training script for the Catch environment using DQN."""
from __future__ import annotations

import argparse
import statistics
from typing import List, Optional

from catch import CatchEnv, CatchRenderer, DQNAgent, DQNConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument(
        "--target-update",
        type=int,
        default=200,
        help="Number of optimizer steps between target network updates",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10_000, help="Replay buffer capacity"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for optimization"
    )
    parser.add_argument(
        "--episodes-log-window",
        type=int,
        default=20,
        help="Number of recent episodes to average in the log output",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display training curves at the end using matplotlib.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Save training curves figure to this path (requires matplotlib).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show a Tkinter window with the agent playing during training.",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=400,
        help="Window width when --render is enabled.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=400,
        help="Window height when --render is enabled.",
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between rendered frames.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()

    import random

    random.seed(args.seed)

    env = CatchEnv(seed=args.seed)
    initial_state = env.reset()
    state_dim = len(initial_state)
    action_dim = env.action_space

    config = DQNConfig(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update,
        seed=args.seed,
    )
    agent = DQNAgent(state_dim, action_dim, config=config)

    renderer: Optional[CatchRenderer] = None
    render_delay = max(0.0, args.render_delay)
    if args.render:
        try:
            renderer = CatchRenderer(width=args.render_width, height=args.render_height)
        except Exception as exc:  # pragma: no cover - GUI creation is environment-dependent
            print(f"Renderer disabled: {exc}. Continuing without visualization.")
            renderer = None

    def maybe_render() -> None:
        nonlocal renderer
        if not renderer:
            return
        renderer.render_env(env, delay=render_delay)
        if renderer.closed:
            print("Renderer window closed. Continuing without visualization.")
            renderer = None

    episode_rewards: List[float] = []
    # Track per-episode metrics for optional visualization.
    history = {
        "episode": [],
        "reward": [],
        "avg_reward": [],
        "avg_loss": [],
        "epsilon": [],
    }

    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            maybe_render()
            done = False
            total_reward = 0.0
            losses: List[float] = []

            for _ in range(env.max_steps):
                action = agent.select_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                agent.push_transition(state, action, reward, next_state, done)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                maybe_render()
                total_reward += reward
                if done:
                    break

            episode_rewards.append(total_reward)
            avg_reward = statistics.fmean(episode_rewards[-args.episodes_log_window :])
            avg_loss = statistics.fmean(losses) if losses else 0.0
            history["episode"].append(episode)
            history["reward"].append(total_reward)
            history["avg_reward"].append(avg_reward)
            history["avg_loss"].append(avg_loss)
            history["epsilon"].append(agent.epsilon)
            print(
                f"Episode {episode:04d} | reward={total_reward:6.2f} | "
                f"avg_reward={avg_reward:6.2f} | loss={avg_loss:7.4f} | epsilon={agent.epsilon:5.2f}",
                flush=True,
            )
    finally:
        if renderer:
            renderer.close()

    if args.model_path:
        agent.save(args.model_path)
        print(f"Model saved to {args.model_path}")

    should_plot = args.plot or bool(args.plot_path)
    if should_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "matplotlib is required for plotting. Install it with `pip install matplotlib`."
            )
        else:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
            episodes = history["episode"]

            axes[0].plot(episodes, history["reward"], label="Episode reward", color="tab:blue", alpha=0.35)
            axes[0].plot(
                episodes,
                history["avg_reward"],
                label=f"Moving average ({args.episodes_log_window})",
                color="tab:blue",
            )
            axes[0].set_ylabel("Reward")
            axes[0].legend(loc="upper left")
            axes[0].grid(True, alpha=0.2)

            axes[1].plot(episodes, history["avg_loss"], label="Average loss", color="tab:orange")
            axes[1].set_ylabel("Loss")
            axes[1].legend(loc="upper left")
            axes[1].grid(True, alpha=0.2)

            axes[2].plot(episodes, history["epsilon"], label="Epsilon", color="tab:green")
            axes[2].set_xlabel("Episode")
            axes[2].set_ylabel("Epsilon")
            axes[2].legend(loc="upper right")
            axes[2].grid(True, alpha=0.2)

            fig.tight_layout()

            if args.plot_path:
                fig.savefig(args.plot_path)
                print(f"Training curves saved to {args.plot_path}")
            if args.plot:
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
