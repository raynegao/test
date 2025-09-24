"""Deep Q-Network agent implemented with the Python standard library."""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass
class DQNConfig:
    """Configuration for the DQN agent."""

    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 10_000
    learning_rate: float = 1e-3
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    learning_starts: int = 1_000
    train_freq: int = 1
    hidden_size: int = 128
    seed: Optional[int] = None


class ReplayBuffer:
    """Simple FIFO experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[List[float], int, float, List[float], bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        return len(self.buffer)

    def push(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple[List[float], int, float, List[float], bool]]:
        return random.sample(self.buffer, batch_size)


class NeuralNetwork:
    """Single hidden-layer neural network using tanh activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: random.Random) -> None:
        limit1 = 1.0 / math.sqrt(input_dim)
        self.W1 = [
            [rng.uniform(-limit1, limit1) for _ in range(input_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0 for _ in range(hidden_dim)]

        limit2 = 1.0 / math.sqrt(hidden_dim)
        self.W2 = [
            [rng.uniform(-limit2, limit2) for _ in range(hidden_dim)]
            for _ in range(output_dim)
        ]
        self.b2 = [0.0 for _ in range(output_dim)]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, inputs: Sequence[float]) -> List[float]:
        hidden = [math.tanh(sum(w * x for w, x in zip(weights, inputs)) + bias) for weights, bias in zip(self.W1, self.b1)]
        return [sum(w * h for w, h in zip(weights, hidden)) + bias for weights, bias in zip(self.W2, self.b2)]

    def forward_with_hidden(self, inputs: Sequence[float]) -> Tuple[List[float], List[float]]:
        hidden = [
            math.tanh(sum(w * x for w, x in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.W1, self.b1)
        ]
        outputs = [
            sum(w * h for w, h in zip(weights, hidden)) + bias
            for weights, bias in zip(self.W2, self.b2)
        ]
        return outputs, hidden

    def copy_from(self, other: "NeuralNetwork") -> None:
        self.W1 = [row[:] for row in other.W1]
        self.b1 = other.b1[:]
        self.W2 = [row[:] for row in other.W2]
        self.b2 = other.b2[:]

    def to_dict(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def load_dict(self, data: dict) -> None:
        self.W1 = [row[:] for row in data["W1"]]
        self.b1 = list(data["b1"])
        self.W2 = [row[:] for row in data["W2"]]
        self.b2 = list(data["b2"])


class DQNAgent:
    """Basic DQN agent with manual gradient computation."""

    def __init__(self, state_dim: int, action_dim: int, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.rng = random.Random(self.config.seed)

        hidden = self.config.hidden_size
        self.policy_net = NeuralNetwork(state_dim, hidden, action_dim, self.rng)
        self.target_net = NeuralNetwork(state_dim, hidden, action_dim, self.rng)
        self.target_net.copy_from(self.policy_net)

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.epsilon = self.config.epsilon_start
        self.steps = 0
        self.updates = 0

    def select_action(self, state: List[float], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)
        q_values = self.policy_net.forward(state)
        max_value = max(q_values)
        for idx, value in enumerate(q_values):
            if value == max_value:
                return idx
        return 0

    def push_transition(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps += 1

    def _optimize(self, batch: List[Tuple[List[float], int, float, List[float], bool]]) -> float:
        hidden_dim = self.policy_net.hidden_dim
        action_dim = self.action_dim
        input_dim = self.state_dim

        grad_W1 = [[0.0 for _ in range(input_dim)] for _ in range(hidden_dim)]
        grad_b1 = [0.0 for _ in range(hidden_dim)]
        grad_W2 = [[0.0 for _ in range(hidden_dim)] for _ in range(action_dim)]
        grad_b2 = [0.0 for _ in range(action_dim)]

        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            q_values, hidden = self.policy_net.forward_with_hidden(state)
            selected_q = q_values[action]

            next_q_values = self.target_net.forward(next_state)
            target = reward
            if not done:
                target += self.config.gamma * max(next_q_values)

            error = selected_q - target
            total_loss += 0.5 * error * error

            delta_output = [0.0 for _ in range(action_dim)]
            delta_output[action] = error

            # Accumulate gradients for output layer
            for j in range(action_dim):
                if delta_output[j] == 0.0:
                    continue
                for i in range(hidden_dim):
                    grad_W2[j][i] += delta_output[j] * hidden[i]
                grad_b2[j] += delta_output[j]

            # Propagate to hidden layer
            delta_hidden = [0.0 for _ in range(hidden_dim)]
            for i in range(hidden_dim):
                back_signal = 0.0
                for j in range(action_dim):
                    if delta_output[j] == 0.0:
                        continue
                    back_signal += delta_output[j] * self.policy_net.W2[j][i]
                delta_hidden[i] = (1.0 - hidden[i] * hidden[i]) * back_signal

            for i in range(hidden_dim):
                for k in range(input_dim):
                    grad_W1[i][k] += delta_hidden[i] * state[k]
                grad_b1[i] += delta_hidden[i]

        batch_size = len(batch)
        scale = self.config.learning_rate / batch_size

        # Gradient step with simple clipping to stabilise training
        def clip(value: float, limit: float = 5.0) -> float:
            if value > limit:
                return limit
            if value < -limit:
                return -limit
            return value

        for i in range(hidden_dim):
            for k in range(input_dim):
                update = clip(grad_W1[i][k]) * scale
                self.policy_net.W1[i][k] -= update
            self.policy_net.b1[i] -= clip(grad_b1[i]) * scale

        for j in range(action_dim):
            for i in range(hidden_dim):
                update = clip(grad_W2[j][i]) * scale
                self.policy_net.W2[j][i] -= update
            self.policy_net.b2[j] -= clip(grad_b2[j]) * scale

        return total_loss / batch_size

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.learning_starts:
            return None
        if self.steps % self.config.train_freq != 0:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        loss = self._optimize(batch)

        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target_net.copy_from(self.policy_net)

        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return loss

    def save(self, path: str) -> None:
        data = {
            "config": self.config.__dict__,
            "network": self.policy_net.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        network_state = data.get("network", {})
        self.policy_net.load_dict(network_state)
        self.target_net.copy_from(self.policy_net)
