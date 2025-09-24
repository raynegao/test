from catch.env import CatchEnv, CatchState


def test_reset_state_shape_and_range():
    env = CatchEnv(seed=123)
    state = env.reset()
    assert len(state) == 4
    assert all(0.0 <= value <= 1.0 for value in state)


def test_step_moves_paddle():
    env = CatchEnv(seed=0, paddle_speed=0.2)
    env.reset()
    original_paddle_x = env.state.paddle_x
    env.step(2)  # move right
    assert env.state.paddle_x > original_paddle_x


def test_successful_catch_reward():
    env = CatchEnv(seed=0, paddle_width=0.6)
    env.reset()
    env.state = CatchState(ball_x=0.5, ball_y=0.95, paddle_x=0.5, velocity_y=0.05)
    _, reward, done, info = env.step(1)
    assert done is True
    assert reward == 1.0
    assert info["caught"] == 1.0


def test_missed_catch_reward():
    env = CatchEnv(seed=0, paddle_width=0.2)
    env.reset()
    env.state = CatchState(ball_x=0.1, ball_y=0.95, paddle_x=0.9, velocity_y=0.05)
    _, reward, done, info = env.step(1)
    assert done is True
    assert reward == -1.0
    assert info["caught"] == 0.0
