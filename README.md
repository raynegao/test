# Catch 强化学习环境

Catch 是一个极简的二维接球小游戏。本仓库提供：

- `catch/env.py` 中定义的可微调 Catch 环境；
- `catch/dqn.py` 中仅依赖 Python 标准库实现的 DQN 智能体；
- `train_dqn.py` 训练脚本，可用于复现基线、导出模型并可视化训练过程；
- `tests/` 目录下用于校验环境动力学的单元测试。

## 环境需求

- Python 3.10 及以上版本（项目使用类型注解与标准库特性）。
- 额外可选依赖：
  - `matplotlib`（启用训练曲线绘制所需）。
  - `pytest`（运行单元测试所需）。

安装可选依赖示例：

```bash
pip install matplotlib pytest
```

## 快速开始

1. 训练 500 个回合并输出日志：
   ```bash
   python train_dqn.py --episodes 500
   ```
2. 调整关键超参数并保存模型：
   ```bash
   python train_dqn.py \
       --episodes 800 \
       --learning-rate 5e-4 \
       --batch-size 128 \
       --model-path catch_agent.json
   ```

训练过程中脚本会输出每个回合的即时奖励、滑动平均奖励、最近一次更新的平均损失以及当前探索率（epsilon）。

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--episodes` | 训练回合数 | `500` |
| `--seed` | 随机种子 | `42` |
| `--model-path` | 训练结束后保存模型的路径（JSON） | `None` |
| `--target-update` | 目标网络更新间隔（优化步数） | `200` |
| `--learning-rate` | 优化器学习率 | `1e-3` |
| `--buffer-size` | 经验回放容量 | `10000` |
| `--batch-size` | 每次优化的样本数 | `64` |
| `--episodes-log-window` | 日志中奖励滑动平均窗口 | `20` |
| `--plot` | 训练完成后弹出可视化窗口（需 `matplotlib`） | `False` |
| `--plot-path` | 将训练曲线保存到指定路径 | `None` |
| `--render` | 在训练时实时展示 Tkinter 游戏界面 | `False` |
| `--render-width` | 渲染窗口宽度，仅在 --render 时生效 | `400` |
| `--render-height` | 渲染窗口高度，仅在 --render 时生效 | `400` |
| `--render-delay` | 每帧额外暂停时间(秒)，最小值 0.0 | `0.0` |

## 训练过程可视化

执行以下命令可在训练结束后展示或保存曲线图：

```bash
python train_dqn.py --episodes 600 --plot --plot-path curves.png
```

生成的图像包含三条子图：

1. 每回合奖励与滑动平均奖励；
2. 训练期间的平均损失；
3. ε-贪心策略中的探索率变化。

要实时观察训练过程，可以使用 `--render` 打开 Tkinter 图形窗口，实时查看每个回合的游戏状态。
可结合 `--render-delay` 设置每帧停顿时长，值为 0.0 时表示不额外延迟；也可以用 `--render-width` 和 `--render-height` 调整窗口尺寸。

```bash
python train_dqn.py --episodes 120 --render --render-delay 0.05
```

若未安装 `matplotlib`，脚本会给出提示而不会终止训练。

## 模型保存与再训练

- 使用 `--model-path` 可以在训练结束后将策略网络参数导出为 JSON。
- `catch.dqn.DQNAgent` 支持 `load(path)` 方法，可用于在自定义脚本中加载模型继续训练或评估。

## 运行测试

项目附带基础单元测试，用于保障环境动力学不被破坏：

```bash
pytest
```

若尚未安装 `pytest`，可参考前文命令安装。

## 项目结构

```text
├── catch
│   ├── __init__.py         # 暴露 CatchEnv、DQNAgent 等核心类
│   ├── dqn.py              # DQN 智能体与经验回放实现
│   └── env.py              # Catch 环境定义与交互接口
├── tests
│   ├── conftest.py         # 测试夹具
│   └── test_env.py         # 环境行为相关测试
├── train_dqn.py            # 训练入口脚本
├── README.md               # 使用说明（当前文件）
└── .gitignore
```

## 常见问题

- **训练效果波动较大？** 可适当增大回合数、提高缓冲区容量或调整学习率。
- **绘图窗口无法显示？** 请确认已安装 `matplotlib`，并在桌面环境中运行；若在远程服务器上可改用 `--plot-path` 输出图片。
- **如何复现实验？** 固定 `--seed` 参数即可确保环境与智能体初始化一致。

欢迎在体验过程中提 Issue 或提交改进！
