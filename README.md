# G1-D Astra

Unitree G1-D 白板任务 monorepo：遥操 SDK + Astra agent。

## 概览

| 目录 | 作用 |
|---|---|
| [`teleop/`](teleop/) | G1-D 机器人 SDK / harness：配置与增益、Unitree DDS 发布、MCAP 录制与回放、相机客户端，以及白板示教数据 |
| [`agent/`](agent/) | 任务 agent：用 GPT-6 Astra 看四路实时图像，分段门控执行示教动作；含三个白板 Notebook 与运行记录 |

`teleop/` 负责“怎么连上机器人、怎么发控制、怎么读写示教”；`agent/` 负责“按任务提示做闭环决策并驱动示教段执行”。Agent 通过环境变量 `G1D_TELEOP_ROOT` 定位 teleop；未设置时也可使用本地绝对路径。

本地独立工作目录仍可用：

```text
/home/karthus_chen/ycb_ws/g1d_teleop
/home/karthus_chen/ycb_ws/g1d-astra
```

在本 monorepo 内运行时：

```bash
export G1D_TELEOP_ROOT="$PWD/teleop"
```

然后按 [`agent/README.md`](agent/README.md) 与 [`teleop/README.md`](teleop/README.md) 安装依赖并启动任务。
