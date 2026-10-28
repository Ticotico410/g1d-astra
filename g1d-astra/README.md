# G1-D Agent 白板操作

本仓库是 G1-D 的 agent 层，提供三个可直接运行的白板任务：

| 任务 | Notebook | 示教数据 |
|---|---|---|
| 画圆 | [`notebooks/01_g1d_draw_circle.ipynb`](notebooks/01_g1d_draw_circle.ipynb) | `g1d_teleop/data/demo/draw_circle/episode_0000.mcap` |
| 写名字 | [`notebooks/02_g1d_write_name.ipynb`](notebooks/02_g1d_write_name.ipynb) | `g1d_teleop/data/demo/write_g1d/episode_0001.mcap` |
| 双臂山水画 | [`notebooks/03_g1d_draw_landscape.ipynb`](notebooks/03_g1d_draw_landscape.ipynb) | `g1d_teleop/data/demo/draw_picture/episode_0001.mcap` |

三个任务共用 [`scripts/g1d_reference_agent.py`](scripts/g1d_reference_agent.py) 和 [`scripts/ui.py`](scripts/ui.py)。Notebook 负责选择任务示教、任务提示和运行目录，不包含另一套机器人控制实现。

## 与摇操 Harness 的关系

G1-D 的机器人底层能力位于独立仓库：

```text
/home/karthus_chen/ycb_ws/g1d_teleop
```

它是本仓库使用的 harness，负责：

- `robot_teleop.config`：合并 `common.yaml` 与 `g1_d.yaml`，提供关节、DDS 话题和控制增益配置。
- `robot_teleop.replay`：解析 `teleop_frame` MCAP、计算实际采样率、把示教 action 转为 publisher 参数，并在启动时从实测姿态平滑进入任务起始姿态。
- `robot_teleop.publisher.G1DDSPublisher`：通过 Unitree DDS 下发双臂、腰部、升降、底盘和 Dex1 指令。
- `image_client.ImageClient`：订阅头部双目及左右腕相机。本流程使用头部右目、左腕和右腕。
- `hikvision_image_client.HikvisionImageClient`：订阅本机第三视角 USB 相机服务。

Agent 通过 `G1D_TELEOP_ROOT` 定位 harness；未设置时默认使用 `/home/karthus_chen/ycb_ws/g1d_teleop`。启动时把以下路径加入 Python import path：

```text
$G1D_TELEOP_ROOT/robot_teleop/src
$G1D_TELEOP_ROOT/image/robot_driver/image_server
```

核心调用链为：

```text
Notebook
  -> scripts/ui.py                         # 四路画面、中文状态、录像
  -> scripts/g1d_reference_agent.py
       -> ImageClient + third camera       # 实时视觉观测
       -> core.decider.Astra.choose()      # 选择继续、等待或完成
       -> robot_teleop.replay._load_frames # 加载任务示教 MCAP
       -> replay._publish_args             # action 转控制参数
       -> replay._ramp_in                  # 平滑进入任务起始姿态
       -> publisher.make_publisher         # 创建 G1DDSPublisher
       -> publisher.publish                # 下发 G1-D 动作指令
```

`g1d_reference_agent.py` 直接创建 DDS publisher，因此运行 agent 时不要同时启动：

```bash
python robot_teleop/teleop_robot.py --robot g1_d --mode robot ...
```

否则两个控制进程会争用机器人。

## 闭环执行机制

MCAP 提供人工验证的动作示教，不要求包含图像。实时相机是 agent 的视觉观测来源。

每轮执行顺序如下：

1. 等待头部右目、左右腕和第三视角图像全部到达，且帧龄小于 2 秒。
2. 把四张 RGB 图像、当前任务进度和任务文字交给 `GPT-6 Astra`。
3. 模型只能选择以下技能之一：
   - `advance_reference_step`：继续执行任务动作。
   - `hold_and_look`：保持当前姿态并重新观察。
   - `done`：参考动作已全部执行，且最终结果在画面中可见。
4. 首次执行前，harness 从机器人实测双臂姿态平滑过渡到任务起始姿态。
5. 动作执行频率根据示教实测 FPS 确定，默认以较慢速度运行；底盘前三维速度始终强制为零。
6. Agent 持续获取新图像，根据实时视觉和任务进度决定继续、观察或结束。

模型不直接生成关节角。它负责基于实时视觉和任务状态驱动动作执行，并判断最终任务是否完成。

## 环境安装

先安装 G1-D harness：

```bash
cd /home/karthus_chen/ycb_ws/g1d_teleop
conda create -y -n robot_teleop python=3.10 pip
conda activate robot_teleop
python -m pip install -e '.[unitree,record]'
```

再安装本仓库：

```bash
cd /home/karthus_chen/ycb_ws/piper-astra-jev
conda activate robot_teleop
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

在 `.env` 中填写：

```dotenv
OPENAI_API_KEY=
# 使用兼容 Responses API 的 relay 时填写；官方 API 可留空。
OPENAI_BASE_URL=
```

默认设备参数：

| 项目 | 默认值 |
|---|---|
| DDS 网卡 | `enp4s0` |
| 机器人图像主机 | `192.168.123.164` |
| 头部图像端口 | `55555` |
| 左腕图像端口 | `55556` |
| 右腕图像端口 | `55557` |
| 第三视角服务 | `127.0.0.1:5560` |

如 harness 不在默认位置：

```bash
export G1D_TELEOP_ROOT=/path/to/g1d_teleop
```

## 启动相机

机器人端需发布头部与双腕相机流。本机第三视角服务可通过 harness 启动：

```bash
conda activate robot_teleop
cd /home/karthus_chen/ycb_ws/g1d_teleop
python image/robot_driver/image_server/hikvision_image_server.py \
  --device /dev/video1 --bind 0.0.0.0 --port 5560 --fps 30
```

## 使用 Notebook

启动 Jupyter：

```bash
conda activate robot_teleop
cd /home/karthus_chen/ycb_ws/piper-astra-jev
jupyter lab
```

打开三个 Notebook 中的一个并按顺序运行：

1. 初始化单元检查 harness、G1-D 配置、参考 MCAP 和 API 配置，并创建新的运行目录。
2. 确认机器人已允许自动运动、笔已固定、工作区已清空、急停可触达。
3. 运行闭环单元。Notebook 先启动 UI，再启动 `g1d_reference_agent.py`，等待 agent 结束后关闭子进程并打印最终状态和录像路径。

三个 Notebook 传给同一个控制脚本的差异只有：

- `--demo`：对应任务的参考 MCAP。
- `--task-text`：UI 和模型看到的任务目标。
- `--agent-instructions`：任务特定的完成判据；画圆使用通用指令。
- `--phase-label`：UI 进度名称。

## 手动使用控制脚本

不使用 Notebook 时，先创建运行目录并启动 UI：

```bash
conda activate robot_teleop
cd /home/karthus_chen/ycb_ws/piper-astra-jev
RUN_DIR="$PWD/data/g1d_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
python scripts/ui.py "$RUN_DIR"
```

在另一个终端运行 agent。下面以双臂山水画为例：

```bash
conda activate robot_teleop
cd /home/karthus_chen/ycb_ws/piper-astra-jev
python scripts/g1d_reference_agent.py \
  --run-dir "$RUN_DIR" \
  --demo /home/karthus_chen/ycb_ws/g1d_teleop/data/demo/draw_picture/episode_0001.mcap \
  --task-text '左臂画太阳，右臂画简化山体与飞鸟' \
  --agent-instructions '左臂画太阳，右臂画简化山体轮廓和线条飞鸟；根据四路现场画面检查安全和最终画面。' \
  --phase-label '双臂山水画' \
  --net enp4s0 \
  --segment-frames 25 \
  --slowdown 2.0
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--run-dir` | UI、状态、决策日志和录像的共享目录，必填。 |
| `--demo` | `teleop_frame` MCAP 任务示教。 |
| `--task-text` | 任务目标，同时显示在 UI。 |
| `--agent-instructions` | 可选的任务特定模型指令。 |
| `--phase-label` | UI 进度阶段名称。 |
| `--net` | Unitree DDS 网卡。 |
| `--segment-frames` | 每轮动作执行的内部控制窗口，默认 25。 |
| `--slowdown` | 相对示教速度的减速倍率，默认 2.0，即半速。 |
| `--max-decisions` | 最大决策轮数，默认 80。 |
| `--robot-image-host` | 机器人图像服务地址。 |
| `--third-camera-host/port` | 第三视角服务地址和端口。 |

## UI 与输出

UI 左侧显示第三视角，右侧显示头部右目、左腕、右腕和中文任务信息。UI 自动将完整界面录制到运行目录。

每次运行生成：

```text
RUN_DIR/
├── operator_view.mp4  # 四路画面和状态 UI 录像
├── decisions.jsonl    # 每轮模型决策、说明和任务进度
├── status.json        # UI 使用的当前状态与最终状态
└── stop_requested     # UI 请求停止时创建
```

UI 的 Q/Esc 会创建 `stop_requested`，agent 检测到后停止发送新动作并关闭 publisher。紧急情况始终优先使用实体急停。

## 其他脚本

- [`scripts/g1d_sim_smoke.py`](scripts/g1d_sim_smoke.py)：离线检查 DirectVR、IK 和 MuJoCo 链路。

## 安全要求

- 清空双臂、升降和底盘活动范围，固定笔和相机线缆。
- 确认实体急停可触达，不得无人值守运行。
- 开始前核对 `--demo` 与目标任务一致。
- 只允许一个 G1-D DDS 控制进程运行。
- 视觉检查不能替代力传感器；本流程没有笔尖接触力控制。
- UI 停止属于软件停止，发生碰撞风险时直接使用实体急停。
