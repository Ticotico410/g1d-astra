# G1-D 摇操使用向导

本项目提供 G1-D 的 MuJoCo 仿真、DirectVR 摇操、Unitree DDS 实机控制、MCAP 录制与回放，以及相机图像传输接口。

## 安装环境

```bash
conda create -y -n robot_teleop python=3.10 pip
conda activate robot_teleop
python -m pip install -e .
python -m pip install -e '.[unitree,record]'
```

## 仿真摇操

```bash
python robot_teleop/teleop_robot.py --robot g1_d --mode sim --teleop-control-mode wbc
python robot_teleop/teleop_robot.py --robot g1_d --mode sim --teleop-control-mode manual
```

## 实机摇操

启动前确认机器人和 DDS 网络配置正确，清空机器人活动区域，并将实体急停放在可触达位置。

```bash
python robot_teleop/teleop_robot.py --robot g1_d --mode robot --net enp4s0 --teleop-control-mode manual
```

G1-D publisher 向配置的 Unitree DDS 话题发送双臂和腰部指令，向 `rt/cmd_hispeed` 发送升降指令，向 `rt/cmd_vel_no_limit` 发送底盘速度。硬件接口、话题和控制增益位于 `robot_teleop/config/g1_d.yaml`。

## 录制与回放

如需录制图像，先启动机器人图像服务。摇操数据采用 `teleop_frame` MCAP 格式，保存 action、state 以及可用的图像帧。

```bash
python robot_teleop/teleop_robot.py --robot g1_d --mode robot --net enp4s0 \
  --record --record-task-name draw_circle
```

在 MuJoCo 中检查 MCAP，不向实机发送指令：

```bash
python robot_teleop/replay_robot.py --robot g1_d \
  --input data/demo/draw_picture/episode_0001.mcap --no-publish
```

确认轨迹正确并清空机器人工作区后，才可回放到实机：

```bash
python robot_teleop/replay_robot.py --robot g1_d \
  --input data/demo/draw_picture/episode_0001.mcap --net enp4s0 --source action
```

## 相机

头部和腕部图像接口位于 `image/robot_driver/image_server`。本机第三视角 USB 相机服务可通过以下命令启动：

```bash
python image/robot_driver/image_server/hikvision_image_server.py \
  --device /dev/video1 --bind 0.0.0.0 --port 5560 --fps 30
```
