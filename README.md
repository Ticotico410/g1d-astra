# G1-D

Unitree G1-D 白板任务 monorepo：遥操 harness + Astra agent。

| 目录 | 作用 |
|---|---|
| [`g1d_teleop/`](g1d_teleop/) | 机器人底层：配置、DDS publisher、MCAP 解析、相机客户端、示教数据 |
| [`g1d-astra/`](g1d-astra/) | Agent 层：GPT-6 Astra 分段门控 + 三个白板 Notebook |

本地默认仍可按绝对路径使用：

```text
/home/karthus_chen/ycb_ws/g1d_teleop
/home/karthus_chen/ycb_ws/g1d-astra
```

若直接在本 monorepo 内运行，设置：

```bash
export G1D_TELEOP_ROOT="$PWD/g1d_teleop"
```

然后按 [`g1d-astra/README.md`](g1d-astra/README.md) 与 [`g1d_teleop/README.md`](g1d_teleop/README.md) 安装依赖并启动任务。

**注意：** 不要提交 `.env`；运行录像与状态文件在 `g1d-astra/data/`，默认已忽略。
