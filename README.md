# Z-Bot 外肢体抓取

[在线演示（部署后可直接打开运行）](https://zhuyanhe1975-hit.github.io/zbot_mjlab_exolimb_grasp/)

上面的链接是浏览器里的运动学演示地址。Pages 还没有发布成功时，打开它会是 404；仓库里的代码已经在 `main` 上。请在 Settings → Pages 把 Source 选成 GitHub Actions 并保存一次，之后推送到 `main` 会把页面部署到这个地址。

网页演示可以切换六模块构型。基准构型的五个连接朝向都是 0，关节轴仍是 `0 0 1`，与 `assets/zbot_6/ZBOT6.xml` 一致。另一种把中间模块（从肩部数起的第 4 个，XML 里的 `base`）相对上一模块绕连接轴转动 90°。在仓库根目录运行 `python scripts/evaluate_config.py`（需要 NumPy）会用同一批 20000 组关节角比较这两条的工作空间体素占用，以及平移 Yoshikawa 可操作度的中位数，并把结果写入 `web/public/config_metrics.json`。

五个连接各自只能取 0 或 π/2，合法序列一共 32 条。`python scripts/evaluate_batch.py` 会把这 32 条全部评完，写出 `web/public/config_batch.csv` 和帕累托摘要 `web/public/config_batch_summary.json`。页面下拉框保留基准构型和「中间模块旋转 90°」，并加上摘要里比基准更好的一到两个代表构型。

本仓库包含两部分：

1. **本地 MJLab 训练**。肩部固定的六模块 Z-Bot 手臂学习接触托盘上的橙色工具。训练在本机运行，不在浏览器里进行。
2. **网页运动学演示**。用 Three.js 显示同一条机械臂、托盘和工具，可拖动关节、播放从伸直到名义姿态的接近过程。页面不计算接触力，也不运行仿真器。

## 模型

权威模型来自 [zbot_rl_mjlab_R3](https://github.com/zhuyanhe1975-hit/zbot_rl_mjlab_R3) 提交 `0c0b686c69f8f78e79bf9f875dead868f25e1cdc` 的 `assets/zbot_6/ZBOT6.xml` 与 `meshes/a.STL`、`meshes/b.STL`。执行器采用该提交 `src/zbot_rl_mjlab/robot.py` 里实际用于仿真的位置伺服。

相对原模型只做了两处几何改动：去掉根部自由关节，把肩部 `foot_0` 固定在 `(0, 0, 0.40)` m。质量、惯量、关节行程、关节阻尼和网格没有改成 USD 转换值。

| 项目 | 取值 |
| --- | --- |
| a 连杆质量 | 0.54171 kg |
| b 连杆质量 | 0.17138 kg |
| 整臂质量 | 4.27854 kg |
| 关节 | joint1–joint6，轴 `0 0 1`，行程 ±6.28 rad |
| 关节阻尼 | 0.05 |
| armature | 0.001 |
| 位置伺服 | kp 15，kd 1.5，力矩上限 20 N·m |
| 名义姿态 | `(0.312, 0.837, -2.02, 2.02, -0.837, -0.312)` rad |
| 伸直 | 六个关节均为 0 |

抓取成功要同时满足：末端模块 `foot_1` 与工具接触，并且末端原点到工具原点的距离小于 **0.08 m**。奖励依次鼓励接近、接触、成功；成功之后再鼓励工具线速度变小。网页演示只判断这 0.08 m 的距离。

## 本地训练

需要 Python 3.10 及以上。CPU 安装：

```bash
bash scripts/install_cpu.sh
python scripts/smoke_test.py
```

冒烟检查确认质量、肩部高度、关节参数，以及名义姿态下的距离和接触。已安装 MJLab 时，它还会创建 1 个环境并走一步。

开始训练（没有 GPU 时自动使用 CPU）：

```bash
python scripts/train.py --num-envs 8 --max-iterations 20
```

日志默认写到本地 TensorBoard 目录 `logs/rsl_rl/zbot_exolimb_grasp/`，不需要 Weights & Biases 账号。正式训练可以加大 `--num-envs` 和 `--max-iterations`。有 NVIDIA GPU 时脚本会使用 GPU；加上 `--cpu` 可强制使用 CPU。脚本在导入 MJLab 之前把 `MUJOCO_GL` 默认设为 `glfw`，这样没有 EGL 的机器也能加载 MuJoCo。机器有 EGL 时可以先执行 `export MUJOCO_GL=egl`。

零动作查看场景：

```bash
python scripts/play.py --agent zero
```

回放训练结果：

```bash
python scripts/play.py --agent trained --checkpoint logs/rsl_rl/zbot_exolimb_grasp/<时间戳>/model_20.pt
```

没有图形界面时查看器会使用 Viser。任务名是 `Mjlab-Zbot-Exolimb-Grasp`。

## 网页演示

打开上面的 Pages 地址后：

- 拖动画面旋转，滚动缩放。
- 「构型」列出基准构型、「中间模块旋转 90°」，以及 32 条合法朝向里选出的代表构型，并显示体素占用和可操作度中位数。
- 六个滑块对应 joint1–joint6，范围 ±6.28 rad。
- 「名义姿态」和「伸直」立即切换姿态。
- 「播放接近演示」用约 3 秒从伸直插值到名义姿态。
- 末端原点到工具原点的距离小于 0.08 m 时，状态变为「进入抓取距离」。

本地预览：

```bash
cd web
npm install
npm run dev
```

开发服务器地址带有 Pages 相同的路径前缀：<http://localhost:5173/zbot_mjlab_exolimb_grasp/>。
