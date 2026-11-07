# 模块化可重构臂综合设计与参数优化平台

[在线演示（部署后可直接打开运行）](https://zhuyanhe1975-hit.github.io/zbot_mjlab_exolimb_grasp/)

这是一个串联可重构臂的设计平台：模块库和接口规则决定哪些模块能装在一起，构型搜索再决定模块类型、数量、次序和连接方位，并与允许的连续结构参数一起做多目标优化。ZBot R3 是第一套放进模块库的实物，用来回归已核对的运动学；它不是平台唯一能表达的机构。抓取强化学习还留在仓库里，但不是这个设计工具的主体。

设计页面在 `web/design.html`。上面的链接仍是原来的抓取运动学演示；Pages 只在推送到 `main` 之后更新，未合并的分支不会出现在这个地址上。

## 使用

需要 Python 3.10 和 NumPy。在仓库根目录：

```bash
python -m unittest tests.test_platform
python scripts/design_optimize.py --mode mixed --count-min 3 --count-max 5 --pop 12 --gen 5 --samples 180 --seed 1 --out experiments/platform_seed1.json
```

`--mode physical` 只搜索 ZBot R3 的数量和连接方位，不缩放实物尺寸。`--mode mixed` 允许插入轴向概念连杆，并优化它的长度。网页里的搜索在 Web Worker 中完成，预算低于命令行；两边用同一套目标函数和约束，但随机数流不同，同一种子不会逐个个体相同。

本地打开设计页：

```bash
cd web
npm install
npm run dev
```

然后打开 <http://localhost:5173/zbot_mjlab_exolimb_grasp/design.html>。

架构、指标边界和后续验收见 [docs/平台架构与验收.md](docs/平台架构与验收.md)。

## 既有抓取任务与早期构型对比

早期页面可以切换六模块构型。基准构型的五个连接朝向都是 0，关节轴仍是 `0 0 1`，与 `assets/zbot_6/ZBOT6.xml` 一致。`python scripts/evaluate_config.py` 和 `python scripts/evaluate_batch.py` 仍可复现那次 32 条合法朝向的采样对比，结果在 `web/public/config_metrics.json` 与 `web/public/config_batch.csv`。那不是本平台的优化器。

本仓库因此还有两部分旧工作：

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
