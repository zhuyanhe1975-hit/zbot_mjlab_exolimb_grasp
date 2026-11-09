# generative-robot-design（摘录）

上游：https://github.com/JonathanKuelz/generative-robot-design  
许可证：MIT，见 `LICENSE`。`transforms/batched.py` 含 PyTorch3D 的 BSD 声明。  
钉住的提交写在 `PINNED_COMMIT`。

这里只放入运动学核心（`robot_modules`、`transforms`、`robot`），用来按
`Custom(proximal, distal)` 做可微正运动学。不包含设计网络训练，也不跑官方穷举基线。

论文里的三种参数化在上游是 Hydra 配置，不是三个类名：

- `modular`：`uv run python main.py`，模块来自 `robot_modules/robco_modules.py`
- `economic`：`constraints=economic`，`goals.parameter_convention=dh`
- `free`：`constraints=free`，`goals.parameter_convention=dh`

`transforms/parameterized.py` 里的枚举是 `MDH`、`DH`、`ROBCO`。
ZBot 的 0/π/2 接口用两个 `Custom` 模块表达，因为 Robco 的 D86 模块族不是 ZBot。
