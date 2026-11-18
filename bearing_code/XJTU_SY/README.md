# XJTU-SY 退化阶段识别与剩余寿命预测

本模块实现 XJTU-SY 轴承全寿命数据的预处理、GAT-BiLSTM 模型训练、退化阶段识别、RUL 回归评估及结果绘图。

## 运行

在本目录打开 PowerShell，执行：

```powershell
& 'D:/XJTU_SY_runtime/venv/Scripts/python.exe' experiment.py all --config config.json --resume
```

后台运行：

```powershell
& .\run_background.ps1
```

查看进度：

```powershell
& 'D:/XJTU_SY_runtime/venv/Scripts/python.exe' progress.py
```

配置文件为 `config.json`。默认数据目录为 `D:/Desktop/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets`，缓存目录为 `D:/XJTU_SY_runtime/cache`。在其他电脑上运行时，修改这两个目录并安装 `requirements.txt` 中的依赖；后台脚本可通过 `-Python` 指定 Python 路径。无 CUDA 设备时，将配置中的 `device` 改为 `cpu`。

## 数据划分与任务

每种工况下，轴承 1、2、3 用于训练，轴承 4 用于验证，轴承 5 用于测试。共计 9 个训练轴承、3 个验证轴承和 3 个测试轴承。

使用水平振动通道，每个采集文件划分为 32 个不重叠的 1024 点窗口，提取时域序列和频域幅值谱。归一化参数仅在训练数据上拟合。最近 10 个采集时刻构成模型输入历史片段，图构建和序列建模只使用当前及过去的观测。

退化阶段由振动 RMS 相对初始 10 次采集基线的变化定义，阈值为 3 和 6，用于描述振动退化阶段，不对应物理缺陷尺寸。RUL 标签为数据记录终点与当前采集时刻的时间差，单位为分钟。

两个任务分别训练独立模型。阶段识别模型按验证集 Macro-F1 选择检查点，RUL 模型按验证集 MAE 选择检查点，然后在独立测试轴承上评估。修改实验设置重新训练时，应指定新的 `output_dir`；修改预处理设置时还应指定新的 `cache_dir`。

## 文件

- `data.py`：数据读取、预处理、标签与历史片段构建。
- `model.py`：卷积编码、超图聚合、注意力融合及 BiLSTM 模型。
- `experiment.py`：训练、验证、测试与流程入口。
- `figures.py`：数据划分表、测试汇总表、阶段识别图和 RUL 曲线。
- `config.json`：数据路径、超参数及数据划分。
- `run_background.ps1`：后台运行入口。
- `progress.py`：进度查询。
- `test_pipeline.py`：数据处理与模型完整性检查。
- `test_end_to_end.py`：使用独立合成测试数据检查程序流程。
- `verify_outputs.py`：运行后核验预测、指标与文件一致性。

运行生成的文件保存在 `results`，日志保存在 `logs`。
