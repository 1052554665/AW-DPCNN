# AW-DPCNN Research Workflow

本项目已整理为可复现的科研实验工作流，核心目标是：

- 统一训练与评估入口
- YAML 配置驱动实验
- 自动生成实验目录与结果产物
- 保留现有模型与预处理脚本

## 1) 项目结构（核心）

- `configs/`: 基础配置与实验覆盖配置
- `scripts/train.py`: 统一训练入口
- `scripts/evaluate.py`: 统一评估入口
- `src/datasets/`: 数据加载构建
- `src/models/`: 模型实现与注册
- `src/trainers/workflow.py`: 训练/验证/测试主流程
- `src/utils/`: 指标、绘图、配置、实验管理工具
- `experiments/runs/`: 每次运行自动创建独立目录

## 2) 数据组织规范

训练默认使用 `ImageFolder` 格式：

```text
data/processed/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
  test/
    class_a/
    class_b/
```

可在 `configs/default.yaml` 修改 `data.root_dir` 与 split 名称。

## 3) 配置体系

- `configs/default.yaml`: 全局默认配置
- `configs/exp1.yaml`: ResNet18-SE 实验覆盖
- `configs/exp2.yaml`: ViT 实验覆盖

训练时采用“基础配置 + 覆盖配置”合并。

## 4) 快速运行

先安装依赖：

```powershell
pip install -r requirements.txt
```

基线训练：

```powershell
python scripts/train.py --config configs/default.yaml
```

运行 exp1（覆盖默认配置）：

```powershell
python scripts/train.py --config configs/default.yaml --exp-config configs/exp1.yaml
```

使用已有 checkpoint 评估：

```powershell
python scripts/evaluate.py --config experiments/runs/<run_name>/resolved_config.yaml --checkpoint experiments/runs/<run_name>/checkpoints/best.pt --output-dir experiments/runs/<run_name>/eval
```

## 5) 运行产物

每次训练会在 `output.root_dir` 下创建新目录，包含：

- `resolved_config.yaml`
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `logs/train_log.csv`
- `results/test_metrics.json`
- `figures/confusion_matrix.png`
- `figures/tsne.png`（启用时）

## 6) 旧脚本说明

`scripts/` 下原有音频预处理脚本（如 `mel.py`, `GAF.py`, `segment_wav.py`）仍可继续使用，用于数据构建阶段；
模型训练与评估建议统一通过 `train.py`/`evaluate.py` 执行，以保证实验记录一致。