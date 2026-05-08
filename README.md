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

NOTES:
- 数据集需要手动构建，按照 `数据组织规范` 
- 使用复现环境运行则不需要安装下方依赖

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
python scripts/train.py --config configs/default.yaml --exp-config experiments/exp1/resnet18_se.yaml
```

运行 exp1 的 Patch Transformer：

```powershell
python scripts/train.py --config configs/default.yaml --exp-config experiments/exp1/patch_transformer.yaml
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

# 数据集构建说明
本仓库的数据构建由两个脚本负责：scripts\built_dataset.py 和 scripts\data_spilt.py。

1) scripts\built_dataset.py
- 输入：原始 WAV 文件夹（脚本底部的 INPUT_DIR），输出目录由 OUTPUT_DIR 指定。
- 切窗：窗口长度 WIN_LEN=3000，步长 HOP_LEN=750（约 75% 重叠）。
- 产物：对每个窗口生成两类图像并保存为 PNG：
  - Mel 频谱图：使用 librosa 计算 melspectrogram → 转 dB → 线性归一化 → resize 到 IMG_SIZE(默认 224) → 伪彩图（OpenCV）。
  - GADF：使用 pyts.image.GramianAngularField(method='difference') 生成 Gramian Angular Difference Field → 归一化 → 伪彩图。
- 输出文件名模板：{prefix}_{index:05d}_mel.png 和 {prefix}_{index:05d}_gadf.png（prefix 为原 wav 文件名，不含扩展名）。
- 并行：使用 joblib.Parallel，默认并行核数为 os.cpu_count()。
- 配置项（可在脚本顶部修改）：WIN_LEN、HOP_LEN、IMG_SIZE、N_MELS、FMIN、FMAX 等。
- 运行方式：在脚本底部设置 INPUT_DIR/OUTPUT_DIR，运行：
  python scripts\built_dataset.py

2) scripts\data_spilt.py
- 作用：在已有图像（或其它样本）基础上，按类别做平衡并按比例划分为 train/val/test（默认 ratio=[0.6,0.2,0.2]）。
- 假设：src_root 下包含三个子目录（train、val、test），每个子目录包含若干类别子文件夹，且各类别文件命名格式含有按“_”分隔的序号，例如：xxx_0123_mel.png。
- 流程：
  1. 收集每个类别的所有样本并按文件名中第二段数字排序（保证顺序稳定）。
  2. 取所有类别中的最小样本数作为基准，超出部分从序号最大的开始截断以保持顺序。 
  3. 按比例切分（顺序切分，不打乱）并复制到 dst_root 的 train/val/test 分类目录中，每个类别保持文件名对齐。
- 关键点：脚本通过文件名中第二段（parts[1]）提取序号，文件命名需符合该格式。
- 运行方式：修改脚本中的 src_root 和 dst_root 后运行：
  python scripts\data_spilt.py

依赖（建议通过 pip 安装）：
- numpy, scipy, librosa, opencv-python, pyts, joblib, tqdm

注意事项：
- 保证文件命名格式符合脚本约定（prefix_index_type.png），否则 data_spilt.py 的序号提取会失败。
