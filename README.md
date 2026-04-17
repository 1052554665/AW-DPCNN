# GAF
这份脚本整体实现了“批量把 wav 音频转换为 GAF 图像”的完整流水线，可直接用于数据集预处理的批量 GAF 生成器，支持灰度/彩色可切换、并行加速和目录结构自动对齐。

主要实现功能如下：

1. 音频预处理  
把原始 wav 统一成可做 GAF 变换的标准输入。

2. GAF 图像生成  
使用 GramianAngularField 进行变换，并支持两种输出：
    - 灰度图：output_mode='gray'
    - 彩色图：output_mode='rgb'，通过 colormap 映射（默认 viridis）

3. 批量任务构建与目录结构保留  
根据输入目录的相对路径生成输出路径，保证输出目录结构与输入一致（类别子目录不会丢失）。

4. 多进程并行处理  
使用 ProcessPoolExecutor 并行处理所有 wav，并配合 tqdm 显示进度条。

5. 主入口一键运行  
给出了默认输入/输出路径和参数，当前默认是彩色输出（output_mode='rgb'）。

# Mel
## 主要功能

1. 单文件处理  
- 模式：single  
- 支持训练图风格（无坐标轴）和论文图风格（含坐标轴、色条、标题）

2. 批量处理  
- 模式：batch  
- 递归遍历目录、保持子目录结构、支持多进程并行

3. Mel 曲线绘制  
- 模式：curve  
- 直接生成 Mel 频率尺度曲线图，替代原独立脚本功能

4. 参数统一可配  
- 采样率、n_fft、hop_length、n_mels、fmax、cmap、图像尺寸、并行进程数都可命令行设置

## 可直接运行示例

1. 批量生成训练图（无坐标轴）
python mel_unified.py --mode batch --style dataset --input-dir /home/220242215063/pycharm_project_legion/data5 --output-dir /home/220242215063/pycharm_project_legion/data5_mel --workers 16

2. 单文件生成论文图（含坐标轴和色条）
python mel_unified.py --mode single --style paper --input-wav Normal_part0.wav --output-image Mel_spectrogram.png --sr 0 --n-mels 128 --fmax 8000

3. 绘制 Mel 频率曲线
python mel_unified.py --mode curve --curve-save Mel_filter_curve.png

