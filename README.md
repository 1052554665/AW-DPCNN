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

