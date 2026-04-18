# 文件夹下有三个子文件夹，三个子文件夹下分别有着对应相同的10个文件夹
# 实现：划分数据集，train：val：test=6：2：2,同时各个类别之间保持平衡，如某类数据过多可以删除部分样本
# 如某类数据过多可以删除部分样本，删除时以文件名中两个_之间的数字为序号，从序号最大的开始从大到小删除，从而保证顺序不乱

import os
import shutil
import re

# ========= 路径配置 =========
src_root = "/home/kemove/PycharmProjects/built_dataset/data2/mel"
dst_root = "/home/kemove/PycharmProjects/built_dataset/data3/mel"
splits = ["train", "val", "test"]
ratio = [0.6, 0.2, 0.2]
img_ext = (".png", ".jpg", ".jpeg", ".bmp")
# ===========================


def extract_index(filename: str) -> int:
    """
    从文件名中提取两个 '_' 之间的数字
    例如: xxx_0123_mel.png → 123
    """
    name = os.path.basename(filename)
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"文件名格式不符合规则: {name}")
    return int(parts[1])


# 1️⃣ 读取类别名（以 train 为准）
class_names = sorted(os.listdir(os.path.join(src_root, "train")))
print(f"📂 类别数: {len(class_names)}")
print(class_names)

# 2️⃣ 收集每个类别的全部样本（不打乱）
all_class_files = {}

for cls in class_names:
    files = []
    for sp in splits:
        cls_dir = os.path.join(src_root, sp, cls)
        if not os.path.exists(cls_dir):
            continue

        for f in os.listdir(cls_dir):
            if f.lower().endswith(img_ext):
                files.append(os.path.join(cls_dir, f))

    # 🔑 关键：按“序号”排序（小 → 大）
    files.sort(key=lambda x: extract_index(x))

    all_class_files[cls] = files

# 3️⃣ 计算平衡样本数（取最小）
min_count = min(len(v) for v in all_class_files.values())
print(f"\n⚖️ 平衡后每类样本数: {min_count}")

# 4️⃣ 创建目标目录
for sp in splits:
    for cls in class_names:
        os.makedirs(os.path.join(dst_root, sp, cls), exist_ok=True)

# 5️⃣ 按顺序划分 & 复制
for cls, files in all_class_files.items():
    print(f"\n▶ 处理类别 {cls} | 原始样本数: {len(files)}")

    # 🔑 核心操作：如果过多 → 从后往前删
    files = files[:min_count]

    n_train = int(min_count * ratio[0])
    n_val = int(min_count * ratio[1])
    n_test = min_count - n_train - n_val

    split_files = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    for sp, flist in split_files.items():
        dst_dir = os.path.join(dst_root, sp, cls)
        for src_path in flist:
            fname = os.path.basename(src_path)
            shutil.copy2(src_path, os.path.join(dst_dir, fname))

    print(f"   ✔ train:{n_train} val:{n_val} test:{n_test}")

print("\n✅ 新数据集已生成（顺序保持 & Mel/GADF 完全对齐）")
