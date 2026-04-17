# This is a script for merging mel spectrograms and GAFs using AW-DPCNN.
# NOTE: The most important for the fusion is that all the pair of Mel and GAF should come from the same segment.
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def extract_index(fname):
    name = os.path.splitext(fname)[0]
    parts = name.split('_')
    if len(parts) < 3:
        return None
    for p in parts:
        if p.isdigit():
            return p
    return None

def aw_dpcnn_single_channel(S1, S2, n_iter=8):
    S1 = S1.astype(np.float32)
    S2 = S2.astype(np.float32)

    S1 = (S1 - S1.min()) / (S1.max() - S1.min() + 1e-6)
    S2 = (S2 - S2.min()) / (S2.max() - S2.min() + 1e-6)

    alpha_L, alpha_T, V_T, sigma = 0.001, 0.001, 20, 0.1

    W1 = np.array([
        [-0.5, -0.5, 1, -0.5, -0.5],
        [-0.5, -0.5, 1, -0.5, -0.5],
        [-0.5, -0.5, 1, -0.5, -0.5]
    ], np.float32)

    W2 = np.array([
        [0, 0.01, 0],
        [0.01, 0.01, 0.01],
        [0, 0.01, 0]
    ], np.float32)

    M = np.eye(3, dtype=np.float32) * 0.25
    kernel = np.ones((3, 3), np.float32) / 9.0

    def local_contrast(img):
        mu = cv2.filter2D(img, -1, kernel)
        mu2 = cv2.filter2D(img * img, -1, kernel)
        return np.sqrt(np.abs(mu2 - mu * mu))

    C1 = local_contrast(S1)
    C2 = local_contrast(S2)

    gamma = 10
    beta1 = gamma * C1 / (gamma * C1 + C2 + 1e-6)
    beta2 = C2 / (gamma * C1 + C2 + 1e-6)

    L = np.zeros_like(S1)
    T = np.zeros_like(S1)
    Y = np.zeros_like(S1)
    U_sum = np.zeros_like(S1)

    for _ in range(n_iter):
        F1 = cv2.filter2D(Y, -1, W1) + S1
        F2 = cv2.filter2D(Y, -1, W2) + S2
        L = np.exp(-alpha_L) * L + cv2.filter2D(Y, -1, M)

        U = L * (1 + beta1 * F1) * (1 + beta2 * F2) + sigma
        Y = (U > T).astype(np.float32)
        T = np.exp(-alpha_T) * T + V_T * Y
        U_sum += U

    return (U_sum - U_sum.min()) / (U_sum.max() - U_sum.min() + 1e-6)

def aw_dpcnn_fusion_color(mel_img, gaf_img, n_iter=8):
    if mel_img.shape[:2] != gaf_img.shape[:2]:
        gaf_img = cv2.resize(gaf_img, (mel_img.shape[1], mel_img.shape[0]))

    mel_ch = cv2.split(mel_img)
    gaf_ch = cv2.split(gaf_img)

    fused = []
    for i in range(3):
        f = aw_dpcnn_single_channel(mel_ch[i], gaf_ch[i], n_iter)
        fused.append((f * 255).astype(np.uint8))

    return cv2.merge(fused)

def fuse_one_image(args):
    mel_path, gaf_path, out_path, n_iter, overwrite = args

    if os.path.exists(out_path) and not overwrite:
        return 0

    mel_img = cv2.imread(mel_path)
    gaf_img = cv2.imread(gaf_path)

    if mel_img is None or gaf_img is None:
        return 0

    fused = aw_dpcnn_fusion_color(mel_img, gaf_img, n_iter)
    cv2.imwrite(out_path, fused)
    return 1

def batch_aw_dpcnn_fusion_parallel(
    mel_root,
    gaf_root,
    out_root,
    n_iter=8,
    overwrite=False,
    num_workers=None
):
    os.makedirs(out_root, exist_ok=True)
    tasks = []

    class_names = sorted(d for d in os.listdir(mel_root)
                         if os.path.isdir(os.path.join(mel_root, d)))

    for cls in class_names:
        mel_cls = os.path.join(mel_root, cls)
        gaf_cls = os.path.join(gaf_root, cls)
        out_cls = os.path.join(out_root, cls)

        if not os.path.isdir(gaf_cls):
            continue

        os.makedirs(out_cls, exist_ok=True)

        mel_map = {}
        for f in os.listdir(mel_cls):
            idx = extract_index(f)
            if idx:
                mel_map[idx] = f

        gaf_map = {}
        for f in os.listdir(gaf_cls):
            idx = extract_index(f)
            if idx:
                gaf_map[idx] = f

        common_indices = sorted(mel_map.keys() & gaf_map.keys())

        for idx in common_indices:
            mel_path = os.path.join(mel_cls, mel_map[idx])
            gaf_path = os.path.join(gaf_cls, gaf_map[idx])
            out_path = os.path.join(out_cls, mel_map[idx])

            tasks.append((mel_path, gaf_path, out_path, n_iter, overwrite))

        print(f"✔ 类别 {cls} 融合样本数: {len(common_indices)}")

    print(f"\n✅ 总融合任务数: {len(tasks)}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(fuse_one_image, tasks),
                total=len(tasks),
                desc="AW-DPCNN Fusion",
                ncols=100
            )
        )

if __name__ == '__main__':
    mel_root = '/home/kemove/PycharmProjects/built_dataset/data3/mel/test'
    gaf_root = '/home/kemove/PycharmProjects/built_dataset/data3/gadf/test'
    out_root = '/home/220242215063/pycharm_project_legion/AW-DPCNN/data4/test'

    batch_aw_dpcnn_fusion_parallel(
        mel_root,
        gaf_root,
        out_root,
        n_iter=20,
        overwrite=False,
        num_workers=os.cpu_count() - 1
    )

    print("\n🎉 AW-DPCNN 融合完成（按 index 精确对齐）")
