# AW-DPCNN-based fusion of Mel and GAF images for time series classification.
import numpy as np
from scipy.signal import convolve2d
import os
import cv2
from tqdm import tqdm

def aw_dpcnn_single_channel(S1, S2, n_iter=20):
    rows, cols = S1.shape
    S1 = (S1 - S1.min()) / (S1.max() - S1.min() + 1e-6)
    S2 = (S2 - S2.min()) / (S2.max() - S2.min() + 1e-6)

    alpha_L, alpha_T, V_T, sigma = 0.001, 0.001, 20, 0.1
    W1 = np.array([[-2, 0.5, -2], [-2, 0.5, -2], [-2, 0.5, -2]])
    W2 = np.array([[0, 0.5, 0], [0.5, 0.5, 0.5], [0, 0.5, 0]])
    M = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])

    def get_contrast(img):
        kernel = np.ones((3, 3)) / 9.0
        mu = convolve2d(img, kernel, mode='same')
        mu2 = convolve2d(img ** 2, kernel, mode='same')
        return np.sqrt(np.abs(mu2 - mu ** 2))

    C1, C2 = get_contrast(S1), get_contrast(S2)
    eps = 1e-6
    beta1 = C1 / (C1 + C2 + eps)
    beta2 = C2 / (C1 + C2 + eps)


    L, T, Y = np.zeros_like(S1), np.zeros_like(S1), np.zeros_like(S1)
    U_sum = np.zeros_like(S1)


    for n in range(n_iter):
        feedback1 = convolve2d(Y, W1, mode='same')
        feedback2 = convolve2d(Y, W2, mode='same')
        F1, F2 = feedback1 + S1, feedback2 + S2
        L = np.exp(-alpha_L) * L + convolve2d(Y, M, mode='same')
        U = L * (1 + beta1 * F1) * (1 + beta2 * F2) + sigma
        Y = (U > T).astype(np.float32)
        T = np.exp(-alpha_T) * T + V_T * Y
        U_sum += U

    return (U_sum - U_sum.min()) / (U_sum.max() - U_sum.min() + 1e-6)


def aw_dpcnn_fusion_color(mel_color, gaf_color, n_iter=20):
    if mel_color.shape[:2] != gaf_color.shape[:2]:
        gaf_color = cv2.resize(gaf_color, (mel_color.shape[1], mel_color.shape[0]))
    mel_channels = cv2.split(mel_color)
    gaf_channels = cv2.split(gaf_color)

    fused_channels = []
    for i in range(3):
        channel_fused = aw_dpcnn_single_channel(
            mel_channels[i],
            gaf_channels[i],
            n_iter
        )
        fused_channels.append((channel_fused * 255).astype(np.uint8))

    return cv2.merge(fused_channels)

def batch_aw_dpcnn_fusion_by_class(
        mel_root,
        gaf_root,
        out_root,
        n_iter=20,
        overwrite=False
):
    os.makedirs(out_root, exist_ok=True)

    class_names = sorted([
        d for d in os.listdir(mel_root)
        if os.path.isdir(os.path.join(mel_root, d))
    ])

    print(f'Total classes found: {len(class_names)}')

    for cls in class_names:
        mel_cls_dir = os.path.join(mel_root, cls)
        gaf_cls_dir = os.path.join(gaf_root, cls)
        out_cls_dir = os.path.join(out_root, cls)

        if not os.path.isdir(gaf_cls_dir):
            print(f'[WARN] GAF class missing: {cls}')
            continue

        os.makedirs(out_cls_dir, exist_ok=True)

        mel_files = sorted([
            f for f in os.listdir(mel_cls_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        print(f'[{cls}] {len(mel_files)} files')

        for fname in tqdm(mel_files, desc=f'Fusing {cls}', leave=False):
            mel_path = os.path.join(mel_cls_dir, fname)
            gaf_path = os.path.join(gaf_cls_dir, fname)
            out_path = os.path.join(out_cls_dir, fname)

            if not os.path.exists(gaf_path):
                print(f'[WARN] Missing GAF file: {cls}/{fname}')
                continue

            if os.path.exists(out_path) and not overwrite:
                continue

            mel_img = cv2.imread(mel_path)
            gaf_img = cv2.imread(gaf_path)

            if mel_img is None or gaf_img is None:
                print(f'[ERROR] Read failed: {cls}/{fname}')
                continue

            fused = aw_dpcnn_fusion_color(
                mel_img,
                gaf_img,
                n_iter=n_iter
            )

            cv2.imwrite(out_path, fused)

if __name__ == "__main__":
    mel_root = r'D:\datasets\data5_mel\test'
    gaf_root = r'D:\datasets\data5_gaf\test'
    out_root = r'D:\datasets\data5_awpcnn\test'

    batch_aw_dpcnn_fusion_by_class(
        mel_root=mel_root,
        gaf_root=gaf_root,
        out_root=out_root,
        n_iter=20,
        overwrite=False
    )

    print('Hierarchical batch fusion finished.')