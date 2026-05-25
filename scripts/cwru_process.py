import os
import numpy as np
import scipy.io as sio
import librosa
import librosa.display
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from pyts.image import GramianAngularField

# ==============================
# 全局参数（论文级推荐配置）
# ==============================
SR = 12000                 # 采样率
WIN_LEN = 2048             # 滑窗长度
STEP = 1024                # 滑窗步长（50% overlap）
MEL_NFFT = 1024
MEL_HOP = 512
MEL_BINS = 128
IMG_SIZE = 224

CLASS_MAP = {
    'Normal': 0,
    'IR': 1,
    'OR': 2,
    'B': 3
}
CLASS_NAMES = list(CLASS_MAP.keys())


# ==============================
# 1. 读取 CWRU .mat 文件
# ==============================
def load_cwru_mat(mat_path):
    mat = sio.loadmat(mat_path)
    for key in mat.keys():
        if 'DE_time' in key:
            return mat[key].squeeze()
    raise ValueError(f'No DE_time found in {mat_path}')


# ==============================
# 2. 滑窗切片
# ==============================
def sliding_window(signal, win_len=WIN_LEN, step=STEP):
    segments = []
    for start in range(0, len(signal) - win_len, step):
        seg = signal[start:start + win_len]
        segments.append(seg)
    return np.array(segments)


# ==============================
# 3. 构建样本池
# ==============================
def build_samples(raw_root):
    X, y = [], []

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(raw_root, cls)
        for file in os.listdir(cls_dir):
            if not file.endswith('.mat'):
                continue

            mat_path = os.path.join(cls_dir, file)
            signal = load_cwru_mat(mat_path)
            segments = sliding_window(signal)

            X.extend(segments)
            y.extend([CLASS_MAP[cls]] * len(segments))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f'[INFO] Total samples: {len(X)}')
    return X, y


# ==============================
# 4. 数据集划分（6:2:2）
# ==============================
def split_dataset(X, y):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test)
    )


# ==============================
# 5. Mel 频谱提取
# ==============================
def extract_mel(signal):
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SR,
        n_fft=MEL_NFFT,
        hop_length=MEL_HOP,
        n_mels=MEL_BINS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def save_mel_image(mel, save_path):
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(mel, cmap='jet')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(save_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(save_path)


# ==============================
# 6. GADF 提取
# ==============================
def extract_gadf(signal):
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    gaf = GramianAngularField(
        image_size=IMG_SIZE,
        method='difference'
    )
    gadf = gaf.fit_transform(signal.reshape(1, -1))
    return gadf[0]


def save_gadf_image(gadf, save_path):
    img = Image.fromarray((gadf * 255).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(save_path)


# ==============================
# 7. 生成图像数据集
# ==============================
def generate_images(X, y, split_name, save_root, feature):
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(save_root, split_name, cls), exist_ok=True)

    for idx, (signal, label) in enumerate(zip(X, y)):
        cls = CLASS_NAMES[label]
        save_path = os.path.join(
            save_root, split_name, cls, f'{idx}.png'
        )

        if feature == 'mel':
            mel = extract_mel(signal)
            save_mel_image(mel, save_path)

        elif feature == 'gadf':
            gadf = extract_gadf(signal)
            save_gadf_image(gadf, save_path)

        if idx % 500 == 0:
            print(f'[{feature.upper()}] {split_name}: {idx}')


# ==============================
# 8. 主函数
# ==============================
def main():
    raw_root = 'cwru_raw'
    mel_root = 'datasets_mel'
    gadf_root = 'datasets_gadf'

    print('[STEP 1] Loading and slicing signals...')
    X, y = build_samples(raw_root)

    print('[STEP 2] Splitting dataset...')
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_dataset(X, y)

    print('[STEP 3] Generating Mel dataset...')
    generate_images(X_tr, y_tr, 'train', mel_root, 'mel')
    generate_images(X_va, y_va, 'val', mel_root, 'mel')
    generate_images(X_te, y_te, 'test', mel_root, 'mel')

    print('[STEP 4] Generating GADF dataset...')
    generate_images(X_tr, y_tr, 'train', gadf_root, 'gadf')
    generate_images(X_va, y_va, 'val', gadf_root, 'gadf')
    generate_images(X_te, y_te, 'test', gadf_root, 'gadf')

    print('✅ Dataset generation completed.')


if __name__ == '__main__':
    main()