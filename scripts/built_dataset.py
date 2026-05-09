# 可用
import os
import cv2
import numpy as np
from scipy.io import wavfile
import librosa
from pyts.image import GramianAngularField
from tqdm import tqdm
from joblib import Parallel, delayed

# =========================================================
# 参数区
# =========================================================
WIN_LEN = 3000          # 滑窗长度
HOP_LEN = 750           # 步长（75% overlap）
IMG_SIZE = 224

N_MELS = 128
FMIN = 20
FMAX = 8000

N_JOBS = os.cpu_count()  # 并行核数

# =========================================================
# 灰度 → 伪彩图（OpenCV 加速）
# =========================================================
def gray_to_pseudo(gray_uint8, cmap=cv2.COLORMAP_TURBO):
    """
    gray_uint8: (H, W) uint8
    return: (H, W, 3) uint8
    """
    return cv2.applyColorMap(gray_uint8, cmap)


# =========================================================
# Mel Spectrogram → 伪彩图
# =========================================================
def generate_mel_image(signal, sr):
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=2048,
        hop_length=max(1, WIN_LEN // IMG_SIZE),
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_uint8 = (mel_norm * 255).astype(np.uint8)

    mel_uint8 = cv2.resize(mel_uint8, (IMG_SIZE, IMG_SIZE))
    mel_color = gray_to_pseudo(mel_uint8)

    return mel_color


# =========================================================
# GADF → 伪彩图
# =========================================================
gadf_transformer = GramianAngularField(
    image_size=IMG_SIZE,
    method='difference'
)

def generate_gadf_image(signal):
    signal = signal.astype(np.float32)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    signal = signal * 2.0 - 1.0
    signal = np.clip(signal, -1.0, 1.0)

    gadf = gadf_transformer.fit_transform(signal.reshape(1, -1))[0]

    gadf_norm = (gadf - gadf.min()) / (gadf.max() - gadf.min() + 1e-8)
    gadf_uint8 = (gadf_norm * 255).astype(np.uint8)

    gadf_color = gray_to_pseudo(gadf_uint8)

    return gadf_color


# =========================================================
# 单窗口处理
# =========================================================
def process_one_window(idx, window, sr, prefix, out_dir):
    mel_img = generate_mel_image(window, sr)
    gadf_img = generate_gadf_image(window)

    mel_path = os.path.join(out_dir, f'{prefix}_{idx:05d}_mel.png')
    gadf_path = os.path.join(out_dir, f'{prefix}_{idx:05d}_gadf.png')

    cv2.imwrite(mel_path, mel_img)
    cv2.imwrite(gadf_path, gadf_img)


# =========================================================
# 单 wav 文件处理（并行）
# =========================================================
def process_wav(wav_path, out_dir):
    sr, signal = wavfile.read(wav_path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    signal = signal.astype(np.float32)

    total_len = len(signal)
    prefix = os.path.splitext(os.path.basename(wav_path))[0]

    tasks = []
    idx = 0
    for start in range(0, total_len - WIN_LEN + 1, HOP_LEN):
        window = signal[start:start + WIN_LEN]
        tasks.append((idx, window))
        idx += 1

    Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(process_one_window)(i, w, sr, prefix, out_dir)
        for i, w in tasks
    )


# =========================================================
# 文件夹级处理
# =========================================================
def process_wav_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    wav_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, f))

    print(f'[INFO] Found {len(wav_files)} wav files')

    for wav_path in tqdm(wav_files, desc='Processing WAV files'):
        process_wav(wav_path, output_dir)


# =========================================================
# 主入口
# =========================================================
if __name__ == '__main__':
    INPUT_DIR = 'dataset/test'
    OUTPUT_DIR = 'data1/test'

    process_wav_folder(INPUT_DIR, OUTPUT_DIR)
