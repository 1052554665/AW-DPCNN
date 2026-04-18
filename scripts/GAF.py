# 若需转灰度，请将 output_mode 参数设置为 'gray'，并确保 resize 参数不为 None。
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from pyts.image import GramianAngularField
from scipy.io import wavfile
from tqdm import tqdm


DEFAULT_INPUT_DIR = '/home/220242215063/data5'
DEFAULT_OUTPUT_DIR = '/home/220242215063/pycharm_project_legion/data5_gaf'


def _to_mono_float32(data):
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32, copy=False)


def _fix_length(data, target_length):
    if len(data) >= target_length:
        return data[:target_length]
    return np.pad(data, (0, target_length - len(data)), mode='constant')


def _normalize_to_minus_one_one(data):
    eps = 1e-8
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if abs(max_val - min_val) < eps:
        return np.zeros_like(data, dtype=np.float32)
    data = 2.0 * (data - min_val) / (max_val - min_val + eps) - 1.0
    return np.clip(data, -1.0, 1.0).astype(np.float32, copy=False)


def generate_gaf_image(
    wav_path,
    save_path,
    sequence_length=3000,
    method='difference',
    resize=(224, 224),
    output_mode='gray',
    cmap_name='viridis',
):
    _, data = wavfile.read(wav_path)
    data = _to_mono_float32(data)
    data = _fix_length(data, sequence_length)
    data = _normalize_to_minus_one_one(data)

    gaf = GramianAngularField(image_size=sequence_length, method=method)
    gaf_image = gaf.fit_transform(data.reshape(1, -1))[0]

    image_min = float(np.min(gaf_image))
    image_max = float(np.max(gaf_image))
    eps = 1e-8

    if output_mode == 'rgb':
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgb = (cmap(gaf_image)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode='RGB')
    else:
        normalized = (gaf_image - image_min) / (image_max - image_min + eps)
        gray = (normalized * 255).astype(np.uint8)
        img = Image.fromarray(gray, mode='L')

    if resize is not None:
        img = img.resize(resize, Image.BICUBIC)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def _build_task(
    wav_path,
    output_dir,
    input_dir,
    sequence_length,
    method,
    resize,
    output_mode,
    cmap_name,
):
    rel_dir = os.path.relpath(os.path.dirname(wav_path), input_dir)
    out_dir = os.path.join(output_dir, rel_dir)
    save_name = os.path.splitext(os.path.basename(wav_path))[0] + '.png'
    save_path = os.path.join(out_dir, save_name)
    return (
        wav_path,
        save_path,
        sequence_length,
        method,
        resize,
        output_mode,
        cmap_name,
    )


def _process_one(task):
    (
        wav_path,
        save_path,
        sequence_length,
        method,
        resize,
        output_mode,
        cmap_name,
    ) = task
    try:
        generate_gaf_image(
            wav_path=wav_path,
            save_path=save_path,
            sequence_length=sequence_length,
            method=method,
            resize=resize,
            output_mode=output_mode,
            cmap_name=cmap_name,
        )
        return 1
    except Exception as exc:
        print(f'[ERROR] {wav_path}: {exc}')
        return 0


def batch_generate_gaf(
    input_dir,
    output_dir,
    sequence_length=3000,
    method='difference',
    resize=(224, 224),
    output_mode='gray',
    cmap_name='viridis',
    num_workers=32,
):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            wav_path = os.path.join(root, fname)
            tasks.append(
                _build_task(
                    wav_path=wav_path,
                    output_dir=output_dir,
                    input_dir=input_dir,
                    sequence_length=sequence_length,
                    method=method,
                    resize=resize,
                    output_mode=output_mode,
                    cmap_name=cmap_name,
                )
            )

    total = len(tasks)
    print(f'[INFO] Total wav files: {total}')
    os.makedirs(output_dir, exist_ok=True)

    if total == 0:
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(_process_one, tasks),
                total=total,
                desc='Generating GAF',
                ncols=100,
            )
        )


if __name__ == '__main__':
    batch_generate_gaf(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        sequence_length=3000,
        method='difference',
        resize=(224, 224),
        output_mode='rgb',  # 若需转灰度，将 output_mode 参数设置为 'gray'
        cmap_name='viridis',
        num_workers=32,
    )
