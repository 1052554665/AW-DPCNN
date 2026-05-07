import argparse
import os
from multiprocessing import Pool, cpu_count

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_mel_db(
    wav_path,
    sr=None,
    n_fft=2048,
    hop_length=256,
    n_mels=256,
    fmax=8000,
    power=2.0,
):
    y, sr = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        power=power,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, sr


def save_mel_image(
    mel_db,
    sr,
    save_path,
    hop_length,
    fmax,
    style="dataset",
    figsize=(4, 4),
    dpi=56,
    cmap="viridis",
    title="Mel Spectrogram",
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if style == "paper":
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear",
            fmax=fmax,
            cmap=cmap,
            ax=ax,
        )
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Frequency(kHz)")

        yticks = np.linspace(0, fmax, 9)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(v / 1000)}" for v in yticks])

        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
        cbar.set_label("Power/Frequency(dB/Hz)")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0)
    else:
        librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=hop_length,
            x_axis=None,
            y_axis=None,
            fmax=fmax,
            cmap=cmap,
            ax=ax,
        )
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)


def process_single(
    wav_path,
    save_path,
    sr,
    n_fft,
    hop_length,
    n_mels,
    fmax,
    power,
    style,
    figsize,
    dpi,
    cmap,
    title,
):
    mel_db, used_sr = compute_mel_db(
        wav_path=wav_path,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        power=power,
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    save_mel_image(
        mel_db=mel_db,
        sr=used_sr,
        save_path=save_path,
        hop_length=hop_length,
        fmax=fmax,
        style=style,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=title,
    )


def _worker(task):
    (
        wav_path,
        save_path,
        sr,
        n_fft,
        hop_length,
        n_mels,
        fmax,
        power,
        style,
        figsize,
        dpi,
        cmap,
        title,
    ) = task
    try:
        process_single(
            wav_path=wav_path,
            save_path=save_path,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            power=power,
            style=style,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
            title=title,
        )
        return 1
    except Exception as exc:
        print(f"[ERROR] {wav_path}: {exc}")
        return 0


def collect_tasks(
    input_dir,
    output_dir,
    ext,
    sr,
    n_fft,
    hop_length,
    n_mels,
    fmax,
    power,
    style,
    figsize,
    dpi,
    cmap,
    title,
):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(ext.lower()):
                continue
            wav_path = os.path.join(root, name)
            rel_dir = os.path.relpath(root, input_dir)
            out_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            save_name = os.path.splitext(name)[0] + ".png"
            save_path = os.path.join(out_dir, save_name)
            tasks.append(
                (
                    wav_path,
                    save_path,
                    sr,
                    n_fft,
                    hop_length,
                    n_mels,
                    fmax,
                    power,
                    style,
                    figsize,
                    dpi,
                    cmap,
                    title,
                )
            )
    return tasks


def process_batch(args):
    tasks = collect_tasks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ext=args.ext,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmax=args.fmax,
        power=args.power,
        style=args.style,
        figsize=(args.fig_w, args.fig_h),
        dpi=args.dpi,
        cmap=args.cmap,
        title=args.title,
    )

    print(f"[INFO] Total files: {len(tasks)}")
    if not tasks:
        return

    if args.workers <= 1:
        for task in tqdm(tasks, total=len(tasks), desc="Mel Spectrograms", ncols=100):
            _worker(task)
        return

    workers = min(args.workers, max(cpu_count(), 1))
    with Pool(processes=workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc="Mel Spectrograms",
                ncols=100,
            )
        )


def plot_mel_curve(save_path="Mel_filter_curve.png"):
    frequencies = np.linspace(20, 20000, 500)
    mel_values = 2595 * np.log10(1 + frequencies / 700)

    plt.figure(figsize=(8, 6))
    plt.plot(frequencies, mel_values)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mel Frequency (Mel)")
    plt.grid(False)
    plt.savefig(save_path, dpi=300, pad_inches=0)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Unified Mel spectrogram tool")

    parser.add_argument("--mode", choices=["single", "batch", "curve"], default="batch")
    parser.add_argument("--style", choices=["dataset", "paper"], default="dataset")

    parser.add_argument("--input-wav", default="Normal_part0.wav")
    parser.add_argument("--output-image", default="Mel_spectrogram.png")

    parser.add_argument("--input-dir", default="/home/220242215063/pycharm_project_legion/data5")
    parser.add_argument("--output-dir", default="/home/220242215063/pycharm_project_legion/data5_mel")
    parser.add_argument("--ext", default=".wav")

    parser.add_argument("--sr", type=int, default=0, help="0 means keep original sample rate")
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=256)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--power", type=float, default=2.0)

    parser.add_argument("--fig-w", type=float, default=4.0)
    parser.add_argument("--fig-h", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=56)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--title", default="Mel Spectrogram")

    parser.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1))
    parser.add_argument("--curve-save", default="Mel_filter_curve.png")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Align with librosa API: None means keeping the original sample rate.
    sr = None if args.sr == 0 else args.sr
    args.sr = sr

    if args.mode == "curve":
        plot_mel_curve(save_path=args.curve_save)
        print(f"[INFO] Mel curve saved to: {args.curve_save}")
        return

    if args.mode == "single":
        process_single(
            wav_path=args.input_wav,
            save_path=args.output_image,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmax=args.fmax,
            power=args.power,
            style=args.style,
            figsize=(args.fig_w, args.fig_h),
            dpi=args.dpi,
            cmap=args.cmap,
            title=args.title,
        )
        print(f"[INFO] Image saved to: {args.output_image}")
        return

    process_batch(args)


if __name__ == "__main__":
    main()
