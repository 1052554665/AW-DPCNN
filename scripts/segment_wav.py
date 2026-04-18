# This script is used to segment wav files with overlapping segments.
# It reads wav files from the input folder, segment them into segments of specified length with
# a certain overlap ratio, and saves the segments to the output folder.
# The script also handles audio resampling and channel conversion to ensure consistency in the output segments.
import os
from pydub import AudioSegment

def spilt_wav_with_overlap(input_folder, output_folder,
                           segment_length_ms=1000, overlap_ratio=0.6):
    target_sample_rate = 44100
    step_size = int(segment_length_ms * (1 - overlap_ratio))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_files = 0
    total_segments = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)

            try:
                audio = AudioSegment.from_wav(file_path)
            except Exception as e:
                print(f"读取文件出错：{filename}, 错误：{e}")
                continue

            if audio.channels > 1:
                audio = audio.set_channels(1)
            audio = audio.set_frame_rate(target_sample_rate)

            duration_ms = len(audio)
            base_filename, _ = os.path.splitext(filename)
            segment_count = 0


            for i in range(0, duration_ms - segment_length_ms + 1, step_size):
                segment = audio[i: i + segment_length_ms]
                if len(segment) < segment_length_ms:
                    continue
                segment_filename = f"{base_filename}_seg{segment_count}.wav"
                output_path = os.path.join(output_folder, segment_filename)
                segment.export(output_path, format="wav")
                segment_count += 1
                print(f"{filename}切割完成，共生成{segment_count}段")
                total_files += 1
                total_segments += segment_count
        print(f"\n所有文件处理完成，共处理 {total_files} 个文件，生成 {total_segments} 段音频")


input_folder = r"D:\datasets\data\测试"
output_folder = r"D:\datasets\data1\test"
spilt_wav_with_overlap(input_folder, output_folder, segment_length_ms =
1000, overlap_ratio = 0.6)

