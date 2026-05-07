# 单个音频读取 + 重采样 + 时长计算 + 波形可视化
import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy.io import wavfile

# 步骤 1：读取原始音频（保留原采样率信息）
original_sample_rate, data = wavfile.read('11/22/Loosen1.wav')
print("Original sample rate:", original_sample_rate)

# 如果是立体声，转换为单声道
if len(data.shape) == 2:
    data = data.mean(axis=1)

# 将整数数据转换为 float32（librosa 要求）
data = data.astype(np.float32)

# 步骤 2：使用 librosa 重采样为 44100 Hz
target_sample_rate = 44100
resampled_data = librosa.resample(data, orig_sr=original_sample_rate, target_sr=target_sample_rate)

print("new sample rate:", resampled_data)


# 步骤 3：计算新时长
duration = len(resampled_data) / target_sample_rate
print("New duration (s):", duration)

# 步骤 4：绘制波形图
time = np.linspace(0, duration, num=len(resampled_data))

plt.figure(figsize=(12, 4))
plt.plot(time, resampled_data, color='royalblue')
plt.title("Waveform (Resampled to 44100 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
