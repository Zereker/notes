---
title: "音频数据处理"
weight: 1
---

# 音频数据处理基础

音频处理是语音识别（ASR）和语音合成（TTS）的基础。理解音频信号的本质和处理方法，是深入学习音频大模型的第一步。

## 音频信号基础

### 声波的物理特性

声音是一种机械波，通过空气（或其他介质）传播。描述声波的三个基本参数：

| 参数 | 描述 | 对应的听觉感受 |
|------|------|--------------|
| **频率 (Frequency)** | 每秒振动次数，单位 Hz | 音调高低 |
| **振幅 (Amplitude)** | 振动的强度大小 | 音量大小 |
| **相位 (Phase)** | 振动的起始位置 | 通常人耳不敏感 |

**人耳听觉范围：**
- 频率：20 Hz ~ 20,000 Hz
- 语音信号主要集中在 300 Hz ~ 3,400 Hz

### 模拟信号与数字信号

```
模拟信号 (连续)          数字信号 (离散)
    │                        │
    ∿                    ●─●─●─●
   声波                   采样点
```

计算机只能处理数字信号，需要将模拟的声波转换为数字表示。这个过程涉及两个关键步骤：**采样** 和 **量化**。

## 采样与量化

### 采样定理（奈奎斯特定理）

> **采样频率必须至少是信号最高频率的 2 倍**，才能完整还原原始信号。

```
采样率 ≥ 2 × 最高频率
```

**常见采样率及应用场景：**

| 采样率 | 应用场景 |
|--------|----------|
| 8,000 Hz | 电话语音 |
| 16,000 Hz | 语音识别（常用） |
| 22,050 Hz | 低质量音乐 |
| 44,100 Hz | CD 音质 |
| 48,000 Hz | 专业音频/视频 |

### 量化（位深度）

量化是将连续的振幅值映射到有限的离散级别。

```
位深度    量化级别        动态范围
8-bit     256 级         48 dB
16-bit    65,536 级      96 dB (CD 标准)
24-bit    16,777,216 级  144 dB (专业)
```

### 音频数据存储

一段音频的数据量计算：

```
数据量 = 采样率 × 位深度 × 声道数 × 时长

例：1 分钟 CD 音质立体声
= 44100 × 16 × 2 × 60 = 84,672,000 bits ≈ 10 MB
```

### 使用 Python 加载音频

```python
from datasets import load_dataset, Audio

# 加载数据集并设置采样率
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 查看音频数据
example = dataset[0]["audio"]
print(f"采样率: {example['sampling_rate']} Hz")
print(f"音频数组形状: {example['array'].shape}")
print(f"时长: {len(example['array']) / example['sampling_rate']:.2f} 秒")
```

**音频数据结构：**
```python
{
    'path': '/path/to/audio.wav',
    'array': array([0.001, -0.002, ...]),  # 波形数据
    'sampling_rate': 16000                  # 采样率
}
```

## 频谱分析

### 时域 vs 频域

```
时域 (Waveform)              频域 (Spectrum)
amplitude                    magnitude
    │    ∿∿∿∿                    │  ╷
    │  ∿    ∿∿                   │  │ ╷
    └────────────→ time          └──┴─┴──→ frequency

表示：振幅随时间变化           表示：各频率成分的强度
```

### 傅里叶变换 (FFT)

傅里叶变换将时域信号转换为频域表示：

```python
import numpy as np
import matplotlib.pyplot as plt

# 对音频信号进行 FFT
audio_array = example['array']
fft_result = np.fft.fft(audio_array)
frequencies = np.fft.fftfreq(len(audio_array), 1/16000)

# 只取正频率部分
positive_freq_idx = frequencies > 0
plt.plot(frequencies[positive_freq_idx],
         np.abs(fft_result)[positive_freq_idx])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
```

### 短时傅里叶变换 (STFT)

语音信号是**非平稳**的，不同时刻的频率成分不同。STFT 将信号分成短帧，对每帧做 FFT：

```python
import librosa
import librosa.display

# 计算 STFT
stft = librosa.stft(audio_array,
                    n_fft=1024,      # FFT 窗口大小
                    hop_length=512)  # 帧移

# 转为分贝刻度
stft_db = librosa.amplitude_to_db(np.abs(stft))

# 可视化频谱图
librosa.display.specshow(stft_db,
                         sr=16000,
                         hop_length=512,
                         x_axis='time',
                         y_axis='hz')
plt.colorbar(format='%+2.0f dB')
```

**STFT 的关键参数：**

| 参数 | 描述 | 典型值 |
|------|------|--------|
| `n_fft` | FFT 窗口大小 | 512, 1024, 2048 |
| `hop_length` | 帧移（相邻帧的间隔） | n_fft / 4 |
| `win_length` | 窗口长度 | = n_fft |

### 梅尔频谱 (Mel Spectrogram)

人耳对频率的感知是**非线性**的——对低频变化更敏感，对高频变化不敏感。梅尔刻度模拟了这种特性：

```
梅尔刻度: m = 2595 × log₁₀(1 + f/700)
```

```python
# 计算梅尔频谱
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio_array,
    sr=16000,
    n_mels=80,        # 梅尔滤波器数量
    n_fft=1024,
    hop_length=512
)

# 转为对数刻度（log-mel spectrogram）
log_mel = librosa.power_to_db(mel_spectrogram)

librosa.display.specshow(log_mel,
                         sr=16000,
                         x_axis='time',
                         y_axis='mel')
```

**为什么使用梅尔频谱？**
1. 符合人耳听觉特性
2. 降低特征维度（从数百个频点到 80-128 个梅尔频带）
3. 是大多数语音模型的输入格式

### MFCC (梅尔频率倒谱系数)

MFCC 是对 log-mel 频谱做离散余弦变换 (DCT) 的结果：

```python
# 计算 MFCC
mfcc = librosa.feature.mfcc(
    y=audio_array,
    sr=16000,
    n_mfcc=13,     # MFCC 系数数量
    n_mels=80
)

print(f"MFCC 形状: {mfcc.shape}")  # (n_mfcc, time_steps)
```

**MFCC vs Mel Spectrogram：**

| 特征 | 维度 | 优势 | 使用场景 |
|------|------|------|----------|
| MFCC | 13-40 | 维度低，去相关 | 传统语音识别 |
| Mel Spectrogram | 80-128 | 保留更多信息 | 深度学习模型 |

> **现代趋势**：深度学习模型（如 Wav2Vec2、Whisper）更倾向于使用 log-mel spectrogram 甚至直接使用原始波形，让模型自己学习特征。

## 数据预处理

### 使用 Transformers 特征提取器

Hugging Face 的特征提取器会自动处理音频预处理：

```python
from transformers import AutoFeatureExtractor

# 加载预训练模型的特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base"
)

# 查看预期的采样率
print(f"期望采样率: {feature_extractor.sampling_rate}")

# 处理音频
inputs = feature_extractor(
    example['array'],
    sampling_rate=16000,
    return_tensors="pt"
)

print(f"输入形状: {inputs.input_values.shape}")
```

### 重采样

当音频采样率与模型期望不一致时，需要重采样：

```python
from datasets import Audio

# 方法 1：使用 datasets 自动重采样
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 方法 2：使用 librosa 手动重采样
import librosa
audio_resampled = librosa.resample(
    audio_array,
    orig_sr=44100,
    target_sr=16000
)
```

### 填充与截断

处理变长音频输入：

```python
# 批量处理时自动填充
inputs = feature_extractor(
    [audio1, audio2, audio3],
    sampling_rate=16000,
    padding=True,          # 填充到最长
    max_length=16000 * 30, # 最大长度（30秒）
    truncation=True,       # 超长截断
    return_tensors="pt"
)

# 返回 attention_mask 标记有效位置
print(f"attention_mask 形状: {inputs.attention_mask.shape}")
```

### 数据增强

语音数据增强可以提高模型鲁棒性：

```python
import numpy as np

def add_noise(audio, noise_level=0.005):
    """添加高斯噪声"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def time_shift(audio, shift_max=0.1):
    """时间平移"""
    shift = int(len(audio) * np.random.uniform(-shift_max, shift_max))
    return np.roll(audio, shift)

def speed_change(audio, sr, rate_range=(0.9, 1.1)):
    """变速（不改变音调）"""
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)
```

**常用数据增强技术：**

| 技术 | 描述 | 适用场景 |
|------|------|----------|
| 添加噪声 | 加入背景噪声 | 提高抗噪能力 |
| 时间平移 | 整体移动波形 | 增加样本多样性 |
| 变速/变调 | 改变语速或音调 | 说话人变化 |
| SpecAugment | 频谱图遮挡 | 端到端 ASR |

### 完整预处理流程示例

```python
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

# 1. 加载数据集
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

# 2. 加载特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "openai/whisper-small"
)

# 3. 统一采样率
dataset = dataset.cast_column(
    "audio",
    Audio(sampling_rate=feature_extractor.sampling_rate)
)

# 4. 定义预处理函数
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt"
    )

    return inputs

# 5. 批量应用预处理
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=dataset.column_names
)
```

## 小结

| 概念 | 要点 |
|------|------|
| 采样率 | 语音识别常用 16kHz，需与模型匹配 |
| 位深度 | 16-bit 是常见标准 |
| 频谱表示 | Log-mel spectrogram 是主流输入格式 |
| 特征提取器 | 使用 `AutoFeatureExtractor` 自动处理 |
| 预处理 | 重采样 → 特征提取 → 填充/截断 |

---

*下一节：[Transformers Pipelines](pipelines.md) - 快速上手音频任务*
