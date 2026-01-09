---
title: "Transformers Pipelines"
weight: 2
---

# 音频应用与 Transformers Pipelines

Hugging Face 的 `pipeline` API 提供了开箱即用的音频处理能力，让你无需深入了解模型细节即可快速完成音频任务。

## Pipeline 概述

### 什么是 Pipeline？

Pipeline 是对模型推理流程的高级封装，自动处理：
- 数据预处理（采样率转换、特征提取）
- 模型推理
- 后处理（解码输出）

```python
from transformers import pipeline

# 一行代码创建音频分类器
classifier = pipeline("audio-classification")
```

### 音频相关的 Pipeline 任务

| 任务名称 | 描述 | 典型模型 |
|----------|------|----------|
| `audio-classification` | 音频/语音分类 | Wav2Vec2, HuBERT |
| `automatic-speech-recognition` | 语音转文字 (ASR) | Whisper, Wav2Vec2 |
| `text-to-speech` | 文字转语音 (TTS) | SpeechT5, Bark |
| `zero-shot-audio-classification` | 零样本音频分类 | CLAP |

## 音频分类

音频分类用于识别音频的类别，如：
- 语音情感识别
- 音乐流派分类
- 环境声音识别
- 说话人识别

### 基础用法

```python
from transformers import pipeline

# 创建分类器（自动下载默认模型）
classifier = pipeline("audio-classification")

# 对音频文件分类
result = classifier("audio_sample.wav")
print(result)
# [{'label': 'speech', 'score': 0.95}, {'label': 'music', 'score': 0.03}, ...]
```

### 指定模型

```python
# 使用特定模型进行情感识别
classifier = pipeline(
    "audio-classification",
    model="superb/hubert-base-superb-er"  # 情感识别模型
)

result = classifier("speech.wav")
# [{'label': 'hap', 'score': 0.8}, {'label': 'sad', 'score': 0.1}, ...]
```

### 处理 Dataset 数据

```python
from datasets import load_dataset, Audio

# 加载数据集
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train[:10]")

# 创建分类器
classifier = pipeline(
    "audio-classification",
    model="facebook/wav2vec2-base"
)

# 直接传入音频数组
example = dataset[0]
result = classifier(example["audio"]["array"])
```

### 批量处理

```python
# 批量处理音频文件
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = classifier(audio_files)

# 或者处理数据集
def classify_batch(batch):
    audio_arrays = [x["array"] for x in batch["audio"]]
    predictions = classifier(audio_arrays)
    batch["predictions"] = predictions
    return batch

classified_dataset = dataset.map(classify_batch, batched=True, batch_size=4)
```

### 常用音频分类模型

| 模型 | 任务 | 描述 |
|------|------|------|
| `MIT/ast-finetuned-audioset-10-10-0.4593` | 通用音频分类 | AudioSet 预训练 |
| `superb/hubert-base-superb-er` | 情感识别 | 识别语音情感 |
| `superb/wav2vec2-base-superb-sid` | 说话人识别 | 识别说话人身份 |
| `facebook/wav2vec2-base` | 语音分类 | 通用语音模型 |

## 语音识别 (ASR)

自动语音识别将语音转换为文字。

### 基础用法

```python
from transformers import pipeline

# 创建 ASR pipeline（默认使用 Whisper）
transcriber = pipeline("automatic-speech-recognition")

# 转录音频
result = transcriber("speech.wav")
print(result["text"])
# "Hello, how are you today?"
```

### 使用 Whisper 模型

```python
# 使用 Whisper 大模型
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3"
)

result = transcriber("speech.wav")
```

### 指定语言和任务

```python
# 中文语音识别
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    generate_kwargs={"language": "chinese", "task": "transcribe"}
)

result = transcriber("chinese_speech.wav")
```

### 翻译模式

Whisper 支持直接将非英语语音翻译为英文：

```python
# 语音翻译（任意语言 → 英语）
translator = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    generate_kwargs={"task": "translate"}
)

# 中文语音直接输出英文
result = translator("chinese_speech.wav")
print(result["text"])  # English translation
```

### 返回时间戳

```python
# 获取词级时间戳
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    return_timestamps="word"  # 或 "chunk" 获取句子级
)

result = transcriber("speech.wav")
print(result)
# {
#   'text': 'Hello world',
#   'chunks': [
#     {'text': 'Hello', 'timestamp': (0.0, 0.5)},
#     {'text': 'world', 'timestamp': (0.5, 1.0)}
#   ]
# }
```

### 处理长音频

```python
# 使用分块处理长音频
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s=30,      # 每块 30 秒
    stride_length_s=5       # 重叠 5 秒
)

# 处理长音频文件
result = transcriber("long_audio.wav")
```

### 常用 ASR 模型

| 模型 | 语言 | 特点 |
|------|------|------|
| `openai/whisper-tiny` | 多语言 | 最小最快 |
| `openai/whisper-base` | 多语言 | 平衡选择 |
| `openai/whisper-small` | 多语言 | 性能较好 |
| `openai/whisper-medium` | 多语言 | 高精度 |
| `openai/whisper-large-v3` | 多语言 | 最高精度 |
| `facebook/wav2vec2-large-960h` | 英语 | 英语专用 |
| `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | 中文 | 中文专用 |

### Whisper 模型大小对比

| 模型 | 参数量 | 相对速度 | VRAM 需求 |
|------|--------|----------|-----------|
| tiny | 39M | ~32x | ~1 GB |
| base | 74M | ~16x | ~1 GB |
| small | 244M | ~6x | ~2 GB |
| medium | 769M | ~2x | ~5 GB |
| large-v3 | 1550M | 1x | ~10 GB |

## 零样本音频分类

使用 CLAP 模型进行零样本分类，无需预定义类别。

### 基础用法

```python
from transformers import pipeline

# 创建零样本分类器
classifier = pipeline(
    "zero-shot-audio-classification",
    model="laion/clap-htsat-unfused"
)

# 自定义候选标签
candidate_labels = ["music", "speech", "noise", "silence"]

result = classifier(
    "audio.wav",
    candidate_labels=candidate_labels
)

print(result)
# [{'label': 'music', 'score': 0.85}, {'label': 'speech', 'score': 0.10}, ...]
```

### 细粒度分类

```python
# 音乐流派分类
candidate_labels = [
    "classical music",
    "rock music",
    "jazz music",
    "electronic music",
    "hip hop music"
]

result = classifier("song.wav", candidate_labels=candidate_labels)
```

### CLAP 模型原理

CLAP (Contrastive Language-Audio Pretraining) 类似于图像领域的 CLIP：
- 将音频和文本映射到同一个嵌入空间
- 通过对比学习训练
- 支持零样本分类

```
Audio → Audio Encoder → Audio Embedding ─┐
                                         ├→ Similarity Score
Text  → Text Encoder  → Text Embedding ──┘
```

## 文本转语音 (TTS)

将文字转换为语音。

### 基础用法

```python
from transformers import pipeline
import soundfile as sf

# 创建 TTS pipeline
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# 需要说话人嵌入
from datasets import load_dataset
embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation"
)
speaker_embedding = embeddings_dataset[0]["xvector"]

# 生成语音
speech = synthesizer(
    "Hello, this is a test of text to speech.",
    forward_params={"speaker_embeddings": speaker_embedding}
)

# 保存音频
sf.write("output.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

### 使用 Bark 模型

Bark 是更强大的 TTS 模型，支持更自然的语音合成：

```python
from transformers import pipeline

# 创建 Bark pipeline
synthesizer = pipeline("text-to-speech", model="suno/bark-small")

# Bark 不需要说话人嵌入
speech = synthesizer("Hello! This is Bark speaking.")

# 保存音频
import soundfile as sf
sf.write("bark_output.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

### Bark 高级功能

```python
# Bark 支持特殊标记控制语音效果
text_with_effects = """
[laughs] That's so funny! [sighs]
Can you believe it? [clears throat]
Let me tell you something important.
"""

speech = synthesizer(text_with_effects)
```

**Bark 支持的特殊标记：**
- `[laughs]` - 笑声
- `[sighs]` - 叹气
- `[clears throat]` - 清嗓子
- `[gasps]` - 喘气
- `♪` - 音乐/唱歌

### TTS 模型对比

| 模型 | 特点 | 速度 | 质量 |
|------|------|------|------|
| `microsoft/speecht5_tts` | 需要说话人嵌入 | 快 | 中等 |
| `suno/bark-small` | 支持情感和效果 | 慢 | 高 |
| `suno/bark` | 完整版 Bark | 很慢 | 很高 |

## Pipeline 进阶配置

### 指定设备

```python
# 使用 GPU
classifier = pipeline(
    "audio-classification",
    model="facebook/wav2vec2-base",
    device=0  # 使用第一个 GPU
)

# 使用 MPS (Apple Silicon)
classifier = pipeline(
    "audio-classification",
    model="facebook/wav2vec2-base",
    device="mps"
)

# 自动选择设备
classifier = pipeline(
    "audio-classification",
    model="facebook/wav2vec2-base",
    device_map="auto"
)
```

### 批处理优化

```python
# 设置批处理大小
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    batch_size=8  # 批处理大小
)

# 批量处理
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = transcriber(audio_files)
```

### 使用本地模型

```python
# 从本地路径加载模型
classifier = pipeline(
    "audio-classification",
    model="./my_local_model",
    tokenizer="./my_local_model"
)
```

## 完整示例：语音转文字应用

```python
from transformers import pipeline
from datasets import load_dataset, Audio
import soundfile as sf

# 1. 初始化 ASR pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps=True
)

# 2. 准备音频数据
def load_audio(file_path, target_sr=16000):
    """加载并重采样音频"""
    import librosa
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

# 3. 转录函数
def transcribe_audio(audio_input):
    """
    转录音频，支持文件路径或音频数组
    """
    result = asr(audio_input)
    return {
        "text": result["text"],
        "chunks": result.get("chunks", [])
    }

# 4. 批量处理示例
def batch_transcribe(audio_files):
    """批量转录多个音频文件"""
    results = []
    for file in audio_files:
        try:
            result = transcribe_audio(file)
            results.append({
                "file": file,
                "success": True,
                **result
            })
        except Exception as e:
            results.append({
                "file": file,
                "success": False,
                "error": str(e)
            })
    return results

# 5. 使用示例
if __name__ == "__main__":
    # 单文件转录
    result = transcribe_audio("speech.wav")
    print(f"转录结果: {result['text']}")

    # 带时间戳的输出
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        print(f"[{start:.2f}s - {end:.2f}s]: {chunk['text']}")
```

## 小结

| Pipeline 任务 | 用途 | 推荐模型 |
|--------------|------|----------|
| `audio-classification` | 音频/语音分类 | `MIT/ast-finetuned-audioset` |
| `automatic-speech-recognition` | 语音转文字 | `openai/whisper-small` |
| `zero-shot-audio-classification` | 零样本分类 | `laion/clap-htsat-unfused` |
| `text-to-speech` | 文字转语音 | `suno/bark-small` |

**使用 Pipeline 的优势：**
- 简单易用，一行代码即可运行
- 自动处理预处理和后处理
- 支持多种输入格式（文件路径、音频数组）
- 内置批处理和设备管理

---

*下一节：[音频 Transformer 架构](architectures.md) - 深入理解模型原理*
