---
title: "语音合成 (TTS)"
weight: 6
---

# 语音合成 (TTS)

语音合成 (Text-to-Speech, TTS) 是将文本转换为自然语音的技术。本节介绍 TTS 的基础原理和主流模型的使用方法。

## TTS 基础原理

### 什么是 TTS？

```
文本输入 → [TTS 系统] → 语音波形
"Hello, world!" →        → 🔊 音频
```

TTS 系统需要解决：
1. **文本分析**：分词、韵律预测、发音转换
2. **声学建模**：生成声学特征（梅尔频谱）
3. **声码器**：将声学特征转换为波形

### TTS 系统架构

```
        传统 TTS 流程
┌─────────────────────────────────────────┐
│                                         │
│  文本 → [文本前端] → 音素序列            │
│              ↓                          │
│         [声学模型] → 梅尔频谱            │
│              ↓                          │
│         [声码器] → 音频波形              │
│                                         │
└─────────────────────────────────────────┘
```

**现代端到端 TTS：**
- 单一模型完成全部步骤
- 代表：Bark, VITS, Tacotron

### 主流 TTS 模型

| 模型 | 类型 | 特点 |
|------|------|------|
| SpeechT5 | Seq2Seq | 需要说话人嵌入 |
| Bark | GPT-style | 支持情感、效果、多语言 |
| VITS | End-to-End | 高质量、快速 |
| Tacotron 2 | Seq2Seq | 经典架构 |
| FastSpeech 2 | Non-AR | 并行生成，速度快 |

## SpeechT5 模型

### 模型概述

SpeechT5 是微软发布的统一语音-文本模型：

```
SpeechT5 架构
┌─────────────────────────────────────────┐
│                                         │
│  共享编码器-解码器 Transformer           │
│                                         │
│  支持任务:                               │
│  • TTS (文本→语音)                       │
│  • ASR (语音→文本)                       │
│  • 语音转换                              │
│  • 语音增强                              │
│                                         │
└─────────────────────────────────────────┘
```

### 基础使用

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# 加载模型
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 准备文本
text = "Hello, this is a test of the text to speech system."
inputs = processor(text=text, return_tensors="pt")

# 加载说话人嵌入（必需）
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# 生成语音
speech = model.generate_speech(
    inputs["input_ids"],
    speaker_embeddings,
    vocoder=vocoder
)

# 保存音频
sf.write("output.wav", speech.numpy(), samplerate=16000)
```

### 使用 Pipeline

```python
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

# 创建 TTS pipeline
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# 获取说话人嵌入
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = embeddings_dataset[7306]["xvector"]

# 合成语音
speech = synthesizer(
    "Hello, how are you today?",
    forward_params={"speaker_embeddings": speaker_embedding}
)

# 保存
sf.write("output.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

### 不同说话人

```python
# CMU Arctic 数据集包含多个说话人
# 选择不同索引获取不同声音

speaker_indices = {
    "male_1": 0,
    "male_2": 100,
    "female_1": 7306,
    "female_2": 7500,
}

# 使用不同说话人
for name, idx in speaker_indices.items():
    speaker_emb = torch.tensor(embeddings_dataset[idx]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=vocoder)
    sf.write(f"output_{name}.wav", speech.numpy(), samplerate=16000)
```

### SpeechT5 局限性

- 仅支持英语
- 需要提供说话人嵌入
- 某些音素可能发音不准确
- 不支持情感控制

## Bark 模型

### 模型概述

Bark 是 Suno AI 发布的生成式 TTS 模型：

```
Bark 特点
┌─────────────────────────────────────────┐
│                                         │
│  • GPT-style 自回归生成                  │
│  • 支持 13+ 语言                         │
│  • 支持非语言声音（笑声、叹气等）         │
│  • 支持背景音乐和环境音                   │
│  • 无需说话人嵌入                         │
│                                         │
└─────────────────────────────────────────┘
```

### 基础使用

```python
from transformers import AutoProcessor, BarkModel
import scipy

# 加载模型
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# 准备文本
text = "Hello, my name is Suno. And I am an AI voice generator."
inputs = processor(text, return_tensors="pt")

# 生成语音
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# 保存
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_output.wav", rate=sample_rate, data=audio_array)
```

### 使用 Pipeline

```python
from transformers import pipeline
import soundfile as sf

# 创建 TTS pipeline
synthesizer = pipeline("text-to-speech", model="suno/bark-small")

# 合成语音（无需说话人嵌入）
speech = synthesizer("Hello! This is Bark speaking naturally.")

# 保存
sf.write("bark_output.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

### 多语言支持

```python
# Bark 支持多语言
texts = {
    "en": "Hello, how are you today?",
    "zh": "你好，今天怎么样？",
    "ja": "こんにちは、元気ですか？",
    "de": "Hallo, wie geht es dir heute?",
    "fr": "Bonjour, comment allez-vous aujourd'hui?",
}

for lang, text in texts.items():
    inputs = processor(text, return_tensors="pt")
    audio = model.generate(**inputs)
    audio_np = audio.cpu().numpy().squeeze()
    scipy.io.wavfile.write(f"bark_{lang}.wav", rate=sample_rate, data=audio_np)
```

### 声音预设

```python
# 使用声音预设控制说话人
voice_presets = [
    "v2/en_speaker_0",  # 英语男声 0
    "v2/en_speaker_1",  # 英语男声 1
    "v2/en_speaker_6",  # 英语女声
    "v2/zh_speaker_0",  # 中文说话人
]

for preset in voice_presets:
    inputs = processor(
        text,
        voice_preset=preset,
        return_tensors="pt"
    )
    audio = model.generate(**inputs)
    # 保存...
```

### 非语言声音

Bark 的独特功能是支持非语言声音：

```python
# 支持的特殊标记
text_with_effects = """
[laughs] Ha ha, that's so funny!
[sighs] Well, what can I do...
[clears throat] Anyway, let me continue.
[gasps] Oh my goodness!
"""

inputs = processor(text_with_effects, return_tensors="pt")
audio = model.generate(**inputs)
```

**支持的特殊标记：**

| 标记 | 效果 |
|------|------|
| `[laughs]` | 笑声 |
| `[sighs]` | 叹气 |
| `[clears throat]` | 清嗓子 |
| `[gasps]` | 倒吸气 |
| `[music]` | 背景音乐 |
| `♪` | 唱歌 |
| `...` | 停顿/犹豫 |

### 唱歌模式

```python
# Bark 可以生成歌唱
song_text = """
♪ Twinkle, twinkle, little star,
How I wonder what you are! ♪
"""

inputs = processor(song_text, return_tensors="pt")
audio = model.generate(**inputs)
```

### Bark 模型大小

| 模型 | 参数量 | VRAM | 特点 |
|------|--------|------|------|
| `suno/bark-small` | 300M | ~4GB | 快速，质量中等 |
| `suno/bark` | 1.5B | ~8GB | 高质量，较慢 |

### 优化推理

```python
import torch

# 使用 GPU
model = model.to("cuda")

# 半精度
model = model.half()

# 开启优化
model = model.to_bettertransformer()

# 生成
inputs = processor(text, return_tensors="pt").to("cuda")
audio = model.generate(**inputs)
```

## 其他 TTS 模型

### VITS (Coqui TTS)

```python
from TTS.api import TTS

# 加载 VITS 模型
tts = TTS("tts_models/en/ljspeech/vits")

# 合成语音
tts.tts_to_file(
    text="Hello world!",
    file_path="vits_output.wav"
)
```

### Edge TTS (微软在线)

```python
import edge_tts
import asyncio

async def synthesize():
    communicate = edge_tts.Communicate(
        "Hello, this is Edge TTS speaking.",
        "en-US-JennyNeural"  # 声音选择
    )
    await communicate.save("edge_output.mp3")

asyncio.run(synthesize())
```

**Edge TTS 中文声音：**
- `zh-CN-XiaoxiaoNeural` - 晓晓（女）
- `zh-CN-YunxiNeural` - 云希（男）
- `zh-CN-YunyangNeural` - 云扬（男）

## 语音合成实践

### 完整 TTS 示例

```python
"""文本转语音完整示例"""

from transformers import pipeline
import soundfile as sf
import os

class TTSEngine:
    def __init__(self, model_name="suno/bark-small"):
        """初始化 TTS 引擎"""
        self.synthesizer = pipeline(
            "text-to-speech",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

    def synthesize(self, text, output_path="output.wav"):
        """合成语音"""
        speech = self.synthesizer(text)
        sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])
        return output_path

    def synthesize_long_text(self, text, output_path="output.wav", max_length=200):
        """处理长文本"""
        # 分句
        sentences = self._split_sentences(text, max_length)

        # 逐句合成
        audio_segments = []
        for sentence in sentences:
            speech = self.synthesizer(sentence)
            audio_segments.append(speech["audio"])

        # 合并音频
        import numpy as np
        combined = np.concatenate(audio_segments)
        sf.write(output_path, combined, samplerate=speech["sampling_rate"])
        return output_path

    def _split_sentences(self, text, max_length):
        """分割长文本"""
        import re
        # 按标点分割
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        result = []
        current = ""

        for s in sentences:
            if len(current) + len(s) <= max_length:
                current += " " + s if current else s
            else:
                if current:
                    result.append(current.strip())
                current = s

        if current:
            result.append(current.strip())

        return result

# 使用示例
if __name__ == "__main__":
    tts = TTSEngine()

    # 短文本
    tts.synthesize("Hello, how are you?", "short.wav")

    # 长文本
    long_text = """
    Welcome to the text to speech demonstration.
    This system can convert any text into natural sounding speech.
    It supports multiple languages and voice styles.
    """
    tts.synthesize_long_text(long_text, "long.wav")
```

### 批量生成

```python
from transformers import pipeline
import soundfile as sf
from tqdm import tqdm

# 初始化
tts = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# 加载说话人嵌入
from datasets import load_dataset
embeddings = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_emb = embeddings[7306]["xvector"]

# 批量文本
texts = [
    "Welcome to our service.",
    "Please hold while we connect you.",
    "Thank you for your patience.",
    "Have a great day!",
]

# 批量生成
for i, text in enumerate(tqdm(texts)):
    speech = tts(text, forward_params={"speaker_embeddings": speaker_emb})
    sf.write(f"audio_{i}.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

### 流式合成

```python
import torch
from transformers import BarkModel, AutoProcessor

# 加载模型
model = BarkModel.from_pretrained("suno/bark-small")
processor = AutoProcessor.from_pretrained("suno/bark-small")

def stream_tts(text, chunk_size=100):
    """流式文本转语音"""
    # 分块处理文本
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # 逐块生成
    for chunk in chunks:
        inputs = processor(chunk, return_tensors="pt")
        audio = model.generate(**inputs)
        yield audio.cpu().numpy().squeeze()

# 使用流式生成
for audio_chunk in stream_tts("This is a long text that will be processed in chunks."):
    # 实时播放或处理
    pass
```

## 评估 TTS 质量

### 主观评估 (MOS)

MOS (Mean Opinion Score) 是 TTS 质量的主要评估指标：

| 分数 | 质量 |
|------|------|
| 5 | 优秀 |
| 4 | 良好 |
| 3 | 一般 |
| 2 | 较差 |
| 1 | 很差 |

### 客观指标

```python
# 梅尔倒谱失真 (MCD)
def calculate_mcd(ref_mel, syn_mel):
    """计算梅尔倒谱失真"""
    import numpy as np
    diff = ref_mel - syn_mel
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=1)))
    return mcd

# F0 相关性
def calculate_f0_correlation(ref_audio, syn_audio, sr):
    """计算基频相关性"""
    import librosa
    import numpy as np

    f0_ref, _, _ = librosa.pyin(ref_audio, fmin=50, fmax=500, sr=sr)
    f0_syn, _, _ = librosa.pyin(syn_audio, fmin=50, fmax=500, sr=sr)

    # 计算相关系数
    mask = ~np.isnan(f0_ref) & ~np.isnan(f0_syn)
    corr = np.corrcoef(f0_ref[mask], f0_syn[mask])[0, 1]
    return corr
```

## 小结

| 模型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| SpeechT5 | 速度快，质量稳定 | 仅英语，需说话人嵌入 | 英语朗读 |
| Bark | 多语言，支持情感 | 速度慢，不够稳定 | 创意内容 |
| VITS | 高质量，快速 | 需要额外安装 | 生产环境 |
| Edge TTS | 免费在线，质量高 | 需要网络 | 快速原型 |

**TTS 使用建议：**
1. 快速原型用 Edge TTS
2. 离线英语用 SpeechT5
3. 多语言或情感用 Bark
4. 生产环境考虑 VITS 或商业 API

---

*下一节：[音频应用](applications.md) - 综合应用实战*
