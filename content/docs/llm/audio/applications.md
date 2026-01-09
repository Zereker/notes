---
title: "音频应用"
weight: 7
---

# 音频应用实战

本节介绍如何将音频模型集成到实际应用中，包括架构设计、性能优化和部署实践。

## 典型应用场景

### 语音助手

```
用户语音 → [ASR] → 文本 → [LLM] → 回复文本 → [TTS] → 语音回复

示例: "今天天气怎么样？"
      ↓ ASR
      "今天天气怎么样？"
      ↓ LLM
      "今天北京晴天，气温 25 度。"
      ↓ TTS
      🔊 语音播报
```

### 语音转写服务

```
会议录音 → [VAD] → 分段音频 → [ASR] → 文本 → [格式化] → 会议纪要

功能:
• 说话人分离
• 时间戳标注
• 关键词提取
• 摘要生成
```

### 有声书/播客生成

```
文本内容 → [分段] → [TTS] → 音频片段 → [合并] → 最终音频

功能:
• 多说话人
• 情感控制
• 背景音乐
```

## 应用架构设计

### 简单架构（单机）

```
┌─────────────────────────────────────────────────┐
│                   应用服务                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   ASR    │  │   LLM    │  │   TTS    │      │
│  │  Model   │  │  Model   │  │  Model   │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│                     ↑                           │
│               GPU (共享显存)                     │
└─────────────────────────────────────────────────┘
```

### 微服务架构

```
┌─────────────┐     ┌─────────────┐
│   Client    │────→│   Gateway   │
└─────────────┘     └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ ASR Service │   │ LLM Service │   │ TTS Service │
│  (GPU #1)   │   │  (GPU #2)   │   │  (GPU #3)   │
└─────────────┘   └─────────────┘   └─────────────┘
         ↓                 ↓                 ↓
┌───────────────────────────────────────────────────┐
│                   Message Queue                    │
└───────────────────────────────────────────────────┘
```

### 流式处理架构

```python
"""流式语音处理架构"""

import asyncio
from transformers import pipeline
import sounddevice as sd
import numpy as np

class StreamingAudioProcessor:
    def __init__(self):
        self.asr = pipeline("automatic-speech-recognition",
                           model="openai/whisper-small",
                           chunk_length_s=5)
        self.tts = pipeline("text-to-speech",
                           model="suno/bark-small")

    async def process_stream(self, audio_stream):
        """处理音频流"""
        buffer = []

        async for chunk in audio_stream:
            buffer.append(chunk)

            # 累积足够数据后处理
            if len(buffer) >= 5:  # 5 秒
                audio = np.concatenate(buffer)
                buffer = []

                # ASR
                text = self.asr(audio)["text"]
                yield {"type": "transcription", "text": text}

    async def synthesize_stream(self, text_stream):
        """流式 TTS"""
        async for text in text_stream:
            # 分句处理
            sentences = self._split_sentences(text)
            for sentence in sentences:
                audio = self.tts(sentence)
                yield audio["audio"]

    def _split_sentences(self, text):
        import re
        return re.split(r'(?<=[.!?。！？])\s+', text)
```

## 模型集成

### 语音助手完整实现

```python
"""完整语音助手实现"""

from transformers import pipeline
import soundfile as sf
import numpy as np

class VoiceAssistant:
    def __init__(self,
                 asr_model="openai/whisper-small",
                 tts_model="microsoft/speecht5_tts"):
        """初始化语音助手"""
        # ASR
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            device=0
        )

        # TTS
        self.tts = pipeline(
            "text-to-speech",
            model=tts_model,
            device=0
        )

        # 说话人嵌入 (SpeechT5 需要)
        if "speecht5" in tts_model:
            from datasets import load_dataset
            embeddings = load_dataset(
                "Matthijs/cmu-arctic-xvectors",
                split="validation"
            )
            self.speaker_emb = embeddings[7306]["xvector"]
        else:
            self.speaker_emb = None

    def listen(self, audio_path):
        """语音转文本"""
        result = self.asr(audio_path)
        return result["text"]

    def speak(self, text, output_path="response.wav"):
        """文本转语音"""
        if self.speaker_emb:
            speech = self.tts(
                text,
                forward_params={"speaker_embeddings": self.speaker_emb}
            )
        else:
            speech = self.tts(text)

        sf.write(output_path, speech["audio"],
                samplerate=speech["sampling_rate"])
        return output_path

    def process(self, audio_path, llm_callback):
        """完整处理流程"""
        # 1. 语音转文本
        user_text = self.listen(audio_path)
        print(f"用户: {user_text}")

        # 2. LLM 生成回复
        response_text = llm_callback(user_text)
        print(f"助手: {response_text}")

        # 3. 文本转语音
        audio_path = self.speak(response_text)
        return audio_path

# 使用示例
if __name__ == "__main__":
    assistant = VoiceAssistant()

    # 模拟 LLM 回调
    def simple_llm(text):
        if "天气" in text:
            return "今天天气晴朗，适合出门。"
        return "抱歉，我不太理解你的问题。"

    # 处理
    result = assistant.process("user_audio.wav", simple_llm)
    print(f"回复音频: {result}")
```

### 会议转写系统

```python
"""会议转写系统"""

from transformers import pipeline
from datasets import Audio
import numpy as np

class MeetingTranscriber:
    def __init__(self):
        # ASR with timestamps
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            return_timestamps=True,
            chunk_length_s=30,
            device=0
        )

    def transcribe(self, audio_path):
        """转写会议音频"""
        result = self.asr(audio_path)
        return self._format_transcript(result)

    def _format_transcript(self, result):
        """格式化转写结果"""
        transcript = []

        for chunk in result.get("chunks", []):
            start, end = chunk["timestamp"]
            text = chunk["text"]
            transcript.append({
                "start": start,
                "end": end,
                "text": text.strip()
            })

        return transcript

    def export_srt(self, transcript, output_path):
        """导出 SRT 字幕"""
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with open(output_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(transcript, 1):
                f.write(f"{i}\n")
                f.write(f"{format_time(item['start'])} --> {format_time(item['end'])}\n")
                f.write(f"{item['text']}\n\n")

    def export_txt(self, transcript, output_path):
        """导出纯文本"""
        with open(output_path, "w", encoding="utf-8") as f:
            for item in transcript:
                f.write(f"[{item['start']:.1f}s] {item['text']}\n")

# 使用示例
transcriber = MeetingTranscriber()
transcript = transcriber.transcribe("meeting.wav")
transcriber.export_srt(transcript, "meeting.srt")
```

### 有声书生成器

```python
"""有声书生成器"""

from transformers import pipeline
import soundfile as sf
import numpy as np
import re

class AudiobookGenerator:
    def __init__(self, model="suno/bark-small"):
        self.tts = pipeline("text-to-speech", model=model, device=0)

    def generate(self, text, output_path, max_chunk=200):
        """生成有声书"""
        # 分段
        chunks = self._split_text(text, max_chunk)

        # 逐段合成
        audio_parts = []
        for i, chunk in enumerate(chunks):
            print(f"合成进度: {i+1}/{len(chunks)}")
            speech = self.tts(chunk)
            audio_parts.append(speech["audio"])

            # 添加短暂停顿
            pause = np.zeros(int(speech["sampling_rate"] * 0.5))
            audio_parts.append(pause)

        # 合并
        final_audio = np.concatenate(audio_parts)
        sf.write(output_path, final_audio, samplerate=speech["sampling_rate"])
        return output_path

    def _split_text(self, text, max_length):
        """智能分割文本"""
        # 按段落分割
        paragraphs = text.split("\n\n")
        chunks = []

        for para in paragraphs:
            if len(para) <= max_length:
                chunks.append(para)
            else:
                # 按句子分割
                sentences = re.split(r'(?<=[.!?。！？])\s*', para)
                current = ""
                for s in sentences:
                    if len(current) + len(s) <= max_length:
                        current += s + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = s + " "
                if current:
                    chunks.append(current.strip())

        return chunks

# 使用示例
generator = AudiobookGenerator()
book_text = """
第一章

从前有一座山，山上有一座庙。庙里有个老和尚和一个小和尚。

有一天，老和尚对小和尚说："我给你讲个故事吧。"

小和尚高兴地说："好啊好啊！"
"""
generator.generate(book_text, "audiobook.wav")
```

## 性能优化

### GPU 显存优化

```python
import torch
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig

# 1. 使用半精度
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16
)

# 2. 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    quantization_config=bnb_config
)

# 3. 梯度检查点（训练时）
model.gradient_checkpointing_enable()
```

### 推理加速

```python
# 1. 使用 Flash Attention
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    attn_implementation="flash_attention_2"
)

# 2. 使用 BetterTransformer
model = model.to_bettertransformer()

# 3. 批处理
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    batch_size=8
)
results = asr(["audio1.wav", "audio2.wav", "audio3.wav"])
```

### 并发处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncAudioProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.asr = pipeline("automatic-speech-recognition",
                           model="openai/whisper-small")

    async def process_batch(self, audio_files):
        """并发处理多个音频文件"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.asr, f)
            for f in audio_files
        ]
        results = await asyncio.gather(*tasks)
        return results

# 使用
async def main():
    processor = AsyncAudioProcessor()
    files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    results = await processor.process_batch(files)

asyncio.run(main())
```

## 部署实践

### FastAPI 服务

```python
"""音频处理 API 服务"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from transformers import pipeline
import soundfile as sf
import tempfile
import os

app = FastAPI()

# 初始化模型
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
tts = pipeline("text-to-speech", model="suno/bark-small")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """语音转文本"""
    # 保存上传的文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 转录
        result = asr(tmp_path)
        return {"text": result["text"]}
    finally:
        os.unlink(tmp_path)

@app.post("/synthesize")
async def synthesize(text: str):
    """文本转语音"""
    # 合成语音
    speech = tts(text)

    # 保存并返回
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="output.wav"
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 运行: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 预下载模型
RUN python -c "from transformers import pipeline; \
    pipeline('automatic-speech-recognition', model='openai/whisper-small'); \
    pipeline('text-to-speech', model='suno/bark-small')"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  audio-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 监控与日志

```python
"""添加监控和日志"""

import time
import logging
from functools import wraps
from prometheus_client import Counter, Histogram

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus 指标
REQUEST_COUNT = Counter('audio_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('audio_request_latency_seconds', 'Request latency', ['endpoint'])

def monitor(endpoint):
    """监控装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(endpoint=endpoint).inc()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in {endpoint}: {e}")
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
                logger.info(f"{endpoint} completed in {latency:.2f}s")

        return wrapper
    return decorator

# 使用
@app.post("/transcribe")
@monitor("transcribe")
async def transcribe(file: UploadFile = File(...)):
    # ...
```

## 小结

| 场景 | 关键技术 | 注意事项 |
|------|----------|----------|
| 语音助手 | ASR + LLM + TTS | 端到端延迟优化 |
| 会议转写 | Whisper + 时间戳 | 长音频分块处理 |
| 有声书 | TTS + 文本分割 | 语音连贯性 |

**部署建议：**
1. 使用 GPU 加速推理
2. 模型量化减少显存
3. 批处理提高吞吐
4. 流式处理降低延迟
5. 异步 API 提高并发

---

*下一节：[TTS 模型对比](tts-models.md) - 主流模型评测*
