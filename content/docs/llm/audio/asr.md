---
title: "语音识别 (ASR)"
weight: 5
---

# 语音识别 (ASR)

自动语音识别 (Automatic Speech Recognition, ASR) 是将语音信号转换为文本的技术。本节深入讲解 ASR 的原理、Whisper 模型使用和微调方法。

## ASR 基础

### 什么是 ASR？

```
语音信号 → [ASR 系统] → 文本
"你好世界" →            → "你好世界"
```

ASR 系统需要解决：
1. **声学建模**：学习声音到音素的映射
2. **语言建模**：学习词序列的概率分布
3. **对齐问题**：音频和文本长度不一致

### 传统 ASR vs 端到端 ASR

| 方法 | 架构 | 特点 |
|------|------|------|
| **传统（GMM-HMM）** | 声学模型 + 语言模型 + 解码器 | 模块化，需大量专家知识 |
| **端到端** | 单一神经网络 | 简化流程，数据驱动 |

**现代端到端方法：**
- CTC (Connectionist Temporal Classification)
- Seq2Seq with Attention
- Transducer (RNN-T)

### 主流 ASR 模型

| 模型 | 架构 | 语言支持 | 特点 |
|------|------|----------|------|
| Whisper | Seq2Seq | 99+ 语言 | 多任务、鲁棒性强 |
| Wav2Vec2 | CTC | 需微调 | 自监督预训练 |
| Conformer | CTC/Transducer | 需微调 | 结合 CNN 和 Transformer |
| HuBERT | CTC | 需微调 | 聚类预训练 |

## Whisper 模型

### 模型概述

Whisper 是 OpenAI 发布的多任务语音模型：

```
                      Whisper
┌─────────────────────────────────────────────┐
│                                             │
│  支持任务:                                   │
│  • 语音识别 (transcribe)                     │
│  • 语音翻译 (translate → English)            │
│  • 语言识别 (language detection)             │
│  • 时间戳生成 (timestamp)                    │
│                                             │
│  支持语言: 99+ 种                            │
│  训练数据: 680,000 小时                      │
│                                             │
└─────────────────────────────────────────────┘
```

### 模型大小

| 模型 | 参数量 | 英语 WER | 多语言 WER | VRAM |
|------|--------|----------|------------|------|
| tiny | 39M | 7.6% | - | ~1GB |
| base | 74M | 5.0% | - | ~1GB |
| small | 244M | 3.4% | 6.1% | ~2GB |
| medium | 769M | 2.9% | 4.4% | ~5GB |
| large-v3 | 1550M | 2.5% | 3.0% | ~10GB |

### 基础使用

```python
from transformers import pipeline

# 创建 ASR pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small"
)

# 转录音频
result = asr("audio.wav")
print(result["text"])
```

### 指定语言

```python
# 指定语言（提高准确率）
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={"language": "chinese"}
)

result = asr("chinese_audio.wav")
```

### 语音翻译

```python
# 将任意语言翻译为英语
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={"task": "translate"}
)

# 中文语音 → 英文文本
result = asr("chinese_audio.wav")
print(result["text"])  # English output
```

### 时间戳

```python
# 获取词级时间戳
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps="word"
)

result = asr("audio.wav")
for chunk in result["chunks"]:
    start, end = chunk["timestamp"]
    print(f"[{start:.2f}s - {end:.2f}s] {chunk['text']}")
```

### 长音频处理

```python
# 分块处理长音频
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s=30,      # 每块 30 秒
    stride_length_s=(4, 2)  # 左重叠 4s，右重叠 2s
)

# 处理长音频（自动分块）
result = asr("long_audio.wav")
```

### 底层 API 使用

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# 加载模型和处理器
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 加载音频
audio, sr = librosa.load("audio.wav", sr=16000)

# 准备输入
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# 生成
generated_ids = model.generate(
    inputs["input_features"],
    language="zh",
    task="transcribe",
    max_length=448
)

# 解码
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(transcription[0])
```

## 评估指标

### WER (Word Error Rate)

词错误率是 ASR 最常用的评估指标：

```
WER = (S + D + I) / N × 100%

S = 替换词数 (Substitutions)
D = 删除词数 (Deletions)
I = 插入词数 (Insertions)
N = 参考文本总词数
```

**示例：**
```
参考: The cat sat on the mat
预测: The cat set in the hat

S=2 (sat→set, on→in), D=0, I=0
WER = 2/6 = 33.3%
```

### CER (Character Error Rate)

字符错误率，适用于中文等无明显词边界的语言：

```
CER = (S + D + I) / N × 100%
（基于字符计算）
```

### 使用 evaluate 库计算

```python
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 计算 WER
references = ["the cat sat on the mat"]
predictions = ["the cat set in the hat"]

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"WER: {wer:.2%}")  # WER: 33.33%

# 计算 CER（中文）
references_zh = ["今天天气很好"]
predictions_zh = ["今天天气很号"]

cer = cer_metric.compute(predictions=predictions_zh, references=references_zh)
print(f"CER: {cer:.2%}")
```

### 归一化处理

计算 WER 前通常需要归一化文本：

```python
import re
import string

def normalize_text(text):
    """文本归一化"""
    # 转小写
    text = text.lower()
    # 移除标点
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 合并空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 归一化后再计算 WER
ref_norm = normalize_text(reference)
pred_norm = normalize_text(prediction)
wer = wer_metric.compute(predictions=[pred_norm], references=[ref_norm])
```

## 模型微调

### 为什么需要微调？

- 适应特定领域（医疗、法律等）
- 适应特定口音或方言
- 提高低资源语言性能
- 降低特定场景错误率

### 准备数据集

```python
from datasets import load_dataset, Audio

# 加载数据集（以 Common Voice 为例）
dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "zh-CN",
    split="train[:1000]"  # 先用小子集测试
)

# 只保留需要的列
dataset = dataset.remove_columns([
    "accent", "age", "client_id", "down_votes",
    "gender", "locale", "segment", "up_votes"
])

print(dataset[0])
# {'audio': {...}, 'sentence': '...'}
```

### 数据预处理

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 统一采样率
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # 处理音频
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # 处理文本标签
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# 应用预处理
dataset = dataset.map(
    prepare_dataset,
    remove_columns=["audio", "sentence"],
    num_proc=4
)
```

### 数据收集器

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 处理输入特征
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 处理标签
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 用 -100 替换 padding token（忽略损失计算）
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 移除开头的 decoder_start_token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

### 配置训练

```python
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 加载模型
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# 配置生成参数
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)
```

### 评估函数

```python
import evaluate

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 替换 -100
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 解码
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 计算 WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

### 开始训练

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# 训练
trainer.train()

# 保存
trainer.save_model("./whisper-finetuned-final")
processor.save_pretrained("./whisper-finetuned-final")
```

### 使用 LoRA 微调

使用 PEFT 进行参数高效微调：

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 配置 LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# 应用 LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# trainable params: 3,932,160 || all params: 244,341,760 || trainable%: 1.61%
```

## 推理优化

### 使用 Flash Attention

```python
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)
```

### 批处理推理

```python
from transformers import pipeline

# 创建带批处理的 pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    batch_size=8,
    device=0
)

# 批量处理
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = asr(audio_files)
```

### 量化加速

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 使用 faster-whisper

更快的 Whisper 推理实现：

```python
from faster_whisper import WhisperModel

# 加载模型
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# 转录
segments, info = model.transcribe("audio.wav", language="zh")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## 实战示例：中文语音识别

```python
"""中文语音识别完整示例"""

from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate

# 1. 加载模型
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={"language": "chinese", "task": "transcribe"}
)

# 2. 加载测试数据
test_data = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "zh-CN",
    split="test[:100]"
)
test_data = test_data.cast_column("audio", Audio(sampling_rate=16000))

# 3. 批量转录
predictions = []
references = []

for item in test_data:
    # 转录
    result = asr(item["audio"]["array"])
    predictions.append(result["text"])
    references.append(item["sentence"])

# 4. 评估
cer_metric = evaluate.load("cer")
cer = cer_metric.compute(predictions=predictions, references=references)
print(f"CER: {cer:.2%}")

# 5. 查看示例
for i in range(5):
    print(f"参考: {references[i]}")
    print(f"预测: {predictions[i]}")
    print()
```

## 小结

| 主题 | 要点 |
|------|------|
| 模型选择 | Whisper 多语言首选，Wav2Vec2 需微调 |
| 评估指标 | 英语用 WER，中文用 CER |
| 微调方法 | 全量微调或 LoRA |
| 推理优化 | Flash Attention、量化、faster-whisper |

**Whisper 使用建议：**
1. 指定语言可提高准确率
2. 长音频使用分块处理
3. 大模型效果更好但更慢
4. 领域数据微调可显著提升效果

---

*下一节：[语音合成 (TTS)](tts.md) - 文字转语音技术*
