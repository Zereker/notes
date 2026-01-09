---
title: "音频 Transformer 架构"
weight: 3
---

# 音频 Transformer 架构

音频领域的 Transformer 模型主要分为两大类架构：**CTC（仅编码器）** 和 **Seq2Seq（编码器-解码器）**。理解这两种架构的原理和区别，是选择和使用音频模型的基础。

## 音频任务概述

在深入架构之前，先了解音频处理的核心挑战：

```
输入: 音频波形 (变长序列，每秒 16,000 个采样点)
     ↓
输出: 文本/标签 (长度与输入不对齐)
```

**核心挑战：输入输出长度不一致**
- 一秒音频 = 16,000 个采样点
- 对应文本可能只有几个字
- 需要某种对齐机制

## CTC 架构

### 什么是 CTC？

CTC (Connectionist Temporal Classification) 是一种**仅使用编码器**的架构，通过特殊的损失函数解决对齐问题。

```
音频波形
    ↓
[特征提取] → Log-mel Spectrogram
    ↓
[音频编码器] → 隐藏状态序列
    ↓
[线性层] → 字符概率分布
    ↓
[CTC 解码] → 最终文本
```

### CTC 的工作原理

CTC 引入了一个特殊的**空白标记 (blank token)**，允许模型在不确定时输出空白：

```
原始输出:  h h h _ e e _ l l l l _ l _ o o o
           ↓ ↓ ↓   ↓ ↓   ↓ ↓ ↓ ↓   ↓   ↓ ↓ ↓
合并重复:  h       e     l         l   o
           ↓
移除空白:  h e l l o → "hello"
```

**CTC 解码规则：**
1. 合并连续重复的字符
2. 移除空白标记 `_`

### CTC 代码示例

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# 加载 CTC 模型
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# 准备输入
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")

# 推理
with torch.no_grad():
    logits = model(**inputs).logits

# CTC 解码
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])
print(transcription)
```

### CTC 的优缺点

| 优点 | 缺点 |
|------|------|
| 推理速度快（无自回归） | 不能建模输出间依赖 |
| 结构简单 | 对重复字符处理不佳 |
| 支持流式处理 | 需要字符级词表 |

### 典型 CTC 模型

| 模型 | 描述 |
|------|------|
| Wav2Vec2 | Facebook 的自监督预训练模型 |
| HuBERT | 隐藏单元聚类预训练 |
| SEW | 压缩预训练方法 |

## Seq2Seq 架构

### 什么是 Seq2Seq？

Seq2Seq (Sequence-to-Sequence) 使用**编码器-解码器**结构，通过注意力机制实现输入输出的对齐。

```
音频波形
    ↓
[特征提取] → Log-mel Spectrogram
    ↓
[音频编码器] → 编码器隐藏状态
    ↓              ↓
    ↓         [交叉注意力]
    ↓              ↓
[文本解码器] ← 自回归生成
    ↓
输出文本 (逐 token 生成)
```

### Seq2Seq 的工作原理

```
编码器输出:  [h₁, h₂, h₃, ..., hₙ]  (音频特征序列)
                    ↓
            [交叉注意力机制]
                    ↓
解码器:     <start> → "The" → "cat" → "sat" → <end>
                         ↓      ↓      ↓
            每步都关注编码器的相关位置
```

**自回归生成过程：**
1. 输入起始标记 `<start>`
2. 预测第一个 token
3. 将预测结果作为下一步输入
4. 重复直到生成结束标记 `<end>`

### Whisper 模型详解

Whisper 是目前最流行的 Seq2Seq 语音模型：

```
                    Whisper 架构
┌─────────────────────────────────────────────┐
│                                             │
│   Audio Input                               │
│       ↓                                     │
│   [Log-Mel Spectrogram]                     │
│       ↓                                     │
│   ┌───────────────┐                         │
│   │ Audio Encoder │  (Transformer Encoder)  │
│   │   × N layers  │                         │
│   └───────┬───────┘                         │
│           ↓                                 │
│   Encoder Hidden States                     │
│           ↓                                 │
│   ┌───────────────┐     ┌─────────────┐    │
│   │ Cross-Attention│ ← │Text Decoder │    │
│   └───────────────┘     │  × N layers │    │
│                         └──────┬──────┘    │
│                                ↓           │
│                         Output Tokens       │
│                                             │
└─────────────────────────────────────────────┘
```

### Whisper 代码示例

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 加载 Whisper 模型
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 准备输入
inputs = processor(
    audio_array,
    sampling_rate=16000,
    return_tensors="pt"
)

# 生成（自回归解码）
generated_ids = model.generate(
    inputs["input_features"],
    language="en",
    task="transcribe"
)

# 解码输出
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(transcription[0])
```

### Whisper 的特殊 Token

Whisper 使用特殊 token 控制行为：

```python
# Whisper 的特殊 token 结构
<|startoftranscript|>     # 转录开始
<|en|>                     # 语言标记 (英语)
<|transcribe|>             # 任务 (转录 vs 翻译)
<|notimestamps|>           # 是否输出时间戳
...actual text...          # 转录文本
<|endoftext|>              # 结束标记
```

```python
# 强制特定语言和任务
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="chinese",
    task="transcribe"
)

generated_ids = model.generate(
    inputs["input_features"],
    forced_decoder_ids=forced_decoder_ids
)
```

### Seq2Seq 的优缺点

| 优点 | 缺点 |
|------|------|
| 可建模输出依赖关系 | 推理速度慢（自回归） |
| 支持任意输出词表 | 不支持真正的流式 |
| 可执行多任务（翻译等） | 更大的模型复杂度 |

### 典型 Seq2Seq 模型

| 模型 | 描述 |
|------|------|
| Whisper | OpenAI 多语言/多任务模型 |
| SpeechT5 | 统一语音-文本模型 |
| mBART | 多语言序列到序列 |

## 音频编码器模型

无论 CTC 还是 Seq2Seq，都需要强大的音频编码器。以下是主流的预训练编码器：

### Wav2Vec 2.0

```
原始波形
    ↓
[CNN 特征提取器] → 局部特征 (每 20ms 一帧)
    ↓
[Transformer 编码器] → 上下文表示
    ↓
输出: 每帧的隐藏状态
```

**预训练方法：对比学习**
- 随机遮挡部分输入
- 模型预测被遮挡位置的量化特征
- 类似于 BERT 的 MLM

```python
from transformers import Wav2Vec2Model

# 仅使用编码器（无 CTC 头）
encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
outputs = encoder(**inputs)

# 获取隐藏状态
hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
```

### HuBERT

HuBERT 使用**聚类伪标签**进行预训练：

```
第一轮: 使用 MFCC 聚类生成伪标签
    ↓
训练模型预测伪标签
    ↓
第二轮: 使用模型特征重新聚类
    ↓
再次训练...
```

```python
from transformers import HubertModel

encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
```

### Audio Spectrogram Transformer (AST)

AST 将音频频谱图视为"图像"，使用 Vision Transformer：

```
Log-mel Spectrogram (time × freq)
    ↓
[分割为 patches]
    ↓
[线性嵌入 + 位置编码]
    ↓
[Vision Transformer]
    ↓
分类输出
```

```python
from transformers import ASTModel

# AST 用于音频分类
ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
```

### 编码器对比

| 模型 | 输入 | 预训练方法 | 特点 |
|------|------|------------|------|
| Wav2Vec2 | 原始波形 | 对比学习 | 端到端，支持微调 |
| HuBERT | 原始波形 | 聚类预测 | 更稳定的表示 |
| AST | 频谱图 | ImageNet 迁移 | 利用视觉预训练 |
| Whisper Encoder | 频谱图 | 监督学习 | 大规模标注数据 |

## 架构对比

### CTC vs Seq2Seq

| 特性 | CTC | Seq2Seq |
|------|-----|---------|
| **结构** | 仅编码器 | 编码器 + 解码器 |
| **解码方式** | 非自回归 | 自回归 |
| **输出建模** | 条件独立 | 依赖历史输出 |
| **推理速度** | 快 | 慢 |
| **流式支持** | 原生支持 | 需特殊处理 |
| **错误纠正** | 弱 | 强（语言模型） |
| **代表模型** | Wav2Vec2 | Whisper |

### 何时选择哪种架构？

**选择 CTC：**
- 需要低延迟实时识别
- 流式语音识别
- 计算资源有限
- 语言模型可外部融合

**选择 Seq2Seq：**
- 追求最高准确率
- 多语言/多任务场景
- 需要语音翻译
- 可接受较高延迟

### 混合架构

一些模型结合了两种方法的优点：

```
音频
 ↓
[共享编码器]
 ↓
┌────────────────┬────────────────┐
│   CTC 分支     │  Attention 分支 │
│  (辅助训练)    │   (主要输出)    │
└────────────────┴────────────────┘
         ↓
   联合解码
```

**联合 CTC-Attention 解码：**
```python
# 伪代码
final_score = α × ctc_score + (1-α) × attention_score
```

## 特征提取对比

### 原始波形 vs 频谱图

| 输入类型 | 模型示例 | 优点 | 缺点 |
|----------|----------|------|------|
| 原始波形 | Wav2Vec2 | 端到端学习 | 序列很长 |
| Log-mel | Whisper | 降维，传统特征 | 信息损失 |
| 学习特征 | Encodec | 神经压缩 | 需要预训练 |

### 序列长度对比

```
1 秒音频的序列长度:

原始波形 (16kHz):     16,000 个点
Wav2Vec2 输出:        ~50 帧 (每帧 20ms)
Whisper 编码器输出:   ~50 帧 (每帧 20ms)
```

## 实践建议

### 模型选择指南

```
任务需求
    │
    ├─→ 实时/流式 → Wav2Vec2 + CTC
    │
    ├─→ 高准确率 → Whisper
    │
    ├─→ 多语言 → Whisper / XLS-R
    │
    ├─→ 语音翻译 → Whisper (translate task)
    │
    └─→ 音频分类 → AST / Wav2Vec2
```

### 微调建议

| 场景 | 推荐方案 |
|------|----------|
| 数据少 (<1h) | 冻结编码器，只训练分类头 |
| 数据中等 (1-100h) | 微调全模型，小学习率 |
| 数据充足 (>100h) | 可从头训练或深度微调 |
| 领域特殊 | 先在领域数据上继续预训练 |

## 小结

| 架构 | 核心思想 | 代表模型 | 适用场景 |
|------|----------|----------|----------|
| CTC | 空白标记对齐 | Wav2Vec2 | 实时识别 |
| Seq2Seq | 注意力对齐 | Whisper | 高精度识别 |
| 混合 | 联合训练 | Conformer | 平衡方案 |

**关键要点：**
1. CTC 快但不能建模输出依赖
2. Seq2Seq 准但推理慢
3. 预训练编码器是基础
4. 根据实际需求选择架构

---

*下一节：[音乐流派分类](music-classifier.md) - 动手实践音频分类*
