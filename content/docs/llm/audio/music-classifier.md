---
title: "音乐流派分类"
weight: 4
---

# 音乐流派分类实战

本节通过一个完整的音乐流派分类项目，实践音频分类的全流程：数据准备、模型微调、评估优化。

## 项目概述

**任务目标：** 将音乐片段分类到不同流派（如摇滚、古典、爵士等）

```
输入: 音频片段 (30 秒)
    ↓
[预训练音频模型]
    ↓
输出: 流派标签 (rock, jazz, classical, ...)
```

## 数据集准备

### GTZAN 数据集

GTZAN 是音乐流派分类的经典基准数据集：

| 属性 | 值 |
|------|-----|
| 总样本数 | 1,000 |
| 流派数 | 10 |
| 每流派样本 | 100 |
| 音频时长 | 30 秒 |
| 采样率 | 22,050 Hz |

**流派类别：**
- blues, classical, country, disco, hiphop
- jazz, metal, pop, reggae, rock

### 加载数据集

```python
from datasets import load_dataset, Audio

# 从 Hugging Face Hub 加载
dataset = load_dataset("marsyas/gtzan", "all")

print(dataset)
# DatasetDict({
#     train: Dataset({features: ['file', 'audio', 'genre'], num_rows: 999})
# })

# 查看样本
example = dataset["train"][0]
print(f"流派: {example['genre']}")
print(f"采样率: {example['audio']['sampling_rate']}")
print(f"时长: {len(example['audio']['array']) / example['audio']['sampling_rate']:.1f}s")
```

### 数据集划分

```python
# 划分训练集、验证集、测试集
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="genre")

# 从测试集中再划分验证集
test_valid = dataset["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="genre")

dataset = {
    "train": dataset["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
}

print(f"训练集: {len(dataset['train'])} 样本")
print(f"验证集: {len(dataset['validation'])} 样本")
print(f"测试集: {len(dataset['test'])} 样本")
```

### 标签映射

```python
# 获取标签列表
labels = dataset["train"].features["genre"].names
print(f"标签: {labels}")
# ['blues', 'classical', 'country', 'disco', 'hiphop',
#  'jazz', 'metal', 'pop', 'reggae', 'rock']

# 创建标签映射
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

num_labels = len(labels)
print(f"类别数: {num_labels}")
```

## 模型选择

### 可选模型

| 模型 | 参数量 | 特点 |
|------|--------|------|
| `facebook/wav2vec2-base` | 95M | 通用语音编码器 |
| `MIT/ast-finetuned-audioset` | 87M | 音频分类专用 |
| `openai/whisper-small` | 244M | 多任务模型 |

### 加载预训练模型

```python
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)

model_name = "facebook/wav2vec2-base"

# 加载特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# 加载模型并配置分类头
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

print(f"模型参数量: {model.num_parameters() / 1e6:.1f}M")
```

## 数据预处理

### 统一采样率

```python
# 目标采样率（与模型匹配）
target_sampling_rate = feature_extractor.sampling_rate
print(f"目标采样率: {target_sampling_rate} Hz")

# 重采样数据集
from datasets import Audio

dataset["train"] = dataset["train"].cast_column(
    "audio",
    Audio(sampling_rate=target_sampling_rate)
)
dataset["validation"] = dataset["validation"].cast_column(
    "audio",
    Audio(sampling_rate=target_sampling_rate)
)
dataset["test"] = dataset["test"].cast_column(
    "audio",
    Audio(sampling_rate=target_sampling_rate)
)
```

### 定义预处理函数

```python
max_duration = 30.0  # 最大音频时长（秒）

def preprocess_function(examples):
    # 提取音频数组
    audio_arrays = [x["array"] for x in examples["audio"]]

    # 特征提取
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=target_sampling_rate,
        max_length=int(target_sampling_rate * max_duration),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    return inputs

# 应用预处理
encoded_dataset = {}
for split in ["train", "validation", "test"]:
    encoded_dataset[split] = dataset[split].map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=["audio", "file"]
    )
```

### 数据增强（可选）

```python
import numpy as np
import torch

def augment_audio(audio, sr):
    """音频数据增强"""
    augmentations = []

    # 1. 添加噪声
    noise = np.random.randn(len(audio)) * 0.005
    augmentations.append(audio + noise)

    # 2. 时间偏移
    shift = int(sr * np.random.uniform(-0.1, 0.1))
    augmentations.append(np.roll(audio, shift))

    # 3. 音量变化
    gain = np.random.uniform(0.8, 1.2)
    augmentations.append(audio * gain)

    return augmentations

# 在训练时应用增强
def train_preprocess_with_augment(examples):
    audio_arrays = []
    labels = []

    for audio, label in zip(examples["audio"], examples["genre"]):
        # 原始音频
        audio_arrays.append(audio["array"])
        labels.append(label)

        # 增强音频（训练时）
        for aug_audio in augment_audio(audio["array"], audio["sampling_rate"]):
            audio_arrays.append(aug_audio)
            labels.append(label)

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=target_sampling_rate,
        max_length=int(target_sampling_rate * max_duration),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    inputs["labels"] = labels

    return inputs
```

## 训练配置

### 评估指标

```python
import evaluate
import numpy as np

# 加载准确率指标
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

### 训练参数

```python
training_args = TrainingArguments(
    output_dir="./gtzan-classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    fp16=True,  # 混合精度训练
)
```

### 创建 Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
```

## 模型训练

### 开始训练

```python
# 训练模型
trainer.train()

# 查看训练历史
for log in trainer.state.log_history:
    if "eval_accuracy" in log:
        print(f"Epoch {log['epoch']}: Accuracy = {log['eval_accuracy']:.4f}")
```

### 训练监控

```
Epoch 1: loss=2.31, accuracy=0.35
Epoch 2: loss=1.45, accuracy=0.52
Epoch 3: loss=0.89, accuracy=0.68
...
Epoch 10: loss=0.23, accuracy=0.85
```

### 保存模型

```python
# 保存最佳模型
trainer.save_model("./gtzan-classifier-best")
feature_extractor.save_pretrained("./gtzan-classifier-best")

# 推送到 Hub（可选）
# trainer.push_to_hub()
```

## 模型评估

### 测试集评估

```python
# 在测试集上评估
test_results = trainer.evaluate(encoded_dataset["test"])
print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")
```

### 混淆矩阵

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 获取预测结果
predictions = trainer.predict(encoded_dataset["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# 绘制混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45, cmap="Blues")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
```

### 分类报告

```python
from sklearn.metrics import classification_report

report = classification_report(
    true_labels,
    pred_labels,
    target_names=labels,
    digits=3
)
print(report)
```

**示例输出：**
```
              precision    recall  f1-score   support

       blues      0.850     0.850     0.850        20
   classical      0.950     1.000     0.974        20
     country      0.800     0.750     0.774        20
       disco      0.750     0.800     0.774        20
      hiphop      0.900     0.850     0.874        20
        jazz      0.850     0.900     0.874        20
       metal      0.950     0.950     0.950        20
         pop      0.800     0.800     0.800        20
      reggae      0.750     0.750     0.750        20
        rock      0.700     0.700     0.700        20

    accuracy                          0.835       200
   macro avg      0.830     0.835     0.832       200
```

## 推理使用

### 加载训练好的模型

```python
from transformers import pipeline

# 创建分类 pipeline
classifier = pipeline(
    "audio-classification",
    model="./gtzan-classifier-best",
    device=0  # GPU
)
```

### 预测单个音频

```python
# 预测新音频
result = classifier("new_song.wav")
print(result)
# [{'label': 'rock', 'score': 0.85}, {'label': 'metal', 'score': 0.10}, ...]

# 只取最高分
predicted_genre = result[0]["label"]
confidence = result[0]["score"]
print(f"预测流派: {predicted_genre} (置信度: {confidence:.2%})")
```

### 批量预测

```python
# 批量处理
audio_files = ["song1.wav", "song2.wav", "song3.wav"]
results = classifier(audio_files)

for file, result in zip(audio_files, results):
    print(f"{file}: {result[0]['label']} ({result[0]['score']:.2%})")
```

### 预测音频数组

```python
import librosa

# 从文件加载音频
audio, sr = librosa.load("song.wav", sr=16000)

# 直接预测音频数组
result = classifier({"array": audio, "sampling_rate": sr})
```

## 优化技巧

### 1. 冻结编码器

数据量小时，只训练分类头：

```python
# 冻结预训练层
for param in model.wav2vec2.parameters():
    param.requires_grad = False

# 只有分类头可训练
for param in model.classifier.parameters():
    param.requires_grad = True

# 计算可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable_params / 1e6:.2f}M")
```

### 2. 学习率调度

```python
from transformers import get_scheduler

# 使用 Cosine 退火
training_args = TrainingArguments(
    ...
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)
```

### 3. 梯度累积

内存不足时使用：

```python
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效批大小 = 4 * 4 = 16
)
```

### 4. 早停

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # 3 个 epoch 无提升则停止
            early_stopping_threshold=0.01
        )
    ]
)
```

### 5. 使用更大模型

```python
# 尝试更强的预训练模型
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True  # 分类头大小不匹配时忽略
)
```

## 完整训练脚本

```python
"""音乐流派分类完整训练脚本"""

from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import numpy as np

# 1. 加载数据集
dataset = load_dataset("marsyas/gtzan", "all")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="genre")

# 2. 配置
model_name = "facebook/wav2vec2-base"
labels = dataset["train"].features["genre"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# 3. 加载模型
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label
)

# 4. 预处理
target_sr = feature_extractor.sampling_rate

def preprocess(examples):
    audio = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio,
        sampling_rate=target_sr,
        max_length=target_sr * 30,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    return inputs

for split in dataset:
    dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=target_sr))
    dataset[split] = dataset[split].map(preprocess, batched=True, batch_size=8, remove_columns=["audio", "file"])

# 5. 训练
accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=pred.label_ids)

training_args = TrainingArguments(
    output_dir="./gtzan-classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model("./gtzan-classifier-best")
```

## 小结

| 阶段 | 关键点 |
|------|--------|
| 数据准备 | 划分训练/验证/测试集，统一采样率 |
| 模型选择 | Wav2Vec2 或 AST 都适合音频分类 |
| 训练技巧 | 冻结编码器、数据增强、早停 |
| 评估 | 混淆矩阵分析易混淆类别 |

**典型性能（GTZAN）：**
- Wav2Vec2-base: ~80-85% 准确率
- AST: ~85-90% 准确率

---

*下一节：[语音识别 (ASR)](asr.md) - 语音转文字实战*
