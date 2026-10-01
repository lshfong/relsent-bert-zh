# relsent-bert-zh-large · 知更鸟

> 人际关系情感分析模型 — 对互动文本进行关系类型、情感极性、关系动态三维分类

[![Model](https://img.shields.io/badge/Model-ONNX-blue)](./model.onnx)
[![Base](https://img.shields.io/badge/Base-chinese--roberta--wwm--ext--large-orange)](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](./LICENSE)

`relsent-bert-zh-large` = **Rel**ation + **Sent**iment + BERT + Chinese + **large**  |  中文代号「知更鸟」🐦

> English: [README.md](./README.md)

---

## 📋 模型能力

输入一段人际互动文本（聊天、评论、对话），输出三个维度的分类结果：

| 维度 | 类别数 | 标签 |
|:---|:---:|:---|
| **关系类型** | 7 | love(爱情) / family(亲情) / friendship(友情) / colleague(同事) / teacher_student(师生) / stranger(陌生人) / other(其他) |
| **情感极性** | 3 | positive(正面) / neutral(中性) / negative(负面) |
| **关系动态** | 3 | intimacy(亲近) / stable(稳定) / distance(疏远) |

---

## 🎯 应用场景

除直接分类外，模型的文本表征向量（CLS / mean pooling）可迁移至以下 NLP 下游场景：

| 场景 | 说明 | 关键能力 |
|:---|:---|:---|
| **中文知识图谱的关系三元组抽取** | 从非结构化文本中抽取 (主体, 关系类型, 客体) 三元组，辅助构建中文社交知识图谱 | 关系分类 |
| **关系句语义相似度计算与匹配** | 基于句向量计算文本间语义相似度，用于关系描述句的检索与匹配 | 句向量表征 |
| **知识库问答中的关系链接任务** | 将用户问句中的关系描述映射到知识库预定义关系类型，实现关系消歧与链接 | 关系分类 + 语义匹配 |
| **面向关系检索的RAG系统向量召回** | 以关系维度构建向量索引，在 RAG 流程中实现关系粒度的向量召回，提升检索相关性 | 句向量表征 + 关系分类 |

> **使用方式**：通过 ONNX 模型提取 logits 做关系/情感分类，或提取 hidden states 做句向量表征（CLS token 或 mean pooling），用于上述场景的特征输入。

---

## 📊 模型性能

| 指标 | 测试集 |
|:---|:---:|
| 关系类型 F1 (macro) | 0.799 |
| 情感极性 F1 (macro) | 0.894 |
| 关系动态 F1 (macro) | 0.777 |
| **平均 F1** | **0.823** |

- 基座模型: `chinese-roberta-wwm-ext-large` (330M 参数)
- 训练数据: LLM 合成 + 边界样本 + 人工核验
- 训练策略: 三任务联合分类 + 复合句边界样本增强

---

## 📦 文件说明

```
release/
├── model.onnx          # ONNX 模型 (~1.2 GB)
├── tokenizer/          # Tokenizer 文件
│   ├── vocab.txt           # 词表 (BERT 中文)
│   ├── tokenizer.json      # Fast tokenizer
│   └── tokenizer_config.json
├── LICENSE             # MIT 协议
├── README.md           # 英文说明
└── README_zh.md        # 中文说明（本文件）
```

---

## 🔌 ONNX 模型接口

| | 名称 | 形状 | 类型 |
|:---|:---|:---|:---|
| **输入** | `input_ids` | [batch, seq_len] | int64 |
| | `attention_mask` | [batch, seq_len] | int64 |
| **输出** | `logits` | [batch, 13] | float32 |

输出 logits 拆分方式：

```
relation   = logits[:, 0:7]    # 7 类关系类型
sentiment  = logits[:, 7:10]   # 3 类情感极性
dynamic    = logits[:, 10:13]  # 3 类关系动态
```

取 `argmax` 得到预测类别。

---

## 🚀 使用示例

### Python (ONNX Runtime)

```bash
pip install onnxruntime tokenizers numpy
```

```python
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

session = ort.InferenceSession("model.onnx")
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

REL = ["love", "family", "friendship", "colleague", "teacher_student", "stranger", "other"]
SENT = ["positive", "neutral", "negative"]
DYN = ["intimacy", "stable", "distance"]

def predict(text: str, max_length: int = 256):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_length]
    mask = [1] * len(ids)
    ids += [0] * (max_length - len(ids))
    mask += [0] * (max_length - len(mask))

    logits = session.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.array([mask], dtype=np.int64),
    })[0]

    return {
        "relation": REL[logits[0, :7].argmax()],
        "sentiment": SENT[logits[0, 7:10].argmax()],
        "dynamic": DYN[logits[0, 10:13].argmax()],
    }

print(predict("下班了一起去喝酒，你叫上老张"))
# {'relation': 'friendship', 'sentiment': 'neutral', 'dynamic': 'stable'}
```

### JavaScript (Node.js)

```bash
npm install onnxruntime-node
```

```javascript
const ort = require("onnxruntime-node");

const REL = ["love","family","friendship","colleague","teacher_student","stranger","other"];
const SENT = ["positive","neutral","negative"];
const DYN = ["intimacy","stable","distance"];

async function predict(text, tokenizer, session) {
  const enc = tokenizer.encode(text, { maxLength: 256 });
  const ids = Array(256).fill(0);
  const mask = Array(256).fill(0);
  enc.ids.forEach((id, i) => { if (i < 256) { ids[i] = id; mask[i] = 1; } });

  const feed = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, 256]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(mask.map(BigInt)), [1, 256]),
  };
  const out = await session.run(feed);
  const logits = out.logits.data;

  return {
    relation: REL[argmax(logits.slice(0, 7))],
    sentiment: SENT[argmax(logits.slice(7, 10))],
    dynamic: DYN[argmax(logits.slice(10, 13))],
  };
}

function argmax(arr) { return arr.indexOf(Math.max(...arr)); }
```

### 其他语言

任何支持 ONNX Runtime 的语言都可调用：C#, Java, Go, Rust, Swift, Kotlin 等。

---

## 📐 最佳实践

| 建议 | 说明 |
|:---|:---|
| **最大长度** | 256 tokens，覆盖绝大部分中文互动文本 |
| **输入语言** | 中文为主 |
| **文本类型** | 对话、聊天、评论等互动文本 |
| **置信度** | 可对 logits 做 softmax 获取概率分布 |
| **批处理** | 支持 batch 推理，速度更快 |
| **复合句支持** | 可处理 2-4 句的长文本，支持情感转折识别 |

---

## ⚠️ 免责声明

- 本模型仅用于文本层面的关系/情感状态识别
- **不是**心理诊断工具，**不是**人格评估工具
- 不做婚恋匹配预测
- 输出结果仅供参考，不构成任何建议
- 训练数据为 LLM 合成数据+人工核验数据，可能存在偏差

---

## 📄 许可

MIT License — Copyright (c) 2026 58136364@qq.com