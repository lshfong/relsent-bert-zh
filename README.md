# relsent-bert-zh-large · Robin

> Relationship and Sentiment Analysis for Chinese Text — Multi-dimensional classification of interpersonal interactions

[![Model](https://img.shields.io/badge/Model-ONNX-blue)](./model.onnx)
[![Base](https://img.shields.io/badge/Base-chinese--roberta--wwm--ext--large-orange)](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](./LICENSE)

`relsent-bert-zh-large` = **Rel**ation + **Sent**iment + BERT + zh + **large**  |  Codename: 「Robin」🐦

> Also available: [中文说明](./README_zh.md)

---

## What It Does

Given a piece of Chinese interpersonal text (chat message, comment, dialogue), the model predicts three dimensions:

| Dimension | Classes | Labels |
|:---|:---:|:---|
| **Relation Type** | 7 | love / family / friendship / colleague / teacher_student / stranger / other |
| **Sentiment** | 3 | positive / neutral / negative |
| **Relation Dynamic** | 3 | intimacy / stable / distance |

---

## Use Cases

Beyond direct classification, the model's text representations (CLS / mean pooling) can be transferred to:

| Scenario | Description | Key Capability |
|:---|:---|:---|
| **Relation Triple Extraction for Chinese KGs** | Extract (subject, relation, object) triples from unstructured text to build Chinese social knowledge graphs | Relation Classification |
| **Relational Sentence Similarity & Matching** | Compute semantic similarity via sentence embeddings for relational text retrieval and matching | Sentence Embedding |
| **Relation Linking in KBQA** | Map natural language relation descriptions in user queries to predefined relation types in knowledge bases | Classification + Matching |
| **Vector Recall for Relation-aware RAG** | Build vector indexes along the relation dimension for fine-grained retrieval in RAG pipelines | Embedding + Classification |

> **How to use**: Extract logits for classification, or extract hidden states (CLS token / mean pooling) as sentence embeddings for the scenarios above.

---

## Performance

| Metric | Test Set |
|:---|:---:|
| Relation F1 (macro) | 0.799 |
| Sentiment F1 (macro) | 0.894 |
| Dynamic F1 (macro) | 0.777 |
| **Average F1** | **0.823** |

- Base model: `chinese-roberta-wwm-ext-large` (330M params)
- Training data: synthetic + boundary-enhanced samples with human review
- Training strategy: Multi-task joint classification + compound sentence boundary augmentation

---

## Files

```
release/
├── model.onnx          # ONNX model (~1.2 GB)
├── tokenizer/          # Tokenizer files
│   ├── vocab.txt
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── LICENSE             # MIT License
├── README.md           # English (this file)
└── README_zh.md        # 中文说明
```

---

## ONNX Interface

| | Name | Shape | Dtype |
|:---|:---|:---|:---|
| **Input** | `input_ids` | [batch, seq_len] | int64 |
| | `attention_mask` | [batch, seq_len] | int64 |
| **Output** | `logits` | [batch, 13] | float32 |

Output logits breakdown:

```
relation   = logits[:, 0:7]    # 7 relation types
sentiment  = logits[:, 7:10]   # 3 sentiment polarities
dynamic    = logits[:, 10:13]  # 3 relation dynamics
```

Use `argmax` to get the predicted class.

---

## Quick Start

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

print(predict("Let's grab a drink after work, bring Lao Zhang along"))
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

### Other Languages

Any language with ONNX Runtime bindings works: C#, Java, Go, Rust, Swift, Kotlin, etc.

---

## Best Practices

| Tip | Detail |
|:---|:---|
| **Max Length** | 256 tokens, covers most Chinese interpersonal texts |
| **Input Language** | Primarily Chinese |
| **Text Type** | Dialogues, chat messages, comments |
| **Confidence** | Apply softmax to logits for probability distributions |
| **Batching** | Batch inference supported for better throughput |
| **Compound Sentences** | Handles 2-4 sentence texts with emotional transitions |

---

## Disclaimer

- This model is for text-level relationship and sentiment recognition only
- **NOT** a psychological diagnostic tool
- **NOT** a personality assessment tool
- **NOT** intended for matchmaking predictions
- Results are for reference only and do not constitute professional advice
- Training data is LLM-synthesized with human review; bias may exist

---

## License

MIT License — Copyright (c) 2026 58136364@qq.com