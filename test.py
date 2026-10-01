import os
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

_DIR = os.path.dirname(os.path.abspath(__file__))

session = ort.InferenceSession(os.path.join(_DIR, "model.onnx"))
tokenizer = Tokenizer.from_file(os.path.join(_DIR, "tokenizer/tokenizer.json"))

REL = [
    "love",
    "family",
    "friendship",
    "colleague",
    "teacher_student",
    "stranger",
    "other",
]
SENT = ["positive", "neutral", "negative"]
DYN = ["intimacy", "stable", "distance"]


def predict(text: str, max_length: int = 256):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_length]
    mask = [1] * len(ids)
    ids += [0] * (max_length - len(ids))
    mask += [0] * (max_length - len(mask))

    logits = session.run(
        None,
        {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
        },
    )[0]

    return {
        "relation": REL[logits[0, :7].argmax()],
        "sentiment": SENT[logits[0, 7:10].argmax()],
        "dynamic": DYN[logits[0, 10:13].argmax()],
    }


print(predict("下班了一起去喝酒，你叫上老张"))
# {'relation': 'friendship', 'sentiment': 'neutral', 'dynamic': 'stable'}
