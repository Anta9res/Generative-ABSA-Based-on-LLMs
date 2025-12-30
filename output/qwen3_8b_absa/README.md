---
library_name: transformers
model_name: qwen3_8b_absa
tags:
- generated_from_trainer
- sft
- unsloth
- trl
licence: license
---

# Model Card for qwen3_8b_absa

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.22.1
- Transformers: 4.56.0
- Pytorch: 2.5.1+cu121
- Datasets: 3.6.0
- Tokenizers: 0.22.0