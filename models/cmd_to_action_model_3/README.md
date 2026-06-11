---
base_model: unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
---

# Model Card for Model ID
This model is a fine-tuned version of `unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit`. It has been trained to convert natural language commands into structured JSON actions.

## Model Details

### Model Description
- **Developed by:** Puru Thakkar
- **Model type:** LoRA adapter fine-tuned version of `unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit`.
- **Finetuned from model:** `unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit`
- **Repository:** [More Information Needed]
- **Demo [optional]:** https://github.com/purut18/lucky

## Uses
This model is finetuned to convert natural language commands into structured JSON actions for a MacOS system agent.

### Downstream Use
This model is plugged into a larger ecosystem/app called Lucky.

## Training Details

### Training Data

The SLM has been trained on a dataset of 2,374 samples, available at 
```
https://huggingface.co/datasets/thakkar-puru/compuer-use-commands-and-json
```
