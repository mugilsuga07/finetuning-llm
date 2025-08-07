# Fine-Tuning LLaMA 2 with QLoRA

This project demonstrates how to fine-tune Meta’s LLaMA 2 7B chat model using the QLoRA technique. It focuses on making large language models more accessible and efficient to train on limited hardware such as Google Colab. The notebook includes end-to-end implementation using the Hugging Face ecosystem.

## Project Objectives

- Fine-tune LLaMA 2 7B using instruction-based datasets
- Apply QLoRA to reduce memory usage via 4-bit quantization and low-rank adaptation
- Demonstrate a full supervised fine-tuning pipeline
- Enable scalable, low-cost deployment of custom LLMs

## Techniques Used

- QLoRA (Quantized Low-Rank Adaptation)
- LoRA (Parameter-Efficient Fine-Tuning)
- 4-bit quantization with `bitsandbytes`
- Supervised fine-tuning with Hugging Face `transformers`, `trl`, and `peft`
- Prompt formatting for LLaMA 2's chat-style architecture

## Dataset

**Source**: [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)  
A collection of instruction-following samples designed for training conversational agents.

## Running the Notebook

1. Open the `.ipynb` file in Google Colab or a local Jupyter environment.
2. Make sure to enable GPU (T4 or A100 recommended for Colab).
3. Execute the cells in order. Training should complete in 20–30 minutes for 1k samples.
4. The final model will be saved locally under the name `llama2-7b-chat-finetune`.

## Requirements

- Python 3.8+
- `transformers==4.31.0`
- `trl==0.4.7`
- `peft==0.4.0`
- `bitsandbytes==0.40.2`
- `accelerate==0.21.0`
- GPU with at least 15 GB VRAM (T4 or better)

