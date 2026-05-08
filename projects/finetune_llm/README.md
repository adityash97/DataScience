````markdown
# Fine-Tuning Llama-2-7B using QLoRA with TRL & PEFT

A lightweight and memory-efficient implementation for fine-tuning **Llama-2-7B-Chat** using:

- QLoRA (4-bit quantization)
- PEFT (LoRA adapters)
- TRL `SFTTrainer`
- Hugging Face Transformers
- BitsAndBytes

This project demonstrates how to fine-tune large language models on consumer GPUs with reduced VRAM requirements.

---

# 🚀 Features

- 4-bit quantized model loading using BitsAndBytes
- Parameter-efficient fine-tuning using LoRA
- Supervised Fine-Tuning (SFT) with TRL
- Memory optimization for low-resource GPUs
- TensorBoard logging support
- Cosine learning rate scheduler
- Dataset preprocessing and subset selection

---

# 🧠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Model loading & training |
| PEFT | LoRA fine-tuning |
| TRL | SFTTrainer |
| BitsAndBytes | 4-bit quantization |
| Datasets | Dataset loading |
| TensorBoard | Training visualization |

---

# 📦 Installation

Install required dependencies:

```bash
pip install bitsandbytes trl datasets transformers peft
```

---

# 📚 Dataset

Dataset used:

```python
siddrao11/cs182-storytelling-dataset
```

The dataset is loaded using Hugging Face Datasets library.

Subset used for training:

- Train: 1500 samples
- Test: 500 samples
- Validation: 500 samples

---

# 🏗️ Model Used

Base model:

```python
NousResearch/Llama-2-7b-chat-hf
```

This project uses:

- Llama-2-7B Chat
- 4-bit quantization (QLoRA)
- LoRA adapters for efficient training

---

# ⚡ Quantization Configuration

The model is loaded in 4-bit mode using `BitsAndBytesConfig`.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Benefits

- Reduced VRAM usage
- Faster fine-tuning
- Consumer GPU compatibility
- Efficient large model training

---

# 🔥 LoRA Configuration

LoRA adapters are applied using PEFT.

```python
LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Advantages

- Train only small adapter weights
- Lower compute requirements
- Faster experimentation
- Preserves original model weights

---

# 🏋️ Training Configuration

Training is performed using `SFTTrainer`.

### Important Parameters

| Parameter | Value |
|---|---|
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Epochs | 1 |
| Scheduler | Cosine |
| Optimizer | paged_adamw_32bit |
| Logging Steps | 25 |

---

# 📂 Project Workflow

```text
Load Dataset
      ↓
Select Dataset Subsets
      ↓
Load Quantized Llama-2 Model
      ↓
Load Tokenizer
      ↓
Apply LoRA Adapters
      ↓
Initialize SFTTrainer
      ↓
Train Model
      ↓
Save Fine-Tuned Adapters
```

---

# 🖥️ GPU Requirement

This project requires CUDA-enabled GPU support.

The script automatically checks GPU availability:

```python
torch.cuda.is_available()
```

If CUDA is unavailable:

```text
CUDA is required but not available for bitsandbytes.
```

---

# 📊 TensorBoard Logging

Training logs are stored for TensorBoard visualization.

Launch TensorBoard:

```bash
tensorboard --logdir=./results
```

---

# ▶️ Running the Project

Execute the training script:

```bash
python train.py
```

---

# 📁 Suggested Project Structure

```bash
project/
│
├── train.py
├── results/
├── README.md
└── requirements.txt
```

---

# 🧪 Key Concepts Demonstrated

This project demonstrates:

- QLoRA fine-tuning
- Quantized LLM training
- Parameter-efficient training
- Supervised instruction tuning
- Memory optimization techniques
- Hugging Face training ecosystem

---

# 🎯 Learning Outcomes

By completing this project, you will understand:

- How QLoRA works
- How LoRA adapters reduce training cost
- How to fine-tune LLMs on limited hardware
- How TRL's `SFTTrainer` simplifies instruction tuning
- How quantization helps large model training

---

# 📌 Future Improvements

Possible enhancements:

- Add custom prompt formatting
- Add inference pipeline
- Push model to Hugging Face Hub
- Add evaluation metrics
- Add multi-GPU support
- Add experiment tracking with Weights & Biases
- Implement RAG integration

---

# 🤝 Acknowledgements

Libraries and frameworks used:

- Hugging Face Transformers
- TRL Library
- PEFT Library
- BitsAndBytes
- PyTorch

````
