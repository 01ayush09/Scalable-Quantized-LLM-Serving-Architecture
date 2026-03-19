# Scalable-Quantized-LLM-Serving-Architecture
A flexible, performance-focused architecture for deploying quantized large language models (LLMs) that balances memory, latency, and throughput with production-grade serving practices.

# Overview

This repository implements a scalable and quantization-aware LLM serving architecture designed to efficiently host large language models with reduced resource usage and high inference performance.

The goal is to provide a production-ready foundation for:

✔️ Compressing and quantizing large transformer models

✔ Evaluating quantized model performance against original models

✔ Serving inference requests efficiently with batching and caching

✔ Measuring accuracy, latency, and memory consumption

✔ Extending to distributed and multi-model serving workflows

This architecture is suitable for researchers, engineers, and teams working on large-scale generative systems, self-hosted API backends, and AI infrastructure.

# Key Features
Model Optimization & Quantization

Weight-Only Quantization
Reduce model size and memory requirements by quantizing weights to 4-bit (W4A16) precision while maintaining activation quality.

Activation-Aware Quantization (W8A8)
Experiment with 8-bit activation + weight quantization to trade accuracy vs. compression.

Performance Metrics
Measure and compare memory, latency, and accuracy across quantized and original models.

# Serving & Performance Techniques

> Efficient Batching for high GPU utilization

> KV Cache Analysis using adaptive such as past key-value reuse

> High-Performance Generation with top-k/top-p sampling

> Automatic Device Mapping for CPU/GPU load balancing

# | Path                       | Description                                                                 |
  
  | -------------------------- | --------------------------------------------------------------------------- |
  
  | `step_by_step_guide.ipynb` | Guided walkthrough of model loading, quantization, evaluation, and metrics. |
  
  | `llm_as_eval.py`           | Evaluates LLM responses using a scoring or judging mechanism.               |
  
  | `plt.ipynb`                | Notebook for plotting and visualizing memory/latency results.               |
  
  | `requirements.txt`         | Dependencies for running notebooks and scripts.                             |
  
  | `eval_data.json`           | Example evaluation dataset for prompting and comparison.                    |


#  Setup & Installation

1. Clone the repository

       git clone https://github.com/01ayush09/Scalable-Quantized-LLM-Serving-Architecture.git
       cd Scalable-Quantized-LLM-Serving-Architecture

2. Install dependencies

       pip install -r requirements.txt

3. Optional: Install CUDA/PyTorch with GPU support
   For optimal performance, install PyTorch with CUDA for your system:
   https://pytorch.org/get-started/

# Usage Guide
🔹 Load & Evaluate Base Model

Example code snippet to load a model and generate text:

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
🔹 Quantize Weights (W4A16)

    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4"
    )

    quant_model = AutoModelForCausalLM.from_pretrained(
       model_id,
       quantization_config=quant_config,
       device_map="auto"
    )
🔹 Evaluate Model Performance

Use custom scripts or notebooks to compare:

> Memory usage (MB)

> Latency (s)

> Output fidelity vs ground truth

# Why Quantization Matters

Quantization techniques reduce resource requirements by lowering numeric precision, which:

    Shrinks model size

    Improves throughput on limited hardware

    Reduces GPU memory footprint

    Enables larger batch serving

    Makes self-hosted LLMs feasible economically

For more on quantization strategies such as AWQ, SmoothQuant, and GPTQ, see related research literature.

# 🙌 Contributing

Contributions are welcome! You can:

⭐ Star the repo

🐛 Report issues

🔀 Submit pull requests with enhancements
