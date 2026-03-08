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


Performance Metrics
Measure and compare memory, latency, and accuracy across quantized and original models.
