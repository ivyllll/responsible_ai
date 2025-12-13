# responsible_ai

## Overview

This project seeks to improve the honesty of large language models (LLMs) by utilizing representation engineering (RepE) techniques across different open-source LLMs, including the LLaMA family. The focus is on detecting and controlling dishonest outputs by analyzing internal representations of honesty. The study will compare multiple models and examine how their architecture and training influence the effectiveness of honesty control mechanisms. By applying methods such as Linear Artificial Tomography (LAT), we aim to enhance transparency and evaluate the efficacy of honesty improving interventions across various LLM architectures.

## Installation

```bash
pip install -U pip
pip install -e .
```

Authenticate to Hugging Face (for gated models):
```bash
export HF_TOKEN="<your_huggingface_token>"
python -c "from huggingface_hub import login; import os; login(token=os.getenv('HF_TOKEN'))"
```
## Environemnt 

We ran the main experiments on an HPC cluster to speed up LLM inference and activation collection.
- OS: Linux (HPC login/compute nodes)
- Python: 3.10+ (3.11 also works)
- GPU: Recommended (e.g., 1× NVIDIA GPU with >= 16GB VRAM). CPU will run but is much slower.

## Datasets

CSV format: two columns `statement` (string) and `label` (1=true, 0=false). Provided demo datasets include curated and complex sets (e.g., `animal_class.csv`, `cities.csv`, `sp_en_trans.csv`). Verify redistribution rights if replacing these with your own data.

## Baseline vs experiments

run.sh is the entry point used for our final experiments (baseline vs improved)


## Contribution

Ruksaar Shaik
Collecting the Datasets and Baseline Experiments
Implementing Reading Pipeline
Xianxuan Long
Implementing Control Pipeline
Implementing Intervention Experiments


## Citation

If you use this code or datasets, please cite our paper. See `CITATION.cff`.

## License

Code is released under the MIT License (see `LICENSE`). Model and dataset licenses remain with their respective owners.
