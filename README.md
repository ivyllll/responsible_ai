# responsible_ai

## Overview

This project seeks to improve the honesty of large language models (LLMs) by utilizing representation engineering (RepE) techniques across different open-source LLMs, including the LLaMA family. The focus is on detecting and controlling dishonest outputs by analyzing internal representations of honesty. The study will compare multiple models and examine how their architecture and training influence the effectiveness of honesty control mechanisms. By applying methods such as Linear Artificial Tomography (LAT), we aim to enhance transparency and evaluate the efficacy of honesty improving interventions across various LLM architectures.

## Installation

Option A (pip, recommended):
```bash
pip install -U pip
pip install -r requirements.txt
```

Option B (conda):
```bash
conda env create -f environment.yml
conda activate truth_flip
```

Authenticate to Hugging Face (for gated models):
```bash
export HF_TOKEN="<your_huggingface_token>"
python -c "from huggingface_hub import login; import os; login(token=os.getenv('HF_TOKEN'))"
```

## Quickstart: Reproduce a Minimal Pipeline

Below runs a small Gemma-2 2B chat model to keep resource needs modest. It extracts layer activations for three prompt types, runs linear probes, and visualizes results.

1) Extract residual-stream activations
```bash
python scripts/extract_activations.py \
  --model_family Gemma2 \
  --model_size 2B \
  --model_type chat \
  --prompt_type truthful \
  --layers -1 \
  --datasets all \
  --device cuda:0

python scripts/extract_activations.py --model_family Gemma2 --model_size 2B --model_type chat --prompt_type neutral --layers -1 --datasets all --device cuda:0
python scripts/extract_activations.py --model_family Gemma2 --model_size 2B --model_type chat --prompt_type deceptive --layers -1 --datasets all --device cuda:0
```

2) Run probing across layers and prompt types
```bash
python scripts/run_probing_pipeline.py
```

3) Visualize probing accuracy curves (error bars + table)
```bash
python scripts/vis_probe_results.py \
  --model_family Gemma2 \
  --model_size 9B \
  --model_type chat \
  --save_dir experimental_outputs/probing_and_visualization/accuracy_figures
```

4) SAE-based feature shift (optional; requires `sae-lens` checkpoints)
```bash
python scripts/analyze_feature_shift_sae.py
python scripts/vis_feature_shift.py
```

Outputs are written under `experimental_outputs/` and figures/CSVs are saved in subfolders. Model weights cache to `llm_weights/`.

## Script Entry Points

- `scripts/eval_instruction_following.py`: Output-level accuracy on datasets for truthful/neutral/deceptive prompts (saves CSVs).
- `scripts/extract_activations.py`: Extracts per-layer residual activations for selected datasets/prompt type.
- `scripts/run_probing_pipeline.py`: Trains LR/TTPD probes layerwise and saves results for visualization.
- `scripts/vis_probe_results.py`: Plots accuracy-vs-layer curves with error bars and exports a Markdown table.
- `scripts/analyze_feature_shift_sae.py` and `scripts/vis_feature_shift.py`: SAE-based feature shift metrics and plots.

## Datasets

CSV format: two columns `statement` (string) and `label` (1=true, 0=false). Provided demo datasets include curated and complex sets (e.g., `animal_class.csv`, `cities.csv`, `sp_en_trans.csv`). Verify redistribution rights if replacing these with your own data.

## Citation

If you use this code or datasets, please cite our paper. See `CITATION.cff`.

## License

Code is released under the MIT License (see `LICENSE`). Model and dataset licenses remain with their respective owners.
