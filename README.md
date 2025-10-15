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




Strengths
 1. Innovative focus – Targets honesty rather than only truthfulness, addressing a unique and meaningful aspect of Responsible AI.
 2. Solid technical foundation – Employs advanced methods such as Representation Engineering and Linear Artificial Tomography (LAT).
 3. Clear and structured workflow – The step-by-step plan (evaluation → extraction → control → comparison → assessment) is logical and feasible.
 4. Strong Responsible AI alignment – Directly relates to transparency, explainability, and safety.


Weaknesses / Improvement Opportunities
 1. Currently focuses mainly on one model family (e.g., LLaMA); future work could extend to other architectures to examine cross-model generality.

Suggestions
 1. In later stages, test the approach on different model families (e.g., Mistral, Gemma, or Falcon) to verify generalization.

- Wang Yang
Oct 12 at 9pm
Please address the feedback above and discuss them with the supervising TA going forward, as these will be important for Demos. Additionally, the proposal did not adequately explain the evaluation plans, e.g., datasets, benchmarks, metrics, etc.

Please contact the assigned TA for your project group and set a 20-minute time to meet in person and demo your project by the due date. 

For this demo, each group must:

- Set up a GitHub repository for the project and add the necessary items (e.g., README, etc.) and collaborators. (Done)
- Include the benchmark datasets and ML models in the repository. (You can share the URLs in the README if GitHub does not permit uploading large datasets.) (gonna be cities, inventors, facts, etc.) (LLaMA, Mistral, Gemma)
- Show installation of the necessary IDEs, frameworks, and environments. (copy from the Repe repo) 
- Run the benchmark models for the prediction. (get data from the previous final report?)
