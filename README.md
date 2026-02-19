# Fin-Force

Official repository for the EMNLP paper:

> **Deriving Strategic Market Insights with Large Language Models: A Benchmark for Forward Counterfactual Generation**

---

## Overview

**Fin-Force** is a benchmark for evaluating the ability of large language models (LLMs) to generate *forward counterfactual scenarios* grounded in financial news. Given a real market news headline, a model must generate two plausible alternative market developments:

- **Risk Counterfactual**: a materially adverse market development relative to the original headline.
- **Opportunity Counterfactual**: a materially favorable market development relative to the original headline.

These counterfactuals must be *forward-compatible* — they represent plausible developments that could occur *after* the original news event, rather than negating it; and also *directional* - they represent clear and meaningful market shift to either an opportunity or risk in market conditions.

> [!NOTE]
> We have explored the financial domain for this benchmark, however we believe that the same concepts can be applied to various fields that require dynamic scenario-planning (climate disaster response etc.)

---

## Repository Structure

```
fin_force/
├── data/
│   ├── fin_force.json               # Main benchmark dataset (test split)
│   └── supplementary_data.json      # Supplementary SFT training data
├── evaluation/
│   ├── llm_evaluate.py              # LLM-based evaluation (directionality & forward compatibility)
│   ├── nonllm_evaluate.py           # Non-LLM evaluation (perplexity)
│   ├── run_eval.yaml                # Evaluation config template (with parameter comments)
│   └── rubrics/
│       ├── directionality_risk.txt
│       ├── directionality_opportunity.txt
│       └── forward_compatibility.txt
├── models/
│   ├── baseline_prompting/          # Zero-shot / few-shot prompting baselines
│   ├── cf_distillation/             # Counterfactual distillation via masked headlines
│   ├── self_rewarding_llm/          # Self-Rewarding LLM fine-tuning pipeline
│   └── lm-counterfactuals/          # Gumbel-based counterfactual generation
└── README.md
```

---

## Dataset

### `data/fin_force.json` — Main Benchmark

The primary test benchmark. Each entry contains a financial news headline along with its associated risk and opportunity counterfactuals.

**Key fields:**

| Field | Description |
|-------|-------------|
| `headline` | The original financial news headline |
| `classification` | Event type (e.g., `"market_event"`) |
| `category` | Market category (e.g., `"Financial markets & asset performance"`) |
| `output.original_headline` | Headline used as the evaluation prompt |
| `output.risk_counterfactual` | The risk counterfactual |
| `output.opportunity_counterfactual` | The opportunity counterfactual |

### `data/supplementary_data.json` — SFT Training Data for Self-Rewarding Language Models

Supplementary data for supervised fine-tuning. Shares the same schema as `fin_force.json`.

---

## Evaluation

Fin-Force uses four evaluation criteria across two scripts.

### Criteria

| Criterion | Type | Description |
|-----------|------|-------------|
| `directionality_risk` | LLM-based | Does the risk counterfactual represent a materially adverse market development? |
| `directionality_opportunity` | LLM-based | Does the opportunity counterfactual represent a materially favorable market development? |
| `forward_compatibility` | LLM-based | Is the counterfactual a plausible future development that does not negate the original headline? |
| `perplexity` | Non-LLM | Sentence-level perplexity (fluency proxy) via GPT-2. |

### LLM Evaluation (`evaluation/llm_evaluate.py`)

Uses the OpenAI API with structured outputs to score counterfactuals against the rubrics. Configure via `evaluation/run_eval.yaml`.

```python
from pydantic import BaseModel
from evaluation.llm_evaluate import LLMEvaluator
import os

class DirectionalityOutput(BaseModel):
    directionality: dict  # e.g. {"value": true}

evaluator = LLMEvaluator(
    api_key=os.environ["OPENAI_API_KEY"],
    input_file="data/fin_force.json",
    rubric_filepath="evaluation/rubrics/directionality_risk.txt",
    opportunity_rubric_filepath="evaluation/rubrics/directionality_opportunity.txt",
    model="gpt-4o",
    response_model=DirectionalityOutput,
    load_prompting_directly=True,       # True for fin_force.json format
    raw_output_file="results_raw.jsonl" # Optional: stream results to file
)

results = evaluator.evaluate_responses()
```

**Input format (`load_prompting_directly=True`):** JSON array where each item has an `"output"` key containing `original_headline`, `risk_counterfactual`, and `opportunity_counterfactual`.

**Input format (`load_prompting_directly=False`):** JSONL where each line has `"prompt"`, `"risk counterfactual"`, and `"opportunity counterfactual"`.

### Non-LLM Evaluation (`evaluation/nonllm_evaluate.py`)

Computes perplexity using GPT-2 (or any HuggingFace LM).

```python
from evaluation.nonllm_evaluate import NonLLMEval

evaluator = NonLLMEval("data/fin_force.json", load_prompting_directly=True)
scores = evaluator.compute_perplexity_scores(model_id="gpt2")
print(scores["risk_perplexity"])   # list of floats, one per example
print(scores["opp_perplexity"])
print(scores["org_perplexity"])    # perplexity of original headlines
```

### Configuration (`evaluation/run_eval.yaml`)

All evaluation parameters are documented with inline comments — API key placeholder, input/output paths, model, rubric paths, and which criteria to run.

### Rubrics

| File | Criterion | Output key |
|------|-----------|------------|
| `directionality_risk.txt` | Risk directionality | `directionality.value` (bool) |
| `directionality_opportunity.txt` | Opportunity directionality | `directionality.value` (bool) |
| `forward_compatibility.txt` | Forward compatibility | `forward_compatibility.value` (bool) |

---

## Models

All baseline methods are in `models/`.

### `models/baseline_prompting/` — Prompting Baselines

Zero-shot and few-shot counterfactual generation using frontier LLMs via the OpenAI or OpenRouter APIs.

**Files:**
- `_inference.py` — Config-driven inference script with retry logic (supports OpenAI and OpenRouter)
- `response_config.py` — Pydantic response schemas: `Counterfactuals`, `Counterfactuals_COT`, `Counterfactual_masked`
- `configs/` — One YAML per model/variant, all with commented parameters and API key placeholders
- `prompts/` — System prompt and few-shot example files

**Usage:**
```bash
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
python models/baseline_prompting/_inference.py --config models/baseline_prompting/configs/gpt4o.yaml
```

**Available configs:**

| Config | Model | Variant |
|--------|-------|---------|
| `gpt4o.yaml` | GPT-4o | Zero-shot |
| `gpt4o_fs_sample1.yaml` | GPT-4o | Few-shot (sample 1) |
| `claude3.5_haiku.yaml` | Claude 3.5 Haiku | Zero-shot |
| `gemini_2.0_flash.yaml` | Gemini 2.0 Flash | Zero-shot |
| `llama4_mav.yaml` | Llama 4 Maverick | Zero-shot |
| `qwen2.5_72B.yaml` | Qwen 2.5 72B | Zero-shot |
| `gpt4o_distil.yaml` | GPT-4o | CF Distillation (masked input) |

---

### `models/cf_distillation/` — Counterfactual Distillation

A two-step pipeline: (1) extract the topic word per headline and mask noun phrases to de-identify it, then (2) generate counterfactuals from the masked input using `baseline_prompting` with `gpt4o_distil.yaml`.

**Files:**
- `cf_distillation.py` — Step 1: topic extraction + noun-phrase masking via OpenAI
- `cf_distillation_config.yaml` — Commented config for step 1 (API key placeholder, paths)
- `process_distill_results.py` — Post-processing utility for step 1 output

**Usage:**
```bash
export OPENAI_API_KEY=sk-...
# Step 1: produce masked headlines
python models/cf_distillation/cf_distillation.py
# Step 2: generate counterfactuals from masked input
python models/baseline_prompting/_inference.py --config models/baseline_prompting/configs/gpt4o_distil.yaml
```

---

### `models/self_rewarding_llm/` — Self-Rewarding LLM

Iterative self-improvement pipeline: SFT → generate → LLM-score → DPO → repeat. Adapted from the Self-Rewarding Language Models framework.

**Pipeline:**

| Script | Stage |
|--------|-------|
| `scripts/00_sft.py` | Supervised fine-tuning on `supplementary_data.json` |
| `scripts/01_gen_prompts.py` | Generate evaluation prompts |
| `scripts/02_gen_responses.py` | Generate candidate responses |
| `scripts/03_gen_scores.py` | Score responses with LLM-as-a-judge |
| `scripts/04_gen_preferences.py` | Build preference pairs |
| `scripts/05_dpo.py` | Direct Preference Optimization |

**Other files:**
- `scripts/configs/` — YAML configs for each stage
- `src/srlm/` — Core model, trainer, and inference modules
- `process/` — Data preprocessing utilities
- `my_llm_judge_prompts/` — LLM judge prompt templates
- `self-reward_init_model.sh` / `self-reward_iter_train.sh` — Full pipeline shell scripts
- `multi_self-reward_*.sh` — Multi-model variants

**Usage:**
```bash
bash models/self_rewarding_llm/self-reward_init_model.sh   # Initial SFT + iteration 1
bash models/self_rewarding_llm/self-reward_iter_train.sh   # Subsequent iterations
```

---

### `models/lm-counterfactuals/` — Gumbel-Based Counterfactuals

Counterfactual generation using Gumbel-max sampling and model editing (MEMIT/ROME), adapted from the `lm-counterfactuals` framework.

**Files:**
- `run.py` — Main generation script
- `run_mimic.py` / `mimic.py` — MIMIC counterfactual method
- `sampling.py` — Gumbel sampling utilities
- `utils.py` — Shared utilities
- `analyze.py` / `analyze_edit.py` — Analysis and edit-distance tools
- `gumbel_config.yaml` — Generation configuration
- `example.ipynb` — Usage notebook

**Usage:**
```bash
python models/lm-counterfactuals/run.py
```

---

## Requirements

### Evaluation
```bash
pip install openai pydantic tqdm evaluate
```

### Baseline Prompting & CF Distillation
```bash
pip install openai pydantic tqdm instructor spacy
python -m spacy download en_core_web_sm
```

### Self-Rewarding LLM
```bash
pip install -r models/self_rewarding_llm/requirements.txt
```

### LM Counterfactuals
```bash
pip install -r models/lm-counterfactuals/requirements.txt
```

> **API Keys:** Never hardcode keys. Use environment variables for running the different models:
> ```bash
> export OPENAI_API_KEY=sk-...
> export OPENROUTER_API_KEY=sk-or-...
> ```

---

## Maintained By

**Keane Ong**
[Google Scholar](https://scholar.google.com/citations?user=fMPgRDMAAAAJ&hl=en) | [LinkedIn](https://www.linkedin.com/in/kowy/) | [Email](mailto:keaneong@mit.edu) | [Website](https://keanepotato.github.io)

---

## Citation

If you use Fin-Force in your research, please cite:

```bibtex
@inproceedings{ong-etal-2025-deriving,
    title = "Deriving Strategic Market Insights with Large Language Models: A Benchmark for Forward Counterfactual Generation",
    author = "Ong, Keane  and
      Mao, Rui  and
      Varshney, Deeksha  and
      Liang, Paul Pu  and
      Cambria, Erik  and
      Mengaldo, Gianmarco",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.575/",
    doi = "10.18653/v1/2025.emnlp-main.575",
    pages = "11411--11434",
    ISBN = "979-8-89176-332-6",
    abstract = "Counterfactual reasoning typically involves considering alternatives to actual events. While often applied to understand past events, a distinct form{---}forward counterfactual reasoning{---}focuses on anticipating plausible future developments. This type of reasoning is invaluable in dynamic financial markets, where anticipating market developments can powerfully unveil potential risks and opportunities for stakeholders, guiding their decision-making. However, performing this at scale is challenging due to the cognitive demands involved, underscoring the need for automated solutions. Large Language Models (LLMs) offer promise, but remain unexplored for this application. To address this gap, we introduce a novel benchmark, Fin-Force{---}**FIN**ancial **FOR**ward **C**ounterfactual **E**valuation. By curating financial news headlines and providing structured evaluation, Fin-Force supports LLM based forward counterfactual generation. This paves the way for scalable and automated solutions for exploring and anticipating future market developments, thereby providing structured insights for decision-making. Through experiments on Fin-Force, we evaluate state-of-the-art LLMs and counterfactual generation methods, analyzing their limitations and proposing insights for future research."
}
```

---


