# 💬HyLaT: Efficient Multi-Agent Communication via Hybrid Latent–Text Protocol

This is the official implementation of the paper: **[HyLaT:Efficient Multi-Agent Communication via Hybrid Latent–Text Protocol]** (arXiv link).

## Overview

**HyLAT** enables language model agents to communicate through a *hybrid* channel that combines **continuous latent tokens** (compressed cognitive signals) and **discrete text tokens** (explicit responses). This goes beyond conventional text-only communication by allowing agents to share nuanced, information-dense reasoning states that are too costly to express fully in natural language.

## Contents
- [Train HyLaT](#train-hylat)
- [Evaluation](#evaluation)



## Train HyLaT

The training framework proceeds in two stages:

| Stage | Script | Description |
|-------|--------|-------------|
| **Stage 1** | `hylat/train.py` | Train a single agent to produce hybrid latent+text outputs via self-distillation based on [CODI](https://github.com/zhenyi4/codi/tree/main) |
| **Stage 2** | `hylat/train_mas_dual.py` | Train two agents to mutually understand and respond to each other's hybrid outputs (You can also use `hylat/train_mas_general.py` for more than two agents, or `hylat/train_mas_dual_hetero.py` for heterogeneous agents) |



### Setup

**Clone the repository**:
```bash
git clone https://github.com/xymou/hylat.git
cd hylat
```

**Create environment**:
```bash
conda create -n hylat python=3.12
conda activate hylat
pip install -r requirements.txt
```

---

### Data Format

#### Stage 1

Each training sample is a JSON object with a `messages` list. Each message contains:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "...",
      "short_answer": "",
      "long_answer": ""
    },
    {
      "role": "assistant",
      "content": "",
      "short_answer": "...",
      "long_answer": "..."
    },    
  ]
}
```

| Field | Description |
|-------|-------------|
| `content` | The input question |
| `short_answer` | Concise answer (content of text channel)  |
| `long_answer` | Elaborate reasoning or explanations (content of latent channel) |

#### Stage 2

Each sample is a multi-turn dialogue between two agents (it can also be extended to more than 2 agents):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "",
      "question_agent1": "Solve: Janet's ducks lay 16 eggs per day...",
      "question_agent2": "Solve: Janet's ducks lay 16 eggs per day...",
    },
    {
      "role": "assistant",
      "content": "",
      "long_answer": "...",
      "short_answer": "...",
      "agent": 1
    },
    {
      "role": "assistant",
      "content": "",
      "long_answer": "...",
      "short_answer": "...",
      "agent": 2
    },    
  ],
  "mask_inter_res": true
}
```

| Field | Description |
|-------|-------------|
| `question_agent1` | Question seen by Agent 1 |
| `question_agent2` | Question seen by Agent 2 (may differ to simulate information asymmetry) |
| `short_answer` | Concise answer (content of text channel) |
| `long_answer` | Elaborate reasoning or explanations (content of latent channel) |
|`agent` | id of the agents |
| `mask_inter_res` | If `true`, only the final turn is supervised |

---

## Evaluation

We evaluate HyLaT on multi-agent debating (MAD) tasks using a framework built on top of [SDE](https://github.com/LittleDinoC/StateDelta/).

### Setup

Install the evaluation dependencies (separate from training):

```bash
cd eval
conda create -n SDE python=3.12
conda activate SDE
pip install -r requirements.txt
```

Configure data and project paths in `eval/src/root_path.py`:

```python
DATA_ROOT_PATH = "/path/to/data"   # root directory of evaluation datasets
ROOT_PATH      = "/path/to/eval"   # root directory of this eval folder
```

### Run the Evaluation

**HyLaT agents** (hybrid latent+text communication):

```bash
cd eval
bash scripts/eval_hylat.sh
```

This calls `src/main_latent.py` with `--method latent_stage2` and iterates over all datasets. The key arguments are:

| Argument | Description |
|----------|-------------|
| `--config_file` | Debate configuration (e.g., `configs/debate.json`) |
| `--model_name_or_path` | Path to the base LLM |
| `--conf_path` | Path to the HyLaT test config (set the ckpt path in the yaml) |
| `--dataset` | Dataset name|
| `--method` | Agent communication method (see list below) |
| `--output_dir` | Directory to save results |
| `--temperature` | Sampling temperature|

For setting more MAD methods or datasets, plz refer to [SDE](https://github.com/LittleDinoC/StateDelta/). 

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{xxx2025hylat,
  title   = {[Paper Title]},
  author  = {[Authors]},
  year    = {2025},
  journal = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

```bibtex
@article{shen2025codi,
      title={CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation}, 
      author={Zhenyi Shen and Hanqi Yan and Linhai Zhang and Zhanghao Hu and Yali Du and Yulan He},
      year={2025},
      journal={arXiv preprint arxiv:2502.21074},
}
```