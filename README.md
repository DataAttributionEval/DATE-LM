# <img src="assets/logo.png" alt="DATE-LM Logo" width="30">  DATE-LM: Data Attribution Evaluation Benchmark
**DATE-LM ** is a benchmark suite designed for evaluating data attribution methods in real-world applications of large language models (LLMs).


[![Hugging Face Organization](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DataAttributionEval-blue?style=flat-square&labelColor=gray)](https://huggingface.co/DataAttributionEval)

![Overview](assets/overview.png)


## Key Features

- **Three Core Evaluation Tasks**:
  - Training Data Selection (pre-training and fine-tuning)
  - Toxicity/Bias Filtering
  - Factual Attribution
- Modular pipeline supporting attribution scoring, subset selection, and task evaluation
- Plug-and-play support for new attribution methods
- Pre-trained and fine-tuned model checkpoints for reproducibility and efficiency
- Public leaderboard for standardized benchmarking and community engagement

---

## Environment Setup

Install the required dependencies using the files `env.yml` and `requirements.txt`:

```bash
conda env create --file env.yml --name myenv
conda activate myenv
pip install -r requirements.txt
```

## Evaluation workflow
The evaluation process consists of the following steps:

1. **Download Datasets and Models**: Choose a task and follow the preparation steps to download datasets and models.  

2. **Run Attribution Scoring**: Define a scoring function that takes a model checkpoint, a reference dataset, and a training dataset, and outputs an attribution score for each datapoint in the training dataset. 

3. **Run Task-Specific Evaluation**: Execute the evaluation pipeline for the selected task.

Detailed steps and tutorials are available in the following documentation files:

- **[Pre-train Data Selection](pretrain_data_selection.md)**
- **[Fine-tuning](finetune_data_selection.md)**
- **[Toxicity/Bias Filtering](Applications.md)**
- **[Factual Attribution](Applications.md)**


## Leaderboard
how to submit to leaderboard: 
https://huggingface.co/spaces/DataAttributionEval/DATE-LM-Leaderboard

## Citation