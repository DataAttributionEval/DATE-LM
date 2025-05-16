# Data Attribution Evaluation Benchmark (DATE-LM)

**DATE-LM** is a unified benchmark suite for systematically evaluating data attribution methods in the context of large language models (LLMs). Data attribution methods aim to trace how individual training examples influence model outputs — a capability that is increasingly essential for various tasks.

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
