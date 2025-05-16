# Application Tasks: Toxicity/Bias Filtering and Factual Attribution

DATE-LM supports application-driven evaluation for two critical tasks: **Toxicity/Bias Filtering** and **Factual Attribution**. These tasks assess the ability of data attribution methods to identify influential or harmful training examples that affect model behavior in safety and factual reasoning contexts.

---

## Step 1: Attribution Scoring

To begin, you must define the task configuration in a YAML file under the `configs/` directory.

### Example Configuration (`configs/toxicity-bias.yaml`)
```yaml
method: LESS
# Supported methods: Grad_Dot, Grad_Sim, LESS, DataInf, EKFAC

task: "Toxicity/Bias"
subset: "XSTest-response-Het"
# Supported subsets:
#   - XSTest-response-Het, XSTest-response-Hom
#   - ToxicChat-Het, ToxicChat-Hom
#   - XSTest-Het, XSTest-Hom

base_model_path: "EleutherAI/pythia-1b"
# Supported models: Pythia-1b, Llama-3.2-1B, Llama-3.1-8B

checkpoint: "DataAttributionEval/Pythia-1b-XSTest-response-Het"
device: "cuda"

# Hyperparameters for DataInf
regularization: 1e-5
fim_estimate_data_ratio: 1.0

# Hyperparameters for EKFAC
damping: 1e-7

# Hyperparameters for LESS
proj_dim: 8192

save_path: ./
```

These configurations enable reproducible evaluation and capture method-specific hyperparameters, as detailed in the DATE-LM paper. The benchmark supports plug-and-play attribution method substitution, promoting consistency across experiments.

### Run Attribution Scoring
Use the following script to compute attribution scores:

```sh
bash methods/run-application.sh configs/toxicity-bias.yaml
```

```sh
bash methods/run-application.sh configs/factual-attribution.yaml
```

The scoring step computes relevance scores for training data with respect to a held-out reference set (e.g., toxic prompts or factual outputs), following the unified pipeline described in the paper.

## Step 2: Task Evaluation

Once scores are generated, evaluate performance using the corresponding evaluation script. This step quantifies how well the selected training examples contribute to downstream utility (e.g., detecting toxic samples or retrieving factual evidence).

### Run Evaluation

```sh
bash evaluation/evaluate_application.sh configs/toxicity-bias.yaml /path/to/scores
```

```sh
bash evaluation/evaluate_application.sh configs/factual-attribution.yaml /path/to/scores
```

Evaluation metrics include:

- AUPRC for toxicity/bias filtering (homogeneous and heterogeneous settings)
- Recall@50 and MRR for factual attribution
