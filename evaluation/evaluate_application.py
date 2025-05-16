import torch
import sys
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import precision_recall_curve, auc
import argparse
from omegaconf import OmegaConf
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datamodules.load_data import *

toxicity_bias_tasks = [
    "XSTest-response-Hom", "XSTest-response-Het",
    "ToxicChat-Hom", "ToxicChat-Het",
    "JailbreakBench-Hom", "JailbreakBench-Het"
]

factual_tasks = ["Counterfact", "ftrace"]


def read_jsonl(file_path):
    """
    Read a .jsonl file into a list of JSON objects.

    Args:
        file_path (str): Path to a JSON Lines file.

    Returns:
        list: List of parsed JSON objects.
    """
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file]


def read_json(file_path):
    """
    Read a standard .json file into a dictionary.

    Args:
        file_path (str): Path to a JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def evaluate_fact_score_single(scores, fact_indices, k=5):
    """
    Compute Recall@K and ranks for one example.

    Args:
        scores (list or array): Retrieval scores over the dataset.
        fact_indices (list of int): Indices of known relevant training examples.
        k (int): Number of top scores to consider for Recall@K.

    Returns:
        tuple: (Recall@K, list of ranks for the ground truth indices)
    """
    sorted_indices = np.argsort(-np.array(scores))
    sorted_indices_topk = sorted_indices[:k]
    recall_at_k = len([idx for idx in fact_indices if idx in sorted_indices_topk]) / len(fact_indices)
    index_to_rank = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
    fact_ranks = [index_to_rank[idx] for idx in fact_indices if idx in index_to_rank]
    return recall_at_k, fact_ranks


def evaluate_fact(scores_list, fact_indices_list, k=5):
    recalls = []
    reciprocal_ranks = []
    for scores, fact_indices in zip(scores_list, fact_indices_list):
        recall_at_k, fact_ranks = evaluate_fact_score_single(scores, fact_indices, k=k)
        recalls.append(recall_at_k)
        if fact_ranks:
            reciprocal_ranks.append(1.0 / min(fact_ranks))
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(recalls), np.mean(reciprocal_ranks)


def get_fact_indices_counterfact(train, ref):
    """
    Retrieve relevant indices in training data for each counterfactual reference.

    Args:
        train (list): Training dataset.
        ref (list): Reference set with counterfactual_entity labels.

    Returns:
        list of list: Each sublist contains indices in training data matching a reference.
    """
    fact_indices = []
    for r in ref:
        fact_indices.append([
            idx for idx in range(len(train))
            if train[idx]['counterfactual_entity'] == r['counterfactual_entity'] and train[idx]['true_entity'] == r['true_entity']
        ])
        
    return fact_indices

def get_fact_indices_ftrace(train, ref):
    """
    Retrieve relevant indices in training data for each reference in ftrace.

    Args:
        train (list): Training dataset.
        ref (list): Reference set.

    Returns:
        list of list: Each sublist contains indices in training data matching a reference.
    """
    fact_indices = []
    for r in ref:
        fact_indices.append([
            idx for idx in range(len(train))
            if r["facts"][0] in train[idx]["facts"]
        ])
    return fact_indices

def get_unsafe_indices(train):
    """
    Get indices of unsafe examples in the training data.

    Args:
        train (list): Training dataset.

    Returns:
        list: Indices of examples labeled as 'Unsafe'.
    """
    return [idx for idx in range(len(train)) if train[idx]['type'] == "Unsafe"]


def AUPRC(score, task, unsafe_indices):
    """
    Compute AUPRC for binary classification of unsafe vs. benign examples.

    Args:
        score (list or np.array): Score vector over all training examples.
        task (str): Task name for logging.
        unsafe_indices (list): Indices labeled as unsafe (positive class).

    Returns:
        tuple: (precision, recall, f1_scores, pr_auc)
    """
    num_data = len(score)
    labels = np.zeros(num_data)
    labels[list(unsafe_indices)] = 1
    precision, recall, _ = precision_recall_curve(labels, score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    pr_auc = auc(recall, precision)
    return precision, recall, f1_scores, pr_auc


def evaluate_score_applcation(task_name, train, ref, score):
    """
    Main evaluation dispatcher for toxicity and factual tracing tasks.

    Args:
        task_name (str): Name of the evaluation task.
        train (list): Training dataset.
        ref (list): Reference examples used for evaluation.
        score (list or torch.Tensor): Attribution or similarity scores.

    Prints:
        Evaluation results (AUPRC, Recall@50, MRR) depending on the task.
    """
    # if isinstance(score, torch.Tensor):
    #     if score.dim() != 2:
    #         raise ValueError(f"Expected 2D tensor, got shape {score.shape}")
    #     score = score.tolist()

    if task_name in toxicity_bias_tasks:
        unsafe_indices = get_unsafe_indices(train)
        _, _, _, auprc = AUPRC(score, task_name, unsafe_indices)
        print(f"Task: {task_name}, AUPRC: {auprc:.4f}")
    elif task_name in factual_tasks:
        if task_name == "Counterfact":
            fact_indices_list = get_fact_indices_counterfact(train, ref)
        elif task_name == "ftrace":
            fact_indices_list = get_fact_indices_ftrace(train, ref)
        else:
            raise NotImplementedError(f"Fact index extraction not implemented for: {task_name}")
        recall, mrr = evaluate_fact(score, fact_indices_list, k=50)
        print(f"Task: {task_name}, Recall@50: {recall:.4f}, MRR: {mrr:.4f}")

def evaluate_application(args):
    train, ref = get_dataset(args.task,args.subset)
    score = read_json(args.score_path)
    evaluate_score_applcation(args.task,train,ref,score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--score_path", type=str, help="Path to the score")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    evaluate_application(config)

