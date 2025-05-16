import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Literal
from datasets import load_dataset, get_dataset_config_names
import datasets

task_map = {
    "Counterfact": "DataAttributionEval/Counterfact",
    "Toxicity/Bias": "DataAttributionEval/Toxicity-Bias-Filtering"
}

def get_dataset(task_name, subset_name):
    """
    Loads the specified dataset and model configuration from the Hugging Face Hub.

    Args:
        task_name: A string key identifying the dataset (e.g., "Counterfact").
        model_name: The config name corresponding to a specific model variant.

    Returns:
        A tuple containing the 'train' and 'ref' splits of the dataset.

    Raises:
        ValueError: If the task name or model name is not supported.
    """
    if task_name not in task_map:
        raise ValueError(f"Task name '{task_name}' is not supported. Valid options are: {list(task_map.keys())}")

    repo = task_map[task_name]
    configs = get_dataset_config_names(repo)

    if subset_name not in configs:
        raise ValueError(f"Subset name '{subset_name}' is not supported for task '{task_name}'. "
                         f"Available configs: {configs}")

    dataset = load_dataset(repo, subset_name)
    train = dataset['train']
    ref = dataset['ref']
    return train, ref

def prepare_chat_format(data):
    """
    Converts dataset examples into chat-style message format suitable for instruction-tuned LLMs.

    Args:
        data: A list or iterable of dataset examples, each with 'prompt' and 'response' keys.

    Returns:
        A list of dictionaries in chat format with alternating 'user' and 'assistant' messages.
    """
    records = []
    for d in data:
        record = {}
        messages = []
        user_dict = {
            "role": "user",
            "content": d["prompt"]
        }
        messages.append(user_dict)
        model_dict = {
            "role": "assistant",
            "content": d['response']
        }
        messages.append(model_dict)
        record['messages'] = messages
        records.append(record)
    return records
