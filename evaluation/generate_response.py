import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel


def load_peft_model(base_model, checkpoint_path):
    """
    Load a PEFT (LoRA) model checkpoint and apply it to the base model.

    Args:
        base_model (transformers.PreTrainedModel): The base language model.
        checkpoint_path (str): Path or Hugging Face hub ID of the LoRA checkpoint.

    Returns:
        PeftModel: The base model with the LoRA adapter applied.
    """
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    peft_model.config.pad_token_id = base_model.config.pad_token_id
    return peft_model


def concat_messages(messages, tokenizer):
    """
    Concatenate a list of chat messages into a single string prompt.

    Args:
        messages (list): List of dicts with keys 'role' and 'content'.
        tokenizer (AutoTokenizer): Hugging Face tokenizer used for eos_token.

    Returns:
        str: Formatted prompt string.
    """
    message_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            message_text += "<|system|>\n" + content + "\n"
        elif role == "user":
            message_text += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            message_text += "<|assistant|>\n" + content + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"Invalid role: {role}")
    return message_text


def read_jsonl(file_path):
    """
    Read a .jsonl file and return a list of JSON objects.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: List of parsed JSON objects.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def save_to_jsonl(dataset, file_path):
    """
    Save a list of JSON records to a .jsonl file.

    Args:
        dataset (list): List of JSON serializable objects.
        file_path (str): Path to save the file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            json.dump(example, f)
            f.write('\n')


def batchify(lst, batch_size):
    """
    Split a list into batches of a given size.

    Args:
        lst (list): Input list to batch.
        batch_size (int): Number of items per batch.

    Yields:
        list: A batch of items.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def generate_response(checkpoint_path, model_name, input_path, output_path, batch_size, device='cuda'):
    """
    Generate model responses for prompts from a JSONL dataset.

    Args:
        checkpoint_path (str): Path to the PEFT LoRA checkpoint.
        model_name (str): Hugging Face model name or path.
        input_path (str): Path to input JSONL file with 'messages'.
        output_path (str): Path to save the output JSONL file.
        batch_size (int): Batch size for generation.
        device (str): Device to run model on ('cuda' or 'cpu').
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

    model = load_peft_model(model, checkpoint_path)
    model.to(device)
    model.eval()

    data = read_jsonl(input_path)
    records = []

    for batch in tqdm(list(batchify(data, batch_size))):
        prompts = [concat_messages(d['messages'][:1], tokenizer) for d in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        for idx in range(len(prompts)):
            input_ids = inputs["input_ids"][idx]
            output_ids = outputs[idx]
            generated_ids = output_ids[len(input_ids):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            batch[idx]['messages'][-1]['content'] = generated_text
            records.append(batch[idx])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_jsonl(records, output_path)
