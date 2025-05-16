import numpy as np
import torch
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from litgpt import Tokenizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

def load_user_data_jsonl(jsonl_path: Path) -> List[Dict]:
    """
    Load user training data from a JSONL file.
    Each line is expected to have a 'text' and a 'index' field.
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_edu_model(
    model_name_or_path: str = "HuggingFaceTB/fineweb-edu-classifier"
):
    """
    Load the FineWeb-EDU Hugging Face model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()
    return model, tokenizer

def compute_edu_score(
    text: str,
    model,
    tokenizer
) -> float:
    """
    Compute a single numeric score for educational content using
    the FineWeb-EDU classifier. The huggingface model here
    outputs a single regression-like logit. 
    """
    # print("decoded text: ", text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # The model is typically returning shape [batch_size, 1].
    logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
    score = float(logits.item())
    # print("decoded text score: ", score)
    # print("\n")
    return score

def process_user_data_with_edu_scores(
    user_data: List[Dict],
    model,
    tokenizer
) -> List[Dict]:
    """
    For each example, compute the EDU score
    """
    results = []
        
    for item in user_data:
        text_str = item["text"]
        score = compute_edu_score(text_str, model, tokenizer)
        
        results.append({
            "index": item["index"],
            "prediction": score
        })
    return results

# TODO: the average tokenized sequence len is 1993.26, which is far larger than 512. loss a lot information
# split each doc into several shards and compute the average score
def compute_edu_score_with_dataset(
    dataset_texts: List[str],
    batch_size: int = 64,
    max_length: int = 512,
    model_name_or_path: str = "HuggingFaceTB/fineweb-edu-classifier"
) -> np.ndarray:
    """
    Batched version of EDU scoring:
      1. Load model once
      2. Tokenize in larger batches
      3. Single forward pass per batch
    """
    edu_model, edu_tokenizer = load_edu_model(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edu_model.to(device)
    edu_model.eval()

    print("Calculating EDU metrics in batch ...")
    all_scores = []
    n_docs = len(dataset_texts)

    for start in tqdm(range(0, n_docs, batch_size)):
        end = min(start + batch_size, n_docs)
        batch_docs = dataset_texts[start:end]

        # tokenize the batch with overflowing tokens to split long docs into 512-token chunks.
        tokenized_input = edu_tokenizer(
            batch_docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True
        )
        # overflow_mapping tells which chunk belongs to which original document (index in batch_docs)
        overflow_mapping = tokenized_input.pop("overflow_to_sample_mapping").tolist()

        tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
        with torch.no_grad():
            outputs = edu_model(**tokenized_input)

        # outputs.logits has shape [num_chunks, 1]; squeeze to get a 1D array per chunk.
        chunk_scores = outputs.logits.squeeze(-1).cpu().numpy().astype(float)

        # group chunk scores by document using the overflow mapping.
        doc_chunk_scores = {i: [] for i in range(len(batch_docs))}
        for chunk_idx, doc_idx in enumerate(overflow_mapping):
            doc_chunk_scores[doc_idx].append(chunk_scores[chunk_idx])

        # Average scores of all chunks belonging to the same document.
        for i in range(len(batch_docs)):
            scores = doc_chunk_scores[i]
            if scores:
                avg_score = np.mean(scores)
            else:
                avg_score = float(0)
            all_scores.append(avg_score)

    return np.array(all_scores)

def compute_average_token_length(
    dataset_texts: List[str],
    model_name_or_path: str = "HuggingFaceTB/fineweb-edu-classifier"
) -> float:
    """
    Calculate the average token length of the dataset texts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    total_tokens = 0
    n_docs = len(dataset_texts)

    for text in tqdm(dataset_texts, desc="Calculating token lengths"):
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)

    average_length = total_tokens / n_docs
    print(f"Average token length: {average_length:.2f} tokens per document.")
    return average_length