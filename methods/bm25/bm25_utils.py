import numpy as np
import torch
import json
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from tqdm import tqdm
import bm25s

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from litgpt import Tokenizer

def decode_texts_from_pt(
    pt_file: Path, 
    tokenizer: Tokenizer
) -> Tuple[List[str], List[List[int]]]:
    """
    Load a .pt file (like 'train-1024.pt') which contains a list of dicts:
      [
        {
          "input_ids": [...],
          "labels": [...],
        },
        ...
      ]
    Then decode each "input_ids" to raw text.
    Returns:
      - List of decoded text strings.
      - List of input_ids (used as unique identifiers for reference docs).
    """
    data = torch.load(pt_file)
    decoded_texts = []
    input_ids_list = []
    for item in data:
        input_ids = item["input_ids"]
        input_ids_list.append(input_ids)
        input_ids_tensor = torch.tensor(input_ids)
        text_str = tokenizer.decode(input_ids_tensor)
        decoded_texts.append(text_str)
    return decoded_texts, input_ids_list

def build_bm25_index(reference_texts: List[str]) -> BM25Okapi:
    """
    Build a BM25Okapi index from a list of raw text strings.
    """
    tokenized_corpus = [doc.lower().split() for doc in reference_texts]
    return BM25Okapi(tokenized_corpus)

def compute_bm25_score(
    doc_text: str, 
    bm25: BM25Okapi,
    reference_ids: List[List[int]]
) -> List[Tuple[float, List[int]]]:
    """
    Given a single document's raw text, compute BM25 scores
    relative to the entire reference corpus.
    Returns a list of tuples: (score, reference_doc_id).
    """
    doc_tokens = doc_text.lower().split()
    scores = bm25.get_scores(doc_tokens)
    return list(zip(scores, reference_ids))

def compute_bm25_score_fn(
    doc_text: str, 
    bm25: BM25Okapi,
) -> float:
    """
    Given a single document's raw text, compute its average BM25 score
    relative to the entire reference corpus.
    Returns a single float number.
    """
    doc_tokens = doc_text.lower().split()
    scores = bm25.get_scores(doc_tokens)
    return sum(list(scores)) / len(scores)

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

def process_user_data_with_scores(
    user_data: List[Dict],
    bm25: BM25Okapi,
    reference_ids: List[List[int]]
) -> List[Dict]:
    """
    Given user data (list of dicts with 'text'), compute BM25 scores
    for each example and add them as 'bm25_scores'.
    """
    new_data = []
    for item in user_data:
        text_str = item["text"]
        scored_references = compute_bm25_score(text_str, bm25, reference_ids)
        new_item = dict(item)
        new_item["bm25_scores"] = scored_references
        new_data.append(new_item)
    return new_data

def compute_average_bm25_score(scored_references: List[Tuple[float, List[int]]]) -> float:
    """
    Compute the average BM25 score for a single datapoint given its list
    of (score, reference_doc_id) tuples.
    """
    if not scored_references:
        return 0.0
    total = sum(score for score, _ in scored_references)
    return total / len(scored_references)

def compute_average_bm25_scores(user_data_with_scores: List[Dict]) -> float:
    """
    Compute the average BM25 score for all datapoints given their lists
    of (score, reference_doc_id) tuples.
    """
    if not user_data_with_scores:
        return []
    average_scores = []
    for item in user_data_with_scores:
        avg_score = compute_average_bm25_score(item.get("bm25_scores", []))
        average_scores.append({
            "index": item["index"],
            "prediction": avg_score
        })
    return average_scores

def compute_bm25_score_with_dataset(
    dataset_texts: List[str],
    reference_texts: List[str],
    stopwords: str = "en",
    batch_size: int = 8192
) -> np.ndarray:
    """
    Using bm25s for fast BM25 scoring.
    average BM25 score for the doc over the entire reference corpus.
    """
    print(f"Reference set has {len(reference_texts)} items.")

    print("Tokenizing reference corpus with bm25s...")
    reference_tokens = bm25s.tokenize(
        reference_texts,
        stopwords=stopwords,
        # stemmer=stemmer #TODO: adding a stemmer later
    )

    print("Building bm25s BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(reference_tokens)

    print("Calculating BM25 metrics in batches...")
    n_docs = len(dataset_texts)
    all_scores = np.zeros(n_docs, dtype=np.float32)
    
    for start_idx in tqdm(range(0, n_docs, batch_size)):
        end_idx = min(start_idx + batch_size, n_docs)
        batch_docs = dataset_texts[start_idx:end_idx]

        batch_tokens = bm25s.tokenize(
            batch_docs,
            stopwords=stopwords,
            # stemmer=stemmer
        )
        
        # shape of scores_matrix = (batch_size, k)
        # k is the number of top reference doc selected
        scores_matrix = retriever.retrieve(batch_tokens, k=1024).scores

        batch_averages = scores_matrix.mean(axis=1)
        all_scores[start_idx:end_idx] = batch_averages
    return all_scores