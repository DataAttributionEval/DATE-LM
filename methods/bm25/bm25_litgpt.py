import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from litdata.streaming import StreamingDataset, TokensLoader
from litgpt import Tokenizer
import bm25s


def get_decoded_samples(dataset, tokenizer, batch_size, decoded_train_dir):
    """
    Decode samples from the streaming dataset using the provided tokenizer.
    If a cache exists in `decoded_train_dir`, load and return it;
    otherwise, decode the samples, cache them, and return the list of decoded texts.
    
    Args:
        dataset: The StreamingDataset containing encoded samples.
        tokenizer: Tokenizer instance for decoding.
        batch_size: Number of samples to process per batch.
        decoded_train_dir: Directory to store/read the cached decoded samples.
    
    Returns:
        A list of decoded texts.
    """
    os.makedirs(decoded_train_dir, exist_ok=True)
    cache_file = os.path.join(decoded_train_dir, "decoded_samples.pkl")
    
    if os.path.exists(cache_file):
        print(f"Found cached decoded samples at: {cache_file}")
        with open(cache_file, "rb") as f:
            decoded_samples = pickle.load(f)
    else:
        print("Cached decoded samples not found, performing batch decode...")
        decoded_samples = []
        for start in tqdm(range(0, len(dataset), batch_size), desc="Decoding samples"):
            end = min(start + batch_size, len(dataset))
            batch_encoded = [dataset[i] for i in range(start, end)]
            batch_decoded = [tokenizer.decode(seq) for seq in batch_encoded]
            decoded_samples.extend(batch_decoded)
        print(f"Decoded {len(decoded_samples)} samples, saving to cache.")
        with open(cache_file, "wb") as f:
            pickle.dump(decoded_samples, f)
    return decoded_samples


def get_decoded_reference_data(reference_pt_path, reference_tokenizer_path):
    """
    Load and decode reference data from a .pt file using the provided tokenizer.
    
    Args:
        reference_pt_path: Path to the reference .pt file.
        reference_tokenizer_path: Path to the tokenizer for decoding the reference data.
    
    Returns:
        A list of decoded reference texts.
    """
    ref_tokenizer = Tokenizer(Path(reference_tokenizer_path))
    data = torch.load(Path(reference_pt_path))
    decoded_ref_texts = []
    for item in data:
        input_ids = item["input_ids"]
        input_ids_tensor = torch.tensor(input_ids)
        text_str = ref_tokenizer.decode(input_ids_tensor)
        decoded_ref_texts.append(text_str)
    return decoded_ref_texts


def compute_bm25_score_with_dataset(
    dataset_texts: List[str],
    reference_texts: List[str],
    stopwords: str = "en",
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Compute BM25 scores for each document in the dataset against a reference corpus.
    Each document receives a score equal to the average BM25 score over the reference corpus.
    
    Args:
        dataset_texts: List of texts from the dataset.
        reference_texts: List of texts from the reference corpus.
        stopwords: Language code for stopwords to filter out.
        batch_size: Number of documents to process per batch.
    
    Returns:
        A numpy array containing the BM25 scores.
    """
    print(f"Reference set has {len(reference_texts)} items.")

    print("Tokenizing reference corpus with bm25s...")
    reference_tokens = bm25s.tokenize(reference_texts, stopwords=stopwords)

    print("Building bm25s BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(reference_tokens)

    print("Calculating BM25 metrics in batches...")
    n_docs = len(dataset_texts)
    all_scores = np.zeros(n_docs, dtype=np.float32)

    for start_idx in tqdm(range(0, n_docs, batch_size), desc="Calculating BM25 scores"):
        end_idx = min(start_idx + batch_size, n_docs)
        batch_docs = dataset_texts[start_idx:end_idx]
        batch_tokens = bm25s.tokenize(batch_docs, stopwords=stopwords)
        scores_matrix = retriever.retrieve(batch_tokens, k=1024).scores
        batch_averages = scores_matrix.mean(axis=1)
        all_scores[start_idx:end_idx] = batch_averages

    return all_scores


def main(
    train_data_dir: str = "/data/group_data/cx_group/data/fineweb/train/0",
    reference_pt_path: str = "/data/group_data/cx_group/data/lambada_openai/train/train-1024.pt",
    output_dir: Path = Path("/data/group_data/cx_group/data/fineweb/train/0/bm25"),
    reference_tokenizer_path: str = "checkpoints/EleutherAI/pythia-1b",
    dataset_tokenizer_path: str = "checkpoints/EleutherAI/pythia-1b",
):
    """
    Compute BM25 metrics between a training dataset and a reference dataset, then save the metrics.
    
    Args:
        train_data_dir: Directory containing the training dataset.
        reference_pt_path: Path to the reference .pt file.
        output_dir: Directory to save the BM25 metrics.
        reference_tokenizer_path: Path to the tokenizer for the reference dataset.
        dataset_tokenizer_path: Path to the tokenizer for the training dataset.
    """
    # Load the training dataset using streaming.
    dataset = StreamingDataset(
        input_dir=train_data_dir,
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    print(f"Loading training dataset from: {train_data_dir}")

    # Initialize the tokenizer for the training dataset.
    dataset_tokenizer = Tokenizer(dataset_tokenizer_path)

    # Decode training samples (with caching).
    decode_train_batch_size = 8192
    decoded_train_dir = os.path.join(train_data_dir, "decoded")
    decoded_samples = get_decoded_samples(dataset, dataset_tokenizer, decode_train_batch_size, decoded_train_dir)

    # Decode reference dataset.
    print(f"Decoding reference dataset from: {reference_pt_path}")
    decoded_ref_texts = get_decoded_reference_data(reference_pt_path, reference_tokenizer_path)

    # Compute BM25 metrics.
    metrics = compute_bm25_score_with_dataset(decoded_samples, decoded_ref_texts)

    # Save BM25 metrics.
    print(f"Saving BM25 metrics to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", metrics)
    print("Done!")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)