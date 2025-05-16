import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from litdata.streaming import StreamingDataset, TokensLoader
from litgpt import Tokenizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset


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



def main(
    train_data_dir: str = "/data/group_data/cx_group/data/fineweb/train/0",
    output_dir: Path = Path("/data/group_data/cx_group/data/fineweb/train/0/edu"),
    dataset_tokenizer_path: str = "checkpoints/EleutherAI/pythia-1b",
):
    """
    Compute edu metrics on a training dataset, then save the metrics.
    
    Args:
        train_data_dir: Directory containing the training dataset.
        output_dir: Directory to save the edu metrics.
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

    # Compute edu metrics.
    metrics = compute_edu_score_with_dataset(decoded_samples)

    # Save edu metrics.
    print(f"Saving EDU metrics to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", metrics)
    print("Done!")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)