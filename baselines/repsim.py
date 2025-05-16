import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
print("\n".join(sys.path))
import torch
import shutil
from tqdm import tqdm
from typing import Optional
from torch.nn.functional import normalize, cosine_similarity
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model
from datasets.data_utils import *
from torch.nn.utils.rnn import pad_sequence


def load_peft_model(base_model, checkpoint_path):
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    peft_model.config.pad_token_id = base_model.config.pad_token_id
    return peft_model



def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (representations) has been stored. 
    E.g. if we have files reps-100.pt, reps-200.pt, returns 200.
    Returns -1 if no such file is found.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
    if not files:
        return -1

    indices = []
    for file in files:
        # e.g. file name: reps-100.pt -> index=100
        # split(".")[0] = "reps-100" -> split("-")[1] = "100"
        stem = os.path.splitext(file)[0]
        if "-" in stem:
            idx_str = stem.split("-")[1]
            if idx_str.isdigit():
                indices.append(int(idx_str))
    return max(indices) if indices else -1

def merge_and_normalize_reps(output_dir: str, prefix="reps", final_name="all_orig.pt"):
    """ 
    Merge and L2‐normalize the representation files in ascending order, 
    then save them in a single file: e.g. all_orig.pt 
    """
    files = [f for f in os.listdir(output_dir) if f.startswith(prefix+"-")]
    # Sort numerically by index
    files.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))

    merged_list = []
    for file in files:
        path = os.path.join(output_dir, file)
        data = torch.load(path)
        # L2 normalize along dim=1
        data = normalize(data, dim=1)
        merged_list.append(data)

    if len(merged_list) == 0:
        print(f"No chunked reps found in {output_dir} with prefix={prefix}")
        return

    merged_data = torch.cat(merged_list, dim=0)
    output_file = os.path.join(output_dir, final_name)
    torch.save(merged_data, output_file)
    print(f"Saved merged & normalized reps (shape={merged_data.shape}) to {output_file}.")


def get_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [torch.tensor(x["input_ids"]) for x in batch]
        attention_masks = [torch.tensor(x["attention_mask"]) for x in batch]
        labels = [torch.tensor(x["labels"]) for x in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }
    return collate_fn
def collect_forward_reps(dataloader: torch.utils.data.DataLoader,
                         model: torch.nn.Module,
                         output_dir: str,
                         max_samples: Optional[int] = None,
                         save_interval: int = 160,
                         prefix: str = "reps"):
    """
    Collect forward representations for each batch in 'dataloader' using 'model'.
    Saves them in chunks: <prefix>-<count>.pt, then merges & normalizes at the end.

    The model is assumed to be a causal LM or similar that returns 'hidden_states'.
    We'll take the last hidden state of the last token as the representation just like LESS.

    Args:
        dataloader: DataLoader with columns 'input_ids' and 'attention_mask'.
        model: The model used to compute the representations.
        output_dir: Where partial and final .pt files will be stored.
        max_samples: If given, stops after collecting this many batches.
        save_interval: Save chunk after every this many batches.
        prefix: File prefix for chunked representation files.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_reps = []
    count = 0
    device = next(model.parameters()).device  # assume single GPU
    # Figure out if we already saved partial results in prior runs:
    max_index = get_max_saved_index(output_dir, prefix=prefix)

    for batch in tqdm(dataloader, desc="Collecting Forward Reps"):
        count += 1
        # If we have partially saved up to 'max_index', skip those
        if count <= max_index:
            print(f"Skipping batch {count} (already saved).")
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            # We assume a causal LM returning hidden_states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,             # labels needed for some models
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states
            ids = torch.arange(len(input_ids), device=device)
            pos = attention_mask.sum(dim=1) - 1  # index of last non‐pad token
            reps = hidden_states[-1][ids, pos]    # [batch_size, hidden_dim]

        all_reps.append(reps.cpu())

        # Save intermediate chunk
        if count % save_interval == 0:
            reps_tensor = torch.cat(all_reps)
            outfile = os.path.join(output_dir, f"{prefix}-{count}.pt")
            torch.save(reps_tensor, outfile)
            print(f"Saved partial reps to {outfile} (shape={reps_tensor.shape})")
            all_reps = []

        if max_samples is not None and count >= max_samples:
            break

    # Save any leftover
    if len(all_reps) > 0:
        reps_tensor = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"{prefix}-{count}.pt")
        torch.save(reps_tensor, outfile)
        print(f"Saved partial reps to {outfile} (shape={reps_tensor.shape})")

    # Merge & normalize all partial files
    merge_and_normalize_reps(output_dir, prefix=prefix, final_name="all_orig.pt")
    print("Finished collecting forward representations.")


def compute_forward_similarity(train_reps_dir: str,
                               ref_reps_dir: str,
                               output_file: str,
                               aggregator: str = "average"):
    """
    Loads the final merged representation file (e.g. 'all_orig.pt') from 
    train_reps_dir and ref_reps_dir, then computes similarity scores for 
    each train vector against the entire reference set.

    The aggregator can be:
      - "average"  : average similarity across all ref vectors
      - "max"      : maximum similarity
      - "sum"      : sum of similarities
    """

    train_file = os.path.join(train_reps_dir, "all_orig.pt")
    ref_file   = os.path.join(ref_reps_dir,   "all_orig.pt")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train reps not found at {train_file}")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Ref reps not found at {ref_file}")

    print("Loading normalized train reps from:", train_file)
    train_reps = torch.load(train_file)  # shape: [N, hidden_dim]

    print("Loading normalized ref reps from:", ref_file)
    ref_reps = torch.load(ref_file)      # shape: [N_ref, hidden_dim]

    print("Computing pairwise similarities...")
    # shape: [N, N_ref]
    sim_matrix = train_reps @ ref_reps.T
    # consider aggregator
    if aggregator == "average":
        scores = sim_matrix.mean(dim=1)
    elif aggregator == "sum":
        scores = sim_matrix.sum(dim=1)
    elif aggregator == "max":
        scores = sim_matrix.max(dim=1).values
    elif aggregator == "None":
        torch.save(sim_matrix.cpu(),output_file)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

    # Convert to Python list for saving
    scores_list = scores.cpu().tolist()
    print(f"Computed {len(scores_list)} influence scores with aggregator='{aggregator}'.")

    # Save the final scores
    print(f"Saving final scores to {output_file}")
    torch.save(scores_list, output_file)
    print("Done.")
    return scores_list


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a finetuned GPT-NeoX model checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model.pth")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1b", help="Model config name")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, default="./output.jsonl", help="Path to save generated JSONL")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_type", type=str, default=4, help="Batch size for generation")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    # state_dict = torch.load(Path(args.checkpoint_path) / "model.pth", map_location="cpu")
    # model.load_state_dict(state_dict, strict=False)
    model = load_peft_model(model,args.checkpoint_path)
    model.to("cuda")
    model.eval()
    os.makedirs(args.output_path+"/train_repsim/",exist_ok=True)
    os.makedirs(args.output_path+"/val_repsim/",exist_ok=True)
    data = read_jsonl(args.input_path)
    dataset_repsim = MessageDatasetRepSim(data,
                                   tokenizer=tokenizer,
                                   max_length=1024)
    collate = get_collate_fn(tokenizer)
    loader = DataLoader(dataset_repsim, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    collect_forward_reps(loader,model,args.output_path+"/train_repsim/",save_interval=1000)

    data_val = read_jsonl(args.val_path)
    dataset_repsim_val = MessageDatasetRepSim(data_val,
                                   tokenizer=tokenizer,
                                   max_length=1024)
    collate = get_collate_fn(tokenizer)
    loader_val = DataLoader(dataset_repsim_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    collect_forward_reps(loader_val,model,args.output_path+"/val_repsim/",save_interval=100)
    compute_forward_similarity(args.output_path+"/train_repsim/",args.output_path+"/val_repsim/",args.output_path + "/repsim.pt",args.save_type)
    



if __name__ == "__main__":
    main()
