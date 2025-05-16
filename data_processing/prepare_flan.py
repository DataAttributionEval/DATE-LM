from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import torch
import random
from litgpt import Tokenizer
from litgpt.prompts import PromptStyle

def format_sample(entry, prompt_style, tokenizer, max_seq_length, mask_prompt, ignore_index, debug=False):
    """Format a single dataset entry into input_ids and labels."""
    convo = entry["messages"]
    prompt = prompt_style.apply(prompt=convo[0]["content"], **entry)
    prompt_and_response = prompt + convo[1]["content"]

    if debug:  # Print raw text for debugging
        print("Raw Prompt:")
        # print(prompt)
        # print("Raw Response:")
        # print(convo[1]["content"])
        # print("Combined Prompt and Response:")
        # print(prompt_and_response)

    # Tokenize the prompt and response
    encoded_prompt = tokenizer.encode(prompt, max_length=max_seq_length)
    encoded_prompt_and_response = tokenizer.encode(
        prompt_and_response, eos=True, max_length=max_seq_length
    )

    # Create labels with the prompt masked if required
    labels = encoded_prompt_and_response.clone()
    if mask_prompt:
        labels[: len(encoded_prompt)] = ignore_index

    return {
        "input_ids": encoded_prompt_and_response.tolist(),
        "labels": labels.tolist(),
    }


def prepare_tulu(
    destination_path: Path = Path("/data/group_data/cx_group/data/flan"),
    tokenizer_dir: Path = Path("checkpoints/EleutherAI/pythia-1b"),
    repo_id: str = "allenai/tulu-v2-sft-mixture",
    max_seq_length: int = 512,
    mask_prompt: bool = False,
    ignore_index: int = -1, # TODO litgpt default is 100, but we use -1 to match lambada
    include_multiturn_conversations: bool = False,
    seed: int = 42,
    train_split: float = 0.999,
    debug: bool = False,
) -> None:
    random.seed(seed)
    destination_path.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(repo_id, split="train")
    dataset = dataset.train_test_split(test_size=1 - train_split, seed=seed, shuffle=True)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(tokenizer_dir)
    prompt_style = PromptStyle.from_name("alpaca")

    def process_split(split, include_multiturn):
        formatted = []
        for entry in tqdm(split, desc="Processing dataset"):
            if entry["dataset"] not in ["flan_v2", "cot"]:
                continue
            convo = entry["messages"]
            if include_multiturn:
                for i in range(0, len(convo) - 1, 2):
                    formatted_sample = format_sample(
                        {"messages": convo[i : i + 2]},
                        prompt_style,
                        tokenizer,
                        max_seq_length,
                        mask_prompt,
                        ignore_index,
                        debug=debug,
                    )
                    if debug:  # Print the formatted sample if debugging
                        print("Formatted Sample (Tokenized):")
                        # print(formatted_sample)
                        print("len(input_ids):", len(formatted_sample["input_ids"]))
                    formatted.append(formatted_sample)
            else:
                formatted_sample = format_sample(
                    {"messages": convo[:2]},
                    prompt_style,
                    tokenizer,
                    max_seq_length,
                    mask_prompt,
                    ignore_index,
                    debug=debug,
                )
                if debug:  # Print the formatted sample if debugging
                    print("Formatted Sample (Tokenized):")
                    # print(formatted_sample)
                    print("len(input_ids):", len(formatted_sample["input_ids"]))
                formatted.append(formatted_sample)
        print(f"Processed {len(formatted)} data points in this split.")
        return formatted

    print("Processing train split...")
    train_data = process_split(dataset["train"], include_multiturn_conversations)
    if not debug:  # Only save if not debugging
        torch.save(train_data, destination_path / "train.pt")

    print("Processing test split...")
    test_data = process_split(dataset["test"], include_multiturn_conversations)
    if not debug:  # Only save if not debugging
        torch.save(test_data, destination_path / "test.pt")

    if not debug:
        print(f"Dataset saved to {destination_path}")
    else:
        print("Debugging complete. Data not saved.")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare_tulu)
    # train: Processed 98766 data points in this split.
    # test: Processed 104 data points in this split.