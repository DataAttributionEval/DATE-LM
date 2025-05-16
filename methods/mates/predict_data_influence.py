import argparse
import os

import torch
from datasets import Dataset
from litdata.streaming import StreamingDataset, TokensLoader
from modeling_data_influence_model import BertForSequenceClassification
from transformers import AutoTokenizer


class ModelAnnotator:
    def __init__(self, model_name, device_batch_size):
        self.model_name = model_name
        self.device_batch_size = device_batch_size

        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            problem_type="regression",
            num_labels=1,
        )
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.model.to(self.device)

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "device_batch_size": self.device_batch_size,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @torch.no_grad()
    def __call__(self, example, indices):
        output = {"index": indices}

        outputs = self.model(
            torch.tensor(example["input_ids"], device=self.device),
            attention_mask=torch.tensor(example["attention_mask"], device=self.device),
            token_type_ids=torch.tensor(example["token_type_ids"], device=self.device),
        )
        output["prediction"] = outputs.logits.detach().float().cpu().numpy()

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".") # not used
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--ckpt", type=int, default=10000) # not used
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)
    parser.add_argument("--model_dir", type=str, default="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/10000-data_influence_model-flan")
    parser.add_argument("--output_dir", type=str, default="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/mates_flan")
    parser.add_argument("--select_from_size", type=int, default=None, help="Limit the total number of examples for inference")


    args = parser.parse_args()
    print(args)

    model_dir = args.model_dir
    output_dir = args.output_dir

    num_proc = 8
    dataset = StreamingDataset(
        input_dir="/data/group_data/cx_group/data/fineweb/train/0",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    select_from_size = args.select_from_size if args.select_from_size else len(dataset)
    print(f"Selecting {select_from_size} from {len(dataset)} samples")

    shard_size = select_from_size // args.shard[1]

    start_index = args.base + args.shard[0] * shard_size
    end_index = (
        args.base + (args.shard[0] + 1) * shard_size
        if args.shard[0] + 1 < args.shard[1]
        else select_from_size
    )
    print(f"Processing shard {args.shard[0]} out of {args.shard[1]}")
    print(f"Start index: {start_index}, End index: {end_index}")
    dataset = dataset[start_index:end_index]

    dataset = Dataset.from_list([{"ori_input_ids": d[:2048]} for d in dataset])
    print(dataset)
    print("Total number of examples:", len(dataset))

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("checkpoints/EleutherAI/pythia-1b")
    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints/bert-base-uncased",
        max_length=2048,
        padding="max_length",
    )

    def preprocess_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["ori_input_ids"],
            skip_special_tokens=True,
        )
        encoding = tokenizer.batch_encode_plus(
            texts,
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        return encoding

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    print("After tokenization: Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator(model_dir, args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")