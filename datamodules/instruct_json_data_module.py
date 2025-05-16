import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as L
from minimal_multitask.utils import encode_with_messages_format
from transformers import DataCollatorForSeq2Seq
import random
import numpy as np
import torch

class InstructJsonDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path,
        batch_size=8,
        num_workers=8,
        tokenizer=None,
        max_seq_length=2048,
        ignore_token=-100,
        subset_size=2000, # number of training data
        val_split=0.1,
        seed=42,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_seq_length
        self.subset_size = subset_size
        self.val_split = val_split
        self.seed = seed
        self.ignore_token = ignore_token
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        if os.path.exists(self.train_data_path):
            load_dataset("json", data_files=self.train_data_path)["train"]
        else:
            raise ValueError(f"Train dataset not found at {self.train_data_path}")

    def setup(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Load and process train dataset
        if os.path.exists(self.train_data_path):
            dataset = load_dataset("json", data_files=self.train_data_path)["train"]
            # Shuffle the dataset with a fixed seed for reproducibility
            dataset = dataset.shuffle(seed=self.seed)
            # Take only the first `subset_size` examples if specified
            if self.subset_size:
                adjusted_size = int(self.subset_size / (1 - self.val_split))
                dataset = dataset.select(range(min(adjusted_size, len(dataset))))

            # Split into train and validation datasets
            split = dataset.train_test_split(test_size=self.val_split, seed=self.seed)
            train_dataset = split["train"]
            val_dataset = split["test"]

            def tokenize(x):
                return encode_with_messages_format(x, self.tokenizer, self.max_length, True, False, True, False)

            self.train_dataset = train_dataset.map(
                tokenize, num_proc=self.num_workers, load_from_cache_file=True, keep_in_memory=False
            )
            self.val_dataset = val_dataset.map(
                tokenize, num_proc=self.num_workers, load_from_cache_file=True, keep_in_memory=False
            )

            self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            self.val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        else:
            raise ValueError(f"Train dataset not found at {self.train_data_path}")

    def train_dataloader(self):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

    def val_dataloader(self):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
    
    # TODO WIP
    def select_train_subset(self, selected_indices):
        """
        Select a subset of the train_dataset based on the provided indices.

        Args:
            selected_indices (list): List of indices to select from the train_dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Train dataset has not been set up yet. Call `setup()` first.")
        
        # Select the subset of the train dataset
        self.train_dataset = self.train_dataset.select(selected_indices)