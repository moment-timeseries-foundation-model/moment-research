import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from moment.common import PATHS

from .base import FPTDataset


class NLPDataset(FPTDataset):
    def __init__(
        self, dataset_name: str, model_name: str, batch_size: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # loading dataset
        data = load_dataset(
            dataset_name, cache_dir=os.path.join(PATHS.DATA_DIR, "FPT_datasets/IMDB")
        )

        # tokenizing
        if model_name.startswith("gpt2"):
            print("Using GPT2 tokenizer")
            from transformers import GPT2Tokenizer

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Using Flan tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        print(f"Vocabulary size: {len(tokenizer)}")

        def preprocess_function(examples):
            return tokenizer(examples["text"], padding=True, truncation=True)

        tokenized_data = data.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def custom_collate_fn(batch):
            for d in batch:
                d.pop("text", None)
            batch_labels = [d["label"] for d in batch]
            batch_input_padded = data_collator(batch)
            batch_labels = torch.tensor(batch_labels)
            return {**batch_input_padded, "labels": batch_labels}

        # dataloaders
        self.d_train = DataLoader(
            tokenized_data["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
        )
        self.d_test = DataLoader(
            tokenized_data["test"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
        )

    def get_batch(self, batch_size=None, train=True):
        if train:
            batch = next(iter(self.d_train))
        else:
            batch = next(iter(self.d_test))

        x = batch["input_ids"]
        y = batch["labels"]
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        self._ind += 1
        return x, y
