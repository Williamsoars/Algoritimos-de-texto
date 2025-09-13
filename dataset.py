# my_sener_lib/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ScholarXLDataset(Dataset):
    def __init__(self, file_path: str, plm_name: str, label2id: dict, max_length: int = 512):
        """
        file_path: caminho para o split JSONL (train.jsonl, dev.jsonl, test.jsonl)
        plm_name: nome do modelo (ex: microsoft/deberta-v3-large)
        label2id: dicionário {"ORG":0, "PER":1, ...}
        max_length: truncamento máximo
        """
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, use_fast=True)
        self.label2id = label2id
        self.max_length = max_length

        # carregar jsonl
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        text = example["text"]
        entities = example["entities"]  # lista de {start, end, label}

        # tokenizar
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)         # [L]
        attention_mask = enc["attention_mask"].squeeze(0)  # [L]
        offsets = enc["offset_mapping"].squeeze(0)      # [L,2] -> char spans

        L = input_ids.size(0)
        R = len(self.label2id)
        labels = torch.zeros((L, L, R), dtype=torch.long)

        # converter entidades para índices de tokens
        for ent in entities:
            start_char, end_char, label = ent["start"], ent["end"], ent["label"]
            if label not in self.label2id:
                continue
            r = self.label2id[label]

            # achar tokens que cobrem [start_char, end_char)
            token_start = token_end = None
            for i, (s, e) in enumerate(offsets.tolist()):
                if s <= start_char < e:
                    token_start = i
                if s < end_char <= e:
                    token_end = i
            if token_start is not None and token_end is not None and token_start <= token_end:
                labels[token_start, token_end, r] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch):
    """
    Junta exemplos de tamanhos diferentes em batch com padding.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    R = batch[0]["labels"].shape[-1]
    B = len(batch)

    input_ids = torch.zeros((B, max_len), dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.zeros((B, max_len, max_len, R), dtype=torch.long)

    for i, item in enumerate(batch):
        L = len(item["input_ids"])
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = item["attention_mask"]
        labels[i, :L, :L, :] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

