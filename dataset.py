# my_sener_lib/dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Callable

class BaseNERDataset(Dataset):
    """Dataset base modular para NER."""
    
    def __init__(self, 
                 file_path: str,
                 tokenizer_name: str,
                 label2id: Dict[str, int],
                 max_length: int = 512,
                 transform: Optional[Callable] = None):
        """
        Args:
            file_path: caminho para o arquivo de dados
            tokenizer_name: nome do tokenizer HuggingFace
            label2id: mapeamento de labels para IDs
            max_length: comprimento máximo da sequência
            transform: função de transformação personalizada
        """
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.label2id = label2id
        self.max_length = max_length
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Carrega e parseia os samples do arquivo."""
        samples = []
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo {self.file_path} não encontrado")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    if self.transform:
                        sample = self.transform(sample)
                    samples.append(sample)
                except json.JSONDecodeError:
                    print(f"⚠️  Linha inválida ignorada: {line[:100]}...")
        return samples

    def _convert_entities_to_token_spans(self, text: str, entities: List[Dict], offsets: torch.Tensor) -> torch.Tensor:
        """Converte entidades de char spans para token spans."""
        L = offsets.size(0)
        R = len(self.label2id)
        labels = torch.zeros((L, L, R), dtype=torch.long)
        
        for ent in entities:
            start_char, end_char, label = ent["start"], ent["end"], ent["label"]
            if label not in self.label2id:
                continue
            r = self.label2id[label]

            # Encontrar tokens que cobrem [start_char, end_char)
            token_start = token_end = None
            for i, (s, e) in enumerate(offsets.tolist()):
                if s <= start_char < e:
                    token_start = i
                if s < end_char <= e:
                    token_end = i
                    
            if token_start is not None and token_end is not None and token_start <= token_end:
                labels[token_start, token_end, r] = 1
                
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        entities = sample.get("entities", [])

        # Tokenização
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)

        # Converter entidades para spans de tokens
        labels = self._convert_entities_to_token_spans(text, entities, offsets)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text,  # útil para debug
            "original_entities": entities  # útil para debug
        }

def collate_fn(batch):
    """Junta exemplos com padding."""
    max_len = max(len(item["input_ids"]) for item in batch)
    R = batch[0]["labels"].shape[-1] if batch else 0
    B = len(batch)

    input_ids = torch.zeros((B, max_len), dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.zeros((B, max_len, max_len, R), dtype=torch.long)
    texts = []
    original_entities = []

    for i, item in enumerate(batch):
        L = len(item["input_ids"])
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = item["attention_mask"]
        labels[i, :L, :L, :] = item["labels"]
        texts.append(item.get("text", ""))
        original_entities.append(item.get("original_entities", []))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "texts": texts,
        "original_entities": original_entities
    }

