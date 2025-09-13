# train_sener.py
from my_sener_lib.dataset_transforms import create_scholar_xl_dataset
from my_sener_lib.dataset import collate_fn
from torch.utils.data import DataLoader

# Criar datasets
train_dataset = create_scholar_xl_dataset(
    file_path="data/raw/train",
    tokenizer_name="microsoft/deberta-v3-large",
    label2id_path="data/label2id.json",
    max_length=512
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=collate_fn,
    shuffle=True
)

# Testar
batch = next(iter(train_loader))
print(f"Batch input_ids shape: {batch['input_ids'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")
