# my_sener_lib/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler, AdamW
from tqdm.auto import tqdm

# opcional: para LoRA
from peft import LoraConfig, get_peft_model

def build_loss():
    """Binary cross entropy with logits (sigmoid)."""
    return nn.BCEWithLogitsLoss()

def train(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int = 3,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    output_dir: str = "checkpoints",
    device: str = "cuda"
):
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    loss_fn = build_loss()
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # [B,L,L,R]

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)  # [B,L,L,R]
            loss = loss_fn(logits, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.update(1)

        avg_train_loss = total_loss / len(train_loader)

        # avaliação
        val_loss = evaluate(model, dev_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f}")

        # salvar checkpoint se for o melhor
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(output_dir, f"best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[OK] Melhor modelo salvo em {ckpt_path}")


def evaluate(model, data_loader, loss_fn, device="cuda"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels.float())
            total_loss += loss.item()

    return total_loss / len(data_loader)


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.1, target_modules=["query", "value"]):
    """
    Aplica LoRA no encoder do modelo.
    """
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS"  # não existe "NER span" direto, usamos genérico
    )
    return get_peft_model(model, peft_config)
