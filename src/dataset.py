from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .tokenizer import CharTokenizer, build_tokenizer_from_file
from .config import TrainingConfig


class QuotesDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_data_and_tokenizer(cfg: TrainingConfig) -> Tuple[torch.Tensor, CharTokenizer, TrainingConfig]:
    """Load text data, build tokenizer, and encode to tensor. Updates cfg.vocab_size."""
    if not Path(cfg.data_path).exists():
        raise FileNotFoundError(f"Data file not found at {cfg.data_path}")

    tokenizer = build_tokenizer_from_file(cfg.data_path)
    text = Path(cfg.data_path).read_text(encoding="utf-8")
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)

    cfg.vocab_size = tokenizer.vocab_size
    return data, tokenizer, cfg


def create_dataloaders(
    data: torch.Tensor, cfg: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    dataset = QuotesDataset(data, cfg.block_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    return train_loader, val_loader

