import os
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .dataset import create_dataloaders, load_data_and_tokenizer
from .model import MiniGPT


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: MiniGPT, data_loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    model.train()
    return total_loss / max(1, total_tokens)


def train(cfg: TrainingConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device()
    cfg.device = device

    data, tokenizer, cfg = load_data_and_tokenizer(cfg)
    train_loader, val_loader = create_dataloaders(data, cfg)

    model = MiniGPT(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    step = 0
    for epoch in range(1, (cfg.max_iters // len(train_loader)) + 2):
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1

            if step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                progress.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), cfg.model_path)
                    tokenizer.save(cfg.tokenizer_path)

            if step >= cfg.max_iters:
                break
        if step >= cfg.max_iters:
            break

    if not os.path.exists(cfg.model_path):
        # Save final model if no best checkpoint was saved
        torch.save(model.state_dict(), cfg.model_path)
        tokenizer.save(cfg.tokenizer_path)


if __name__ == "__main__":
    cfg = TrainingConfig()
    train(cfg)

