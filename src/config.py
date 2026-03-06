from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Paths
    data_path: str = "data/quotes.txt"
    model_dir: str = "saved_models"
    model_path: str = "saved_models/model.pt"
    tokenizer_path: str = "saved_models/tokenizer.json"

    # Tokenisation / sequence
    block_size: int = 128  # max context length

    # Model hyperparameters
    vocab_size: int = 0  # filled in after building tokenizer
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 4
    dropout: float = 0.1

    # Optimisation
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 200
    learning_rate: float = 3e-4

    # Misc
    seed: int = 42
    device: str = "cuda"  # overridden at runtime based on availability

