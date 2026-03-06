import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class TokenizerConfig:
    stoi: Dict[str, int]
    itos: Dict[int, str]


class CharTokenizer:
    """
    Very small character-level tokenizer.
    Builds a vocabulary from the training file and encodes/decodes text.
    """

    def __init__(self, vocab: List[str]):
        self.vocab = sorted(set(vocab))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.itos.get(t, "") for t in tokens)

    def to_config(self) -> TokenizerConfig:
        return TokenizerConfig(stoi=self.stoi, itos=self.itos)

    @classmethod
    def from_config(cls, cfg: TokenizerConfig) -> "CharTokenizer":
        vocab = [cfg.itos[i] for i in sorted(cfg.itos.keys())]
        tok = cls(vocab)
        tok.stoi = cfg.stoi
        tok.itos = cfg.itos
        return tok

    def save(self, path: str) -> None:
        cfg = self.to_config()
        data = {
            "stoi": cfg.stoi,
            "itos": {str(k): v for k, v in cfg.itos.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stoi = {k: int(v) if isinstance(v, bool) else v for k, v in data["stoi"].items()} if isinstance(
            next(iter(data["stoi"].values())), (str, bool, int)
        ) else data["stoi"]
        itos = {int(k): v for k, v in data["itos"].items()}
        cfg = TokenizerConfig(stoi=stoi, itos=itos)
        return cls.from_config(cfg)


def build_tokenizer_from_file(path: str) -> CharTokenizer:
    """Read the quotes file and build a character-level tokenizer."""
    text = Path(path).read_text(encoding="utf-8")
    # Ensure newline is present so model can learn quote boundaries
    if "\n" not in text:
        text += "\n"
    vocab = list(set(text))
    return CharTokenizer(vocab)

