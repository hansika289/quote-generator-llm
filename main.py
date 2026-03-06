import argparse

from src.config import TrainingConfig
from src.train import train
from src.generate import generate_quote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini LLM Quote Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the quote generator model")
    train_parser.add_argument(
        "--max-iters",
        type=int,
        default=2000,
        help="Number of optimisation steps to run.",
    )

    gen_parser = subparsers.add_parser("generate", help="Generate a new quote")
    gen_parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional starting text for the generated quote.",
    )
    gen_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainingConfig()
        cfg.max_iters = args.max_iters
        train(cfg)
    elif args.command == "generate":
        cfg = TrainingConfig()
        quote = generate_quote(prompt=args.prompt, max_new_tokens=args.max_new_tokens, cfg=cfg)
        print(quote)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

