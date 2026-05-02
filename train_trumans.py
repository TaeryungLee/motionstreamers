from __future__ import annotations

import argparse

from train_hsi_common import add_common_args, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRUMANS-style HSI comparison model.")
    add_common_args(parser, method="trumans")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
