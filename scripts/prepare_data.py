import argparse
import os
import pandas as pd


LABEL_DICT = {"bird": 0, "cat": 1, "dog": 2}


def main():
    parser = argparse.ArgumentParser(description="Generate train/val splits")
    parser.add_argument("--data-dir",   default="content",                    help="Output directory for train.csv and val.csv")
    parser.add_argument("--csv",        default="latent_vector_mapping.csv",  help="Path to the raw mapping CSV")
    parser.add_argument("--val-split",  type=float, default=0.10,             help="Fraction of data to use for validation")
    parser.add_argument("--seed",       type=int,   default=42,               help="Random seed for shuffling")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Mapping CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df["label"] = df["label"].map(LABEL_DICT)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_size = int((1.0 - args.val_split) * len(df))
    train_df = df[:train_size].reset_index(drop=True)
    val_df   = df[train_size:].reset_index(drop=True)

    os.makedirs(args.data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.data_dir, "val.csv"),     index=False)

    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Total : {len(df):,}")
    print(f"Splits saved to '{args.data_dir}/'")


if __name__ == "__main__":
    main()
