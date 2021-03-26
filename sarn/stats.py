import argparse
from .data import read_dataset
from .labels import Label


parser = argparse.ArgumentParser(
    description="Statistics for a dataset"
)
parser.add_argument("dataset")


if __name__ == "__main__":
    args = parser.parse_args()

    sequences, labels = read_dataset(args.dataset)
