#!/usr/bin/env python3

# Script only checks if the third column has valid values

import argparse

import csv

parser = argparse.ArgumentParser(description="Validate given dataset columns")
parser.add_argument("dataset")

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Checking: {args.dataset}")
    with open(args.dataset, newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            label = row[2]
            if label != "contradiction" and  label != "neutral" and  label != "entailment":
                print(f"Invalid value in line {i}: {label}")
