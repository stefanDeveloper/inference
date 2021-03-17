"""
Script for converting MED dataset to our project's dataset schema.
"""
import pandas as pd

if __name__ == "__main__":
    # Download from https://github.com/verypluming/MED/blob/master/MED.tsv
    df = pd.read_csv("MED.tsv", sep="\t")
    df[["sentence1", "sentence2", "gold_label"]].to_csv("data/med.csv", index=False, header=False)
