"""
Script for converting HELP dataset to our project's dataset schema.
"""
import pandas as pd

# Download from https://github.com/verypluming/HELP/blob/master/output_en/pmb_train_v1.0.tsv
df = pd.read_csv("pmb_train_v1.0.tsv", sep="\t")
df[["ori_sentence", "new_sentence", "gold_label"]].to_csv("data/help.csv", index=False, header=False)
