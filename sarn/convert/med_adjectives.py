"""
Script for replacing adjectives in the MED dataset with their opposites.
Only one adjective at a time is replaced, so there are many resulting candidates
for one input sentence.
Output doesn't include correct labels, the data has to be annotated by hand afterwards.
"""
import csv
import random
import pandas as pd
from ..adjectives import replace_adjectives

if __name__ == "__main__":
    # Download from https://github.com/verypluming/MED/blob/master/MED.tsv
    df = pd.read_csv("MED.tsv", sep="\t")
    results = []
    for _, row in df.iterrows():
        for p, h in replace_adjectives(row["sentence1"], row["sentence2"]):
            results.append((p, h, "", row["gold_label"]))
    # We are limiting the amount of rows as we still have to annotate them by hand
    limit = 1200
    samples = random.sample(results, limit)
    # Split into two so two people can label them in parallel without conflicts
    with open("data/med_adjectives_1.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(samples[: limit // 2])
    with open("data/med_adjectives_2.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(samples[limit // 2 :])
