"""
Script for replacing adjectives in the MED dataset with their opposites.
Only one adjective at a time is replaced, so there are many resulting candidates
for one input sentence.
Output doesn't include labels, these have to be generated afterwards using
Monalog.
"""
import csv
import pandas as pd
from ..adjectives import replace_adjectives_pair

if __name__ == "__main__":
    # Download from https://github.com/verypluming/MED/blob/master/MED.tsv
    df = pd.read_csv("MED.tsv", sep="\t")
    with open("data/med_adjectives.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for _, row in df.iterrows():
            for output in replace_adjectives_pair(row["sentence1"], row["sentence2"]):
                writer.writerow(output)
