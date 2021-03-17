"""
Script for replacing adjectives in the SuperGlue dataset with their opposites.
Only one adjective at a time is replaced, so there are many resulting candidates
for one input sentence.
Output doesn't include labels, these have to be generated afterwards using
Monalog.
"""
import csv
import pandas as pd
from ..adjectives import replace_adjectives_pair
from .superglue import logic_categories

if __name__ == "__main__":
    # Download from https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1
    df = pd.read_csv("diagnostic-full.tsv", sep="\t")
    df = df[
        df["Logic"].apply(
            lambda x: isinstance(x, str)
            and any(cat in logic_categories for cat in x.split(";"))
        )
    ]
    with open("data/superglue_adjectives.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for _, row in df.iterrows():
            for output in replace_adjectives_pair(row["Premise"], row["Hypothesis"]):
                writer.writerow(output)
