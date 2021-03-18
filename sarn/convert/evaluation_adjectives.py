"""
Script for replacing adjectives in the evaluation dataset with their opposites.
Only one adjective at a time is replaced, so there are many resulting candidates
for one input sentence.
Output doesn't include labels, the data has to be annotated by hand afterwards.
"""
import csv
from ..adjectives import replace_adjectives_pair
from ..data import read_dataset

if __name__ == "__main__":
    sequences, _ = read_dataset("data/evaluation.csv")
    with open("data/evaluation_adjectives.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in sequences:
            for output in replace_adjectives_pair(row[0], row[1]):
                writer.writerow(output)
