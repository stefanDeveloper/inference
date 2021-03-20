"""
Script for replacing adjectives in the evaluation dataset with their opposites.
Only one adjective at a time is replaced, so there are many resulting candidates
for one input sentence.
Output doesn't include correct labels, the data has to be annotated by hand afterwards.
"""
import csv
from ..adjectives import replace_adjectives
from ..data import read_dataset
from ..labels import Label

if __name__ == "__main__":
    sequences, labels = read_dataset("data/evaluation.csv")
    with open("data/evaluation_adjectives.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row, label in zip(sequences, labels):
            for p, h in replace_adjectives(row[0], row[1]):
                writer.writerow((p, h, "", Label(label).name.lower()))
