"""
Script for replacing adjectives in the evaluation dataset with their opposites.
Only one adjective at a time is replaced, so there are many resulting candidates
for one input sentence.
Output doesn't include labels, these have to be generated afterwards using
Monalog.
"""
from ..adjectives import replace_adjectives_pair
from ..data import read_dataset

if __name__ == "__main__":
    sequences, _ = read_dataset("data/evaluation.csv")
    with open("data/evaluation_adjectives.txt", "w") as f:
        for row in sequences:
            for premise, hypothesis in replace_adjectives_pair(row[0], row[1]):
                f.write(premise + "\n")
                f.write(hypothesis + "\n")
