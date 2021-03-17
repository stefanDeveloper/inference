"""
Script for extracting relevant samples from SuperGlue
dataset and converting them to our project's dataset schema.
"""
import pandas as pd

# These values can be found in the "Logic" column of the SuperGlue dataset.
# A dataset entry can have more then one of these, separated by a ";" character.
logic_categories = [
    # Quantification: Universal, Existential
    "Universal",
    "Existential",
    # Monotonicity: Upward Monotone, Downward Monotone, Non-Monotone
    "Upward Monotone",
    "Downward Monotone",
    "Non-Monotone",
]

if __name__ == "__main__":
    # Download from https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1
    df = pd.read_csv("diagnostic-full.tsv", sep="\t")
    df = df[
        df["Logic"].apply(
            lambda x: isinstance(x, str)
            and any(cat in logic_categories for cat in x.split(";"))
        )
    ]
    df[["Premise", "Hypothesis", "Label"]].to_csv(
        "data/superglue.csv", index=False, header=False
    )
