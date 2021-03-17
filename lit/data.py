import pandas
from lit_nlp.api import types as lit_types
from lit_nlp.api.dataset import Dataset


class Data(Dataset):
    """Loader for MultiNLI development set."""

    NLI_LABELS = ["entailment", "neutral", "contradiction"]

    def __init__(self, path):

        # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
        df = pandas.read_csv(path, header=None, names=["premise", "hypothesis", "label"], sep=",")
        # Store as a list of dicts, conforming to self.spec()
        self._examples = [{
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": row["label"],
        } for _, row in df.iterrows()]

    """Should return a flat dictionary that describes the fields in each example"""

    def spec(self):
        return {
            "premise": lit_types.TextSegment(),
            "hypothesis": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.NLI_LABELS),
        }
