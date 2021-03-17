class MultiNLIData(Dataset):
    """Loader for MultiNLI development set."""

    NLI_LABELS = ['entailment', 'neutral', 'contradiction']

    def __init__(self, path):
        # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
        df = pandas.read_csv(path, sep='\t')
        # Store as a list of dicts, conforming to self.spec()
        self._examples = [{
            'premise': row['sentence1'],
            'hypothesis': row['sentence2'],
            'label': row['gold_label'],
            'genre': row['genre'],
        } for _, row in df.iterrows()]

    def spec(self):
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
            'label': lit_types.Label(vocab=self.NLI_LABELS),
            # We can include additional fields, which don't have to be used by the model.
            'genre': lit_types.Label(),
        }
