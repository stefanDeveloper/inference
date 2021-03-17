class Model(Model):
    """Wrapper for a Natural Language Inference model."""

    NLI_LABELS = ['entailment', 'neutral', 'contradiction']

    def __init__(self, model_path, **kw):
        # Load the model into memory so we're ready for interactive use.
        self._model = _load_my_model(model_path, **kw)

    ##
    # LIT API implementations
    def predict(self, inputs: List[Input]) -> Iterable[Preds]:
        """Predict on a single minibatch of examples."""
        examples = [self._model.convert_dict_input(d) for d in inputs]  # any custom preprocessing
        return self._model.predict_examples(examples)  # returns a dict for each input

    def input_spec(self):
        """Describe the inputs to the model."""
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
        }

    def output_spec(self):
        """Describe the model outputs."""
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probas': lit_types.MulticlassPreds(vocab=NLI_LABELS, parent='label'),
        }
