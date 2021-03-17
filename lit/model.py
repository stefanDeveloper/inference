from lit_nlp.api import types as lit_types
from lit_nlp.api import model
from typing import List, Iterable

from transformers import AutoModelForSequenceClassification


class Model(model):
    NLI_LABELS = ['entailment', 'neutral', 'contradiction']

    """Wrapper for a Natural Language Inference model."""

    def __init__(self, model_path):
        # Load the model into memory so we're ready for interactive use.
        # TODO Load model
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)

    ##
    # LIT API implementations
    # TODO Fix these imports
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
        # TODO Import vocab
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probas': lit_types.MulticlassPreds(vocab=NLI_, parent='label'),
        }
