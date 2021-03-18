import numpy as np
from lit_nlp.api import types as lit_types
from lit_nlp.api import model as lit_model

from typing import List, Tuple, Text

import torch
import transformers

class Model(lit_model.Model):
    NLI_LABELS = ["contradiction", "neutral", "entailment"]

    """Wrapper for a Natural Language Inference model."""
    def __init__(self, model_path):
        # Load the model into memory so we"re ready for interactive use.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    def max_minibatch_size(self, config=None) -> int:
        # This tells lit_model.Model.predict() how to batch inputs to
        # predict_minibatch().
        # Alternately, you can just override predict() and handle batching yourself.
        return 32

    def predict_minibatch(self, inputs: List[lit_types.JsonDict], config=None) -> List[lit_types.JsonDict]:
        """Run prediction on a batch of inputs.

        Args:
          inputs: sequence of inputs, following model.input_spec()
          config: (optional) predict-time model config (beam size, num candidates,
            etc.)

        Returns:
          list of outputs, following model.output_spec()
        """

        premises = [line["premise"] for line in inputs]
        hypotheses = [line["hypothesis"] for line in inputs]

        encoded = self._tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, padding=True)
        classification_logits = self._model(**encoded).logits
        results = torch.softmax(classification_logits, dim=1).tolist()

        return [{"probas": probas} for probas in results]

    def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
        pass

    def fit_transform_with_metadata(self, indexed_inputs: List[lit_types.JsonDict]):
        pass

    def input_spec(self):
        """Describe the inputs to the model."""
        return {
            "premise": lit_types.TextSegment(),
            "hypothesis": lit_types.TextSegment(),
        }

    def output_spec(self):
        return {
            # The "parent" keyword tells LIT where to look for gold labels when computing metrics.
            # Note: "label" is a column in the dataset!
            # We use MulticlassPreds like in the examples.
            "probas": lit_types.MulticlassPreds(vocab=self.NLI_LABELS, parent='label'),
        }
