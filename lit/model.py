import numpy as np
from lit_nlp.api import types as lit_types
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

from typing import List, Tuple, Iterable, Iterator, Text

JsonDict = types.JsonDict

import torch
import transformers


class Model(lit_model.Model):
    NLI_LABELS = ['entailment', 'neutral', 'contradiction']

    """Wrapper for a Natural Language Inference model."""
    def __init__(self, model_path):
        # Load the model into memory so we're ready for interactive use.
        # TODO Load model
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict_minibatch(self, inputs: List[JsonDict], config=None) -> List[JsonDict]:
        """Run prediction on a batch of inputs.

        Args:
          inputs: sequence of inputs, following model.input_spec()
          config: (optional) predict-time model config (beam size, num candidates,
            etc.)

        Returns:
          list of outputs, following model.output_spec()
        """

        print(type(inputs))
        print(inputs)

        pass

    # def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
    #     pass
    #
    # def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
    #     # Ignore; only used internally
    #     super.fit_transform_with_metadata(self, indexed_inputs)
    #     pass

    def input_spec(self):
        """Describe the inputs to the model."""
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
        }

    def output_spec(self):
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probas': lit_types.MulticlassPreds(vocab=self.NLI_LABELS, parent='label'),
        }
