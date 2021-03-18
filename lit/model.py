from lit_nlp.api import types as lit_types
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

from typing import List, Tuple, Iterable, Iterator, Text

JsonDict = types.JsonDict

import torch
import transformers

def _from_pretrained(cls, *args, **kw):
  """Load a transformers model in TF2, with fallback to PyTorch weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning("Caught OSError loading model: %s", e)
    logging.warning(
        "Re-trying to convert from PyTorch checkpoint (from_pt=True)")
    return cls.from_pretrained(*args, from_pt=True, **kw)


class Model(lit_model.Model):
    NLI_LABELS = ['entailment', 'neutral', 'contradiction']

    """Wrapper for a Natural Language Inference model."""

    def __init__(self, model_path):
        # Load the model into memory so we're ready for interactive use.
        # TODO Load model
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict_minibatch(self, inputs):
        # TODO: This is just copy&paste coding at the moment!
        pass

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
