import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

from .labels import Label

parser = argparse.ArgumentParser(description="Classify entailment of sequences a and b")
parser.add_argument("a", type=str)
parser.add_argument("b", type=str)
parser.add_argument("--model", type=str, default="facebook/bart-large-mnli")
parser.add_argument("--print-tokens", action="store_true")
parser.add_argument("--print-results", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sequence_0 = args.a
    sequence_1 = args.b

    encoded = tokenizer(sequence_0, sequence_1, return_tensors="pt")
    classification_logits = model(**encoded).logits
    results = torch.softmax(classification_logits, dim=1).tolist()[0]

    if args.print_tokens:
        print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

    print(sequence_0)
    print(sequence_1)
    print(Label(np.argmax(results)).name)
    if args.print_results:
        for label in Label:
            print(f"{label.name}: {int(round(results[label.value] * 100))}%")
