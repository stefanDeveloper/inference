import argparse
from sklearn import metrics
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import numpy as np
from .data import load_evaluation_dataset, tokenize
from .labels import Label


parser = argparse.ArgumentParser(
    description="Accuracy of a model on an evaluation dataset"
)
parser.add_argument("model")
parser.add_argument("dataset")


if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    dataset = load_evaluation_dataset(args.dataset, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    labels = []
    batch_cnt = len(dataloader)
    for i, batch in enumerate(dataloader):
        classification_logits = model(**batch).logits
        probas = torch.softmax(classification_logits, dim=1).detach().numpy()
        labels += probas.argmax(axis=1).tolist()
        if (i + 1) % 10 == 0:
            print("Batch", i + 1, "of", batch_cnt)

    acc = metrics.accuracy_score(dataset.labels, labels)
    print(f"Accuracy of model {args.model} on dataset {args.dataset} is {acc*100:.2f}%")
