import argparse
import re
from sklearn import metrics
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import numpy as np
import matplotlib.pyplot as plt
from .data import load_evaluation_dataset, tokenize
from .labels import Label


def get_valid_filename(s):
    s = str(s).strip().replace(" ", "-")
    return re.sub(r"(?u)[^-\w]", "-", s)


parser = argparse.ArgumentParser(
    description="ROC curve diagrams of a model on an evaluation dataset"
)
parser.add_argument("model")
parser.add_argument("dataset")


if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    dataset = load_evaluation_dataset(args.dataset, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    probas = []
    batch_cnt = len(dataloader)
    for i, batch in enumerate(dataloader):
        classification_logits = model(**batch).logits
        probas += torch.softmax(classification_logits, dim=1).tolist()
        if (i + 1) % 10 == 0:
            print("Batch", i + 1, "of", batch_cnt)
    probas = np.array(probas)

    plt.title("Receiver Operating Characteristic\n" + args.model + ", " + args.dataset)

    cont = probas[:, Label.CONTRADICTION.value]
    fpr, tpr, thresholds = metrics.roc_curve(
        dataset.labels, cont, pos_label=Label.CONTRADICTION.value
    )
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, "r", label=f"Contradiction (AUC: {roc_auc:0.2f})")

    neut = probas[:, Label.NEUTRAL.value]
    fpr, tpr, thresholds = metrics.roc_curve(
        dataset.labels, neut, pos_label=Label.NEUTRAL.value
    )
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, "g", label=f"Neutral (AUC: {roc_auc:0.2f})")

    ent = probas[:, Label.ENTAILMENT.value]
    fpr, tpr, thresholds = metrics.roc_curve(
        dataset.labels, ent, pos_label=Label.ENTAILMENT.value
    )
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, "b", label=f"Entailment (AUC: {roc_auc:0.2f})")

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    base_name = get_valid_filename(args.model) + "_" + get_valid_filename(args.dataset)
    plt.savefig(f"diagrams/roc_{base_name}.pdf")
    plt.savefig(f"diagrams/roc_{base_name}.svg")

    print("The diagram has been saved as PDF and SVG:")
    print(f"diagrams/roc_{base_name}.pdf")
    print(f"diagrams/roc_{base_name}.svg")
