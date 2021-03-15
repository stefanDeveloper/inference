from sklearn.model_selection import train_test_split
import torch
import csv
from .labels import Label

label_map = {
    "contradiction": Label.CONTRADICTION.value,
    "neutral": Label.NEUTRAL.value,
    "entailment": Label.ENTAILMENT.value,
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_dataset(dataset_path):
    sequences = []
    labels = []
    with open(dataset_path, newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            sequences.append((row[0], row[1]))
            labels.append(label_map[row[2]])

    return sequences, labels


def tokenize(texts, tokenizer):
    return tokenizer(
        [premise for premise, _ in texts],
        [hypothesis for _, hypothesis in texts],
        truncation=True,
        padding=True,
    )


def load_training_dataset(dataset_path, tokenizer, test_size=0.2):
    texts, labels = read_dataset(dataset_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size
    )
    train_encodings = tokenize(train_texts, tokenizer)
    test_encodings = tokenize(test_texts, tokenizer)
    train_dataset = Dataset(train_encodings, train_labels)
    test_dataset = Dataset(test_encodings, test_labels)
    return train_dataset, test_dataset


def load_evaluation_dataset(dataset_path, tokenizer):
    texts, labels = read_dataset(dataset_path)
    encodings = tokenize(texts, tokenizer)
    return Dataset(encodings, labels)
