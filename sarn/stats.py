import argparse
from nltk.tokenize import word_tokenize
import numpy as np
from .data import read_dataset
from .labels import Label


parser = argparse.ArgumentParser(description="Statistics for a dataset")
parser.add_argument("dataset")


if __name__ == "__main__":
    args = parser.parse_args()

    sequences, labels = read_dataset(args.dataset)
    premises = [p for p, _ in sequences]
    hypotheses = [h for _, h in sequences]

    p_chars = np.array([len(p) for p in premises])
    h_chars = np.array([len(h) for h in hypotheses])
    p_words = np.array([len(p.split()) for p in premises])
    h_words = np.array([len(h.split()) for h in hypotheses])

    print(
        f"Premise length (characters): {p_chars.mean():.2f} avg, {int(np.median(p_chars))} median, {p_chars.min()} min, {p_chars.max()} max"
    )
    print(
        f"Hypothesis length (characters): {h_chars.mean():.2f} avg, {int(np.median(h_chars))} median, {h_chars.min()} min, {h_chars.max()} max"
    )
    print(
        f"Premise length (words): {p_words.mean():.2f} avg, {int(np.median(p_words))} median, {p_words.min()} min, {p_words.max()} max"
    )
    print(
        f"Hypothesis length (words): {h_words.mean():.2f} avg, {int(np.median(h_words))} median, {h_words.min()} min, {h_words.max()} max"
    )

    total = len(labels)
    print("Entries:", total)

    counts = np.bincount(labels)
    contr = counts[Label.CONTRADICTION.value]
    print(f"Contradictions: {contr} ({contr/total*100:.2f}%)")
    neut = counts[Label.NEUTRAL.value]
    print(f"Neutrals: {neut} ({neut/total*100:.2f}%)")
    ent = counts[Label.ENTAILMENT.value]
    print(f"Entailments: {ent} ({ent/total*100:.2f}%)")
