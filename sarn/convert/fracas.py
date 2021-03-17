"""
Script for extracting relevant samples from FraCaS
problem set and converting them to our project's dataset schema.
"""
from xml.etree import ElementTree
import csv

label_map = {
    "yes": "entailment",
    "no": "contradiction",
    "unknown": "neutral",
    "undef": "neutral"
}

if __name__ == "__main__":
    # Download from https://nlp.stanford.edu/~wcmac/downloads/fracas.xml
    with open("fracas.xml") as f:
        root = ElementTree.fromstring(f.read())

    entries = []

    for node in root:
        if node.tag != "problem":
            continue
        id = int(node.attrib["id"])
        # Category "1 GENERALIZED QUANTIFIERS" ends at id 80
        if id > 80:
            continue
        label = label_map[node.attrib["fracas_answer"]]
        premises = []
        hypothesis = ""
        for child in node:
            if child.tag == "p":
                premises.append(child.text.strip())
            elif child.tag == "h":
                hypothesis = child.text.strip()
        entries.append((" ".join(premises), hypothesis, label))

    with open("data/fracas.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(entries)
