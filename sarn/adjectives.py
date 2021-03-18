from torch.functional import norm
from transformers.utils.dummy_pt_objects import NoRepeatNGramLogitsProcessor
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from pattern.en import wordnet, comparative, superlative

nltk.download("averaged_perceptron_tagger")
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

ADJ = "JJ"
ADJ_CMP = "JJR"
ADJ_SUP = "JJS"


def opposite_adjectives(word):
    synsets = wordnet.synsets(word, pos=ADJ)
    antonyms = [a for s in synsets if s.antonym is not None for a in s.antonym]
    return {syn for a in antonyms for syn in a.synonyms}


def replace_sent_adjectives(sent):
    tokens = tokenizer.tokenize(sent)
    results = []
    original = []
    for token, tag in nltk.pos_tag(tokens):
        results = [tokens + [token] for tokens in results]
        if tag in [ADJ, ADJ_CMP, ADJ_SUP]:
            opposites = opposite_adjectives(token)
            if tag == ADJ_CMP:
                opposites = {
                    comparative(w) if nltk.pos_tag([w])[0][1] != ADJ_CMP else w
                    for w in opposites
                }
            elif tag == ADJ_SUP:
                opposites = {
                    superlative(w) if nltk.pos_tag([w])[0][1] != ADJ_SUP else w
                    for w in opposites
                }
            results += [
                original + [w if original else w.capitalize()] for w in opposites
            ]
        original.append(token)

    return [detokenizer.detokenize(tokens) for tokens in results]


def replace_adjectives(text):
    results = []
    original = []
    for sent in nltk.sent_tokenize(text):
        results = [sents + [sent] for sents in results]
        results += [original + [sent] for sent in replace_sent_adjectives(sent)]
        original.append(sent)
    return [" ".join(sents) for sents in results]


def replace_adjectives_pair(premise, hypothesis):
    for new_premise in replace_adjectives(premise):
        yield new_premise, hypothesis, "unknown"
    for new_hypothesis in replace_adjectives(hypothesis):
        yield premise, new_hypothesis, "unknown"


if __name__ == "__main__":
    sample = "Many delegates obtained interesting results from the survey. Many delegates obtained results from the survey."
    print("Input:", sample)
    for c in replace_adjectives(sample):
        print(c)
