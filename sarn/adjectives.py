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

def opposite_adjective(word):
    synsets = wordnet.synsets(word, pos=ADJ)
    antonyms = [a for s in synsets if s.antonym is not None for a in s.antonym]
    if antonyms:
        return antonyms[0].synonyms[0]
    return None


def replace_adjective(sents_tags, adj):
    token, tag = adj
    opposite = opposite_adjective(token)
    if not opposite:
        return None

    if tag == ADJ_CMP:
        opposite = (
            comparative(opposite) if nltk.pos_tag([opposite])[0][1] != ADJ_CMP else opposite
        )
    elif tag == ADJ_SUP:
        opposite = (
            superlative(opposite) if nltk.pos_tag([opposite])[0][1] != ADJ_SUP else opposite
        )

    sents = []
    for tags in sents_tags:
        tokens = [opposite if pair == adj else pair[0] for pair in tags]
        if tokens:
            tokens[0] = tokens[0].capitalize()
        sents.append(detokenizer.detokenize(tokens))
    return " ".join(sents)


def find_adjectives_sent(sent):
    tokens = tokenizer.tokenize(sent)
    tags = nltk.pos_tag(tokens)
    candidates = set()
    for token, tag in tags:
        if tag in [ADJ, ADJ_CMP, ADJ_SUP]:
            candidates.add((token, tag))
    return tags, candidates


def find_adjectives_text(text):
    sents_tags = []
    candidates = set()
    for sent in nltk.sent_tokenize(text):
        tags, new_candidates = find_adjectives_sent(sent)
        sents_tags += [tags]
        candidates = candidates.union(new_candidates)
    return sents_tags, candidates


def replace_adjectives(premise, hypothesis):
    p_tags, p_candidates = find_adjectives_text(premise)
    h_tags, h_candidates = find_adjectives_text(hypothesis)
    p_candidates = { adj for adj in p_candidates if any(adj in tags for tags in h_tags) }
    h_candidates = { adj for adj in h_candidates if any(adj in tags for tags in p_tags) }
    for adj in p_candidates:
        new_premise = replace_adjective(p_tags, adj)
        if new_premise:
            yield new_premise, hypothesis, "unknown"
    for adj in h_candidates:
        new_hypothesis = replace_adjective(h_tags, adj)
        if new_hypothesis:
            yield premise, new_hypothesis, "unknown"


if __name__ == "__main__":
    a, b = "Many delegates obtained interesting results from the survey.", "Many delegates obtained results from the survey."
    print("Input:", a, b)
    for row in replace_adjectives(a, b):
        print(row)
