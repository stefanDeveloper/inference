import nltk
from pattern.en import wordnet, comparative, superlative

nltk.download("averaged_perceptron_tagger")

ADJ = "JJ"
ADJ_CMP = "JJR"
ADJ_SUP = "JJS"


def opposite_adjectives(word):
    synsets = wordnet.synsets(word, pos=ADJ)
    antonyms = [a for s in synsets if s.antonym is not None for a in s.antonym]
    return {syn for a in antonyms for syn in a.synonyms}


def replace_adjectives(text):
    text = text.split()
    tags = nltk.pos_tag(text)
    candidates = [[]]
    for token, tag in tags:
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
            words = opposites
            words.add(token)
            candidates = [c + [w] for c in candidates for w in words]
        else:
            candidates = [c + [token] for c in candidates]
    return [" ".join(tokens) for tokens in candidates]


for c in replace_adjectives("At least three female commissioners spend time at home."):
    print(c)
