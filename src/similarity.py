from nltk.corpus import wordnet as wn

def synonym(word):
    synsets = wn.synsets(word)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def are_synonyms(word1, word2):
    syns = synonym(word1)
    for s in syns:
        if s==word2:
            return True
    return False