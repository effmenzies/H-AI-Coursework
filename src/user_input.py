import nltk, string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def process_input(input):
    lem = WordNetLemmatizer()
    tag_dict = {'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB,'ADV':wordnet.ADV,'ADJ':wordnet.ADJ}
    input = nltk.pos_tag([word for word in word_tokenize(input) if word not in string.punctuation], tagset='universal')
    input_tokens=[]
    for token in input:
        word= token[0]
        tag = token[1]
        if tag in tag_dict.keys():
            input_tokens.append(lem.lemmatize(word, tag_dict[tag]))
        else: input_tokens.append(lem.lemmatize(word))
    return input_tokens


process_input('hello, how are you doing today?')
