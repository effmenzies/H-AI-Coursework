import nltk, string, pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

def remove_fluff(input):
    tokens = word_tokenize(input)
    greetings = pd.read_csv('data/greetings.csv')
    for word in tokens:
        if word.lower() in greetings['Greeting'].values:
            tokens.remove(word)
    return " ".join(tokens)

def lemmatize(input):
    lem = WordNetLemmatizer()
    tag_dict = {'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB,'ADV':wordnet.ADV,'ADJ':wordnet.ADJ}
    input = tag_input(input)
    input_tokens=[]
    for token in input:
        word= token[0]
        tag = token[1]
        if tag in tag_dict.keys():
            input_tokens.append(lem.lemmatize(word, tag_dict[tag]))
        else: input_tokens.append(lem.lemmatize(word))
    return input_tokens

def tag_input(input):
    return nltk.pos_tag([word for word in word_tokenize(input) if word not in string.punctuation], tagset='universal')

def extract_nouns(input):
    i = remove_fluff(input)
    return [word[0] for word in tag_input(i) if (word[1]=='NOUN') & (word[0]!='hello')]

def extract_names(input):
    pattern = r"(?:call me|my (?:full )?name(?:'s| is)|i'm|i (?:like|prefer)(?: to be called)?)\s+([A-Za-z]+)"
    names = re.findall(pattern, input, re.IGNORECASE)
    return names

print(extract_names("Sure, my name's Effie but I prefer Eff"))