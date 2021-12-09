import nltk
import numpy as np
#nltk.download('punkt')     # run this line only for this first time
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    sentence_tokenized = nltk.word_tokenize(sentence)
    return sentence_tokenized

def stem(word):
    word_lowercased = word.lower()
    word_stemmed = stemmer.stem(word_lowercased)
    return word_stemmed

def bag_of_words(sentence, words):
    sentence_tokenized = [stem(w) for w in sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        if w in sentence_tokenized:
            bag[i] = 1.0
    return bag

