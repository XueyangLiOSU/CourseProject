import pickle
import re
import string

#Load model and vectorizer
with open('classifier', 'rb') as f:
    cv_class, logreg = pickle.load(f)
    
#Ask for sentence input
words=input()

#Delete punctuations
def Text_clean(text):
    text=re.sub(r' \n|\n', ' ', text)
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#', '', text, flags=re.MULTILINE)
    text = re.sub(r'，', '', text, flags=re.MULTILINE)
    text = re.sub(r'！', '', text, flags=re.MULTILINE)
    text = re.sub(r'？', '', text, flags=re.MULTILINE)
    text = re.sub(r'。', '', text, flags=re.MULTILINE)
    text = re.sub(r'“', '', text, flags=re.MULTILINE)
    text = re.sub(r'”', '', text, flags=re.MULTILINE)
    text = re.sub(r'、', '', text, flags=re.MULTILINE)
    text=re.sub('[%s]' % re.escape(string.punctuation), ' ',text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round=lambda x:Text_clean(x)
words=Text_clean(words)

#corpus
corpus=[]
corpus.append(words)
#Vecoterization
test_sentence=cv_class.transform(corpus)
#predict
predict = logreg.predict(test_sentence)


print(predict)