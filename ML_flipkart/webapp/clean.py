import re
import string 
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 


lemmatizer = WordNetLemmatizer()

#we're gonna use this function in Bag of words/Count vectorizer (same thing)
def clean(data):
    #removes everything that isn't alphabets
    sentence = re.sub('^a-zA-Z', ' ', data) 
    #removes .READ MORE and READ MORE from the strings
    sentence = re.sub('.READ MORE', ' ',sentence)
    sentence = re.sub('READ MORE', ' ', sentence)
    
    #using string library to remove punctuations; numbers are also removed
    sentence = ''.join([x for x in sentence if x not in string.punctuation and not x.isdigit() ])
    
    #converting string to lower case
    sentence = sentence.lower()
    
    #tokenization
    tokens =  nltk.word_tokenize(sentence)
    
    #lemmatization
    lemmatized_tokens=[lemmatizer.lemmatize(token) for token in tokens]
    
    #removing stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [x for x in lemmatized_tokens if x not in stop_words]
    
    #joing and return
    return ' '.join(filtered_tokens)
    