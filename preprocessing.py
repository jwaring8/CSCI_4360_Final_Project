#import necessary libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

#load in dataset
dataset = pd.read_csv('data/sarcasm_v2.csv')

#get numpy arrays
Y = dataset['Label'].values
quotes = dataset['Quote Text'].values
responses = dataset['Response Text'].values 

#function for pre-processing text
def preprocess_text(text, tokenizer, stopwords=stopwords.words("english"), stemming=False, stemmer=PorterStemmer()):
    '''
    This function will remove stopwords from the text and perform stemming. Return tokenized sentences. 
    
    Params:
    text -- string we are looking at 
    tokenizer -- string of either 'twitter' or 'word' to specify which tokenizer to use
    stopwords -- list of stopwords to remove, default is the NLTK stopwords list
    stemming -- whether or not to perform stemming
    stemmer -- stemming function to use, default is the PorterStemmer from NLTK
    
    Returns:
    cleaned_text -- text with removed stopwords and applied stemming
    
    '''
    #remove stopwords 
    cleaned_text =  ' '.join([word for word in text.split() if word not in stopwords])
        
    #perform stemming
    if(stemming):
        if(tokenizer == 'twitter'):
            tokens = TweetTokenizer().tokenize(cleaned_text)
            stemmed_tokens = [stemmer.stem(i) for i in tokens]
        elif(tokenizer == 'word'):
            tokens = word_tokenize(cleaned_text)
            stemmed_tokens = [stemmer.stem(i) for i in tokens]
        return stemmed_tokens
    else:
        if(tokenizer == 'twitter'):
            tokens = TweetTokenizer().tokenize(cleaned_text)
        elif(tokenizer == 'word'):
            tokens = word_tokenize(cleaned_text)
        return tokens

#perform pre-processing of text
for i in range(quotes.shape[0]):
    quotes[i] = preprocess_text(quotes[i], 'twitter')
for i in range(responses.shape[0]):
    responses[i] = preprocess_text(responses[i], 'twitter')
