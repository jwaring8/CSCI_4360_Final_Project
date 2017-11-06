#import necessary libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

#load in dataset and take a peek at it
dataset = pd.read_csv('data/sarcasm_v2.csv')
print('Dataset Head')
print(dataset.head())

#let's take a look at more specifics
print('\n Number of Sarcastic v. Non-Sarcstic Comments')
print(dataset['Label'].value_counts())
print('\n Further Breakdown into Type of Comments')
print(dataset['Corpus'].value_counts())

#get numpy arrays
Y = dataset['Label'].values
quotes = dataset['Quote Text'].values
responses = dataset['Response Text'].values 

#Two examples of potential tokenizers to use
print("\n Tweet Tokenizer Example")
print(TweetTokenizer().tokenize(quotes[0])) #tweet tokenizer to recogonize potential emoticons and 
print("\n Word Tokenizer Example")
print(word_tokenize(quotes[0])) #more standard tokenizer on punctuation and words

#function for pre-processing text
def preprocess_text(text, tokenizer, stopwords=stopwords.words("english"), stemmer=PorterStemmer()):
    '''
    This function will remove stopwords from the text and perform stemming. Return tokenized sentences. 
    
    Params:
    text -- string we are looking at 
    tokenizer -- string of either 'twitter' or 'word' to specify which tokenizer to use
    stopwords -- list of stopwords to remove, default is the NLTK stopwords list
    stemmer -- stemming function to use, default is the PorterStemmer from NLTK
    
    Returns:
    cleaned_text -- text with removed stopwords and applied stemming
    
    '''
    #remove stopwords 
    cleaned_text =  ' '.join([word for word in text.split() if word not in stopwords])
        
    #perform stemming
    if(tokenizer == 'twitter'):
        tokens = TweetTokenizer().tokenize(cleaned_text)
        stemmed_tokens = [stemmer.stem(i) for i in tokens]
    elif(tokenizer == 'word'):
        tokens = word_tokenize(cleaned_text)
        stemmed_tokens = [stemmer.stem(i) for i in tokens]
    
    return stemmed_tokens

#call above function
for i in range(quotes.shape[0]):
    quotes[i] = preprocess_text(quotes[i], 'twitter')
for i in range(responses.shape[0]):
    responses[i] = preprocess_text(responses[i], 'twitter')

#let's make sure we get what we expect: tokenized sentences with no stopwords and removed stems
print("\n Check Result of Pre-Processing Text")
print(quotes[0])
print(responses[0])
