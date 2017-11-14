import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from afinn import Afinn

#get our filepaths
trainDataPath = 'data/SARC/SARC-train.csv'
testDataPath = 'data/SARC/SARC-test.csv'

#read in train data
trainDataset = pd.read_csv(trainDataPath)
trainDataset = trainDataset.dropna(axis=0)
trainDataset['Response Text'] = trainDataset['Response Text'].astype(str)

Y_train = trainDataset['Label'].values
trainQuotes = trainDataset['Quote Text'].values
trainResponses = trainDataset['Response Text'].values 

# Read in test data
testDataset = pd.read_csv(testDataPath)
testDataset = testDataset.dropna(axis=0)
testDataset['Response Text'] = testDataset['Response Text'].astype(str)

Y_test = testDataset['Label'].values
testQuotes = testDataset['Quote Text'].values
testResponses = testDataset['Response Text'].values 

#function for preprocessing text
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

# run our preprocess text function
for i in range(trainQuotes.shape[0]):
    trainQuotes[i] = preprocess_text(trainQuotes[i], 'twitter')
for i in range(trainResponses.shape[0]):
    trainResponses[i] = preprocess_text(trainResponses[i], 'twitter')
for i in range(testQuotes.shape[0]):
    testQuotes[i] = preprocess_text(testQuotes[i], 'twitter')
for i in range(testResponses.shape[0]):
    testResponses[i] = preprocess_text(testResponses[i], 'twitter')

# function to create word2vec embeddings 
def w2v(sentences,sizeArg=100,windowArg=7,ngrams=1):
    
    '''
    This function generates word2vec embeddings of our sentences. 
    
    Params:
    sentences -- list of tokenized texts (i.e. list of lists of tokens)  
    sizeArg -- dimension of embedding. Default is 100. .
    windowArg -- context window used in embedding. Default is 7, which is a little large. We do this 
                because sarcasm has a very complex structure and consequently prediction of a token
                will require knowledge of a lot of the nearby tokens
    ngram -- transforms tokens into phrases of at most n words
    
    Returns: 
    model --  a dictionary mapping word -> sizeArg-dimensional vector
    '''
    
    #checks if we need to use the bigram transformer at all
    if ngrams == 1:
        model = Word2Vec(size=sizeArg , window=windowArg, min_count = 2, workers=-1, iter=10)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)


    else:
        #perform bigramming n times. Note we only perform it a max of 5 times since there are a negligent amount
        #of phrases of length bigger than 5
        for i in range(0,min(ngrams,5)):
            #Phrases creates an object with all the bigrams, then Phraser is a wrapper class used to access the
            #resulting corpus of bigrams. Using phraser also speeds up computation time when making the model
            if i==0:
                bigram = Phrases(sentences)
                bigram_phraser = Phraser(bigram)
            else:
                bigram = Phrases(bigram_phraser[sentences])
                bigram_phraser = Phraser(bigram)
        
        model = Word2Vec(size=sizeArg, window=windowArg, min_count = 2, workers=-1, iter=10)
        model.build_vocab(bigram_phraser[sentences])
        model.train(bigram_phraser[sentences], total_examples=model.corpus_count, epochs=model.iter)

    return model

# function to get feature vectors from word embeddings 
def getFeatureVectors(sentences, model, size=100):
    '''
    This function generates feature vectors from our sentences using word2vec embeddings.  
    
    Params:
    sentences -- list of tokenized texts (i.e. list of lists of tokens)   
    model --  a dictionary mapping word -> sizeArg-dimensional vector
    size -- should be same as sizeArg from w2v function
    
    Returns:
    feature_matrix -- matrix containing our feature vectors 
    '''
    feature_matrix = np.zeros((sentences.shape[0], size))
    for i in range(sentences.shape[0]):        
        feature_vector = np.zeros(size)
        num_ignored = 0
        for j in range(len(sentences[i])):
            try:
                feature_vector +=  model.wv[sentences[i][j]]
            except KeyError:
                num_ignored += 1
                
        feature_vector /= (len(sentences[i]) - num_ignored)
        feature_matrix[i] = feature_vector
    
    return feature_matrix

# run function to get word2vec feature matrix 
trainQuote_model = w2v(trainQuotes)
train_w2v_Xquote = getFeatureVectors(trainQuotes, trainQuote_model)
trainResponse_model = w2v(trainResponses)
train_w2v_Xresponse = getFeatureVectors(trainResponses, trainResponse_model)
testQuote_model = w2v(testQuotes)
test_w2v_Xquote = getFeatureVectors(testQuotes, testQuote_model)
testResponse_model = w2v(testResponses)
test_w2v_Xresponse = getFeatureVectors(testResponses, testResponse_model)

# save our numpy matrices to disk 
np.save('data/numpy/trainQuotes_w2v.npy', train_w2v_Xquote)
np.save('data/numpy/trainResponses_w2v.npy', train_w2v_Xresponse)
np.save('data/numpy/testQuotes_w2v.npy', test_w2v_Xquote)
np.save('data/numpy/testResponses_w2v.npy', test_w2v_Xresponse)
np.save('data/numpy/Y_train.npy', Y_train)
np.save('data/numpy/Y_test.npy', Y_test)

# Load in NRC Word-Emotion Association Lexicon dataset 
NRCdata = {}
with open("data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt", "r", encoding="utf-8") as nrc_file:
            for line in nrc_file.readlines():
                splited = line.replace("\n", "").split("\t")
                word, emotion, value = splited[0], splited[1], splited[2]
                if word in NRCdata.keys():
                    NRCdata[word].append((emotion, int(value)))
                else:
                    NRCdata[word] = [(emotion, int(value))]

# function to vectorize based off NRC dataset 
def NRCVectorization(sentences):
    '''
    This will create NRC lexicon feature vectors. 
    
    Params:
    sentences -- list of tokenized texts (i.e. list of lists of tokens)  
    
    Returns: 
    wordEmotionMatrix -- feature matrix containing NRC lexicons 
    '''
    wordEmotionMatrix = np.zeros((sentences.shape[0], 10))
    for i in range(sentences.shape[0]):
        wordEmotionVectors = []
        for j in range(len(sentences[i])):
            wordEmotVec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if sentences[i][j].lower() in NRCdata.keys():
                for k in range(len(NRCdata[sentences[i][j].lower()])):
                    wordEmotVec[k] = NRCdata[sentences[i][j].lower()][k][1]
            wordEmotionVectors.append(wordEmotVec)
        wordEmotionVectors = [sum(i) for i in zip(*wordEmotionVectors)]
        if wordEmotionVectors != []:
            wordEmotionMatrix[i] = wordEmotionVectors
        else:
            wordEmotionMatrix[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    return wordEmotionMatrix

# function to add afinn features as well 
def getLexiconFeatures(sentences):
    '''
    This will create lexicon feature matrix. 
    
    Params:
    sentences -- list of tokenized texts (i.e. list of lists of tokens)  
    
    Returns: 
    lexiconMatrix -- feature matrix containing lexicons
    '''
    
    emotionMatrix = NRCVectorization(sentences)
    afinnVector = np.zeros((sentences.shape[0], 1))
    afinn = Afinn()
    for i in range(sentences.shape[0]):
        afinnVector[i] = afinn.score(" ".join(sentences[i]))
    lexiconMatrix = np.hstack((emotionMatrix, afinnVector))
    return lexiconMatrix

# run functions to get lexicon features 
trainQuotes_lex = getLexiconFeatures(trainQuotes)
trainResponses_lex = getLexiconFeatures(trainResponses)
testQuotes_lex = getLexiconFeatures(testQuotes)
testResponses_lex = getLexiconFeatures(testResponses)

# save lexicon matrices to disk 
np.save('data/numpy/trainQuotes_lex.npy', trainQuotes_lex)
np.save('data/numpy/trainResponses_lex.npy', trainResponses_lex)
np.save('data/numpy/testQuotes_lex.npy', testQuotes_lex)
np.save('data/numpy/testResponses_lex.npy', testResponses_lex)