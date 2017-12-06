# CSCI 4360 Final Project

## Data 

### SARC

The Self-Annotated Reddit Corpus (SARC), a large corpus for sarcasm research and for training and evaluating systems for sarcasm detection can be found here: http://nlp.cs.princeton.edu/SARC/2.0/. The file is too large to host on GitHub, and thus we have only provided the URL. 

The SARC-train.csv and SARC-test.csv files contains a sizeable subset of the larger SARC dataset and can be found in the data/SARC folder. Each statement is furthermore self-annotated -- sarcasm is labeled by the author and not an independent annotator. 

Should be cited as: Khodak, M., Saunshi, N., & Vodrahalli, K. (2017). A Large Self-Annotated Corpus for Sarcasm. arXiv preprint arXiv:1704.05579.

## Docs

The docs folder contains both our final presentation in PDF form, as well as our NIPS style paper. We have provided both a PDF and the tex, sty, and bib files used to produce our final writeup. 

## Code 

### data.py 

This python script contains code used to produce the SARC-train.csv and SARC-test.csv files. The JSON file from the SARC dataset is ~2.5 GB, so we have to split up the JSON file into 10 different files. This code then reads in the 10 JSON files containing all the SARC data and pulls the subset described by the indices in the train-balanced.csv and test-balanced.csv files found in the data/SARC folder. Note that running this script from this repo alone is not sufficient, as we have not included all the different JSON files, as it is too large to host on GitHub. 

### vectorization.py 

This python script performs the vectorization described in our writeup and presentation (word2vec, Afinn, NRC Word-Emotion association). It saves all of our feature and label vectors as numpy files, which once again are not included in this repository becasue they are too large to host. 

### modeling.ipynb 

This Jupyter Notebook contains all the modeling and hyperparameter optimization that was pefromed to obtain our experimental SARC results. 
