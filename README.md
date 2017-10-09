# CSCI 4360 Final Project

## Data 

The sarcasm_v2.csv file in the data folder contains The Sarcasm Corpus V2, which is a subset of the Internet Argument Corpus. Each instance includes response text from quote-response pairs annotated for sarcasm. It contains data representing three categories of sarcasm: sarcasm, hyperbole, and rhetorical questions. 

The corpus is avaliable for download here: https://nlds.soe.ucsc.edu/sarcasm2

Each download is a random sample of the full dataset. 

The sample is a single CSV file with the following fields:

Corpus: the corpus type - one of GEN (general sarcasm), HYP (hyperbole), and RQ (rhetorical questions).
Label: the class label of the response utterance - one of "sarc" (sarcastic) or "notsarc" (not-sarcastic)
ID: a unique ID for the quote-response pair - {corpus}_{label}_{ID}. Each quote-response is independent, i.e. pairs with the same ID numbers across different datasets are not related.
Quote Text: the text of the dialogic parent of the response post, for contextResponse Text: the text of the response to the quote, annotated for sarcasm (i.e. the sarcasm label relates to this utterance)

The corpus should be cited as: Shereen Oraby, Vrindavan Harrison, Lena Reed, Ernesto Hernandez, Ellen Riloff and Marilyn Walker. "Creating and Characterizing a Diverse Corpus of Sarcasm in Dialogue." In The 17th Annual SIGdial Meeting on Discourse and Dialogue (SIGDIAL), Los Angeles, California, USA, 2016.