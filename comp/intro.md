# Comparing Fintech and NBFC using Twitter Data

## Introduction

The project compares the fintech and NBFC based on Twitter data.
The following fintechs are considered for the evaluation:
1. Faircent
2. Lendingkart
3. Mobikwik
4. Paytm
5. Pine labs

The following NBFCs are considered for analysis:
1. Aditya Birla Finance
2. Bajaj Finance
3. Cholamandalam
4. Muthoot Finance

## High-Level Steps

Following are the steps to be followed:
1. Scrap company data from Twitter
2. Select data which is not posted by the company
3. Divide the data into pre-COVID and post-COVID subsets
4. Perform the following preprocessing steps:
   1. Remove company handle names
   2. Remove hastags data
   3. Remove special characters
   4. Remove stopwords
5. Convert sentences to words
6. Create bigram and trigram models
7. Use the above models to create bigram and trigram
8. Create a dictionary from bigrams
9. Create a corpus using the dictionary
10. Perform hyper-parameter tuning for LDA model, using UMASS coherence score for evaluation
11. Use the selected hyper-parameters to create LDA model and determine the topic for each tweet
12. Use the model to find dominant words in each topic