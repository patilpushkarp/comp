import re
import pandas as pd
import gensim
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess_tweet(text):
    """Function to perform preprocessing of tweets"""
    text = str(text).lower()
    text = re.sub('@ *\w*', '', str(text))
    text = re.sub('#\w+', '', str(text))
    text = re.sub('\n', ' ', str(text))
    text = re.sub('\xa0', ' ', str(text))
    text = re.sub('&amp', ' ', str(text))
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^A-Za-z0-9 ]+', '', str(text))
    return text

def sentence_to_words(sentences):
    """Function to convert sentences to words"""
    return (gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in sentences)

def lemmatization(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Function to lemmatized tokenied sentence"""
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]



# def remove_hashtag(text):
#     """Function to remove hashtags which are not part of main text"""
    
#     # Having a single word with multiple hashtags
#     sep_text = text.split(" ")
#     rel_sent_words = [word for word in sep_text if word.count('#') < 2]
    
#     # Consecutive hashtags
#     remove_index = []
#     hashtag_index = []
#     for i, word in enumerate(rel_sent_words):
#         if word.startswith('#'):
#             hashtag_index.append(i)
#         else:
#             if len(hashtag_index) > 2:
#                 remove_index.append(hashtag_index)
#             else:
#                 hashtag_index = []
    
#     for indexes in remove_index:
#         start = indexes[0]
#         end = indexes[-1]
#         del rel_sent_words[start:end]
