import re
import os
import json
import glob
import tqdm
import pandas as pd
import numpy as np
import gensim
import spacy
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

class Processor:

    def __init__(self, config_file_path):

        fintech_path = './../data/fintech/'
        fintech_folders = glob.glob(f"{fintech_path}*/", recursive=True)
        self.fintech = [path.split('/')[-2] for path in fintech_folders]

        nbfc_path = './../data/nbfc/'
        nbfc_folders = glob.glob(f"{nbfc_path}*/", recursive=True)
        self.nbfc = [path.split('/')[-2] for path in nbfc_folders]

        with open(config_file_path) as f:
            self.config_data = json.load(f)
        
        self._fintech_stopwords = []
        for comp in self.fintech:
            self._fintech_stopwords = self._fintech_stopwords + self.config_data["stopwords"]["fintech"][comp]
        
        self._nbfc_stopwords = []
        for comp in self.nbfc:
            self._nbfc_stopwords = self._nbfc_stopwords + self.config_data["stopwords"]["fintech"][comp]

        self._config_stopwords = self._fintech_stopwords + self._nbfc_stopwords
        self.stopwords = STOPWORDS.union(set(self._config_stopwords))


    def preprocess_tweet(self, text):
        """Method to perform preprocessing of tweets"""
        text = str(text).lower()
        text = re.sub(r'@ *\w*', '', str(text))
        text = re.sub(r'#\w+', '', str(text))
        text = re.sub('\n', ' ', str(text))
        text = re.sub('\xa0', ' ', str(text))
        text = re.sub('&amp', ' ', str(text))
        text = re.sub(r'http\S+', '', str(text))
        text = re.sub(r'[^A-Za-z0-9 ]+', '', str(text))
        text = self.remove_stopwords(text)
        return text


    def remove_stopwords(self, text):
        """Method to remove stopwords from tweet text"""
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in self.stopwords]
        return " ".join(tokens_without_sw)


    def sentence_to_words(self, sentences):
        """Function to convert sentences to words"""
        return (gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in sentences)


    def lemmatization(self, words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """Function to lemmatized tokenied sentence"""
        doc = nlp(" ".join(words))
        return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

    def preprocess(self, df):

        # Select tweets with english language
        df = df[df['language'] == "en"]

        # Segregate the gate
        pre_df = df[df['date'] < '2020-02-01']
        post_df = df[df['date'] >= '2020-02-01']

        # Preprocess tweets
        pre_df.loc[:, 'preprocessed'] = pre_df['tweet'].apply(self.preprocess_tweet)
        post_df.loc[:, 'preprocessed'] = post_df['tweet'].apply(self.preprocess_tweet)

        # Create words data
        pre_data = pre_df['preprocessed'].values.tolist()
        pre_data_words = list(self.sentence_to_words(pre_data))

        # Create words data
        post_data = post_df['preprocessed'].values.tolist()
        post_data_words = list(self.sentence_to_words(post_data))

        # Build bigram and trigram models
        pre_bigram = gensim.models.phrases.Phrases(pre_data_words, min_count=5, threshold=10, connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
        pre_bigram_model = gensim.models.phrases.Phraser(pre_bigram)

        # Build bigram and trigram models
        post_bigram = gensim.models.phrases.Phrases(post_data_words, min_count=5, threshold=10, connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
        post_bigram_model = gensim.models.phrases.Phraser(post_bigram)

        # Convert sentence to words
        pre_df.loc[:, 'sep_words'] = pre_df['preprocessed'].apply(lambda x: list(self.sentence_to_words([x]))[0])
        post_df.loc[:, 'sep_words'] = post_df['preprocessed'].apply(lambda x: list(self.sentence_to_words([x]))[0])

        # Create bigrams
        pre_df['bigram'] = pre_df['sep_words'].apply(lambda x: pre_bigram_model[x])
        post_df['bigram'] = post_df['sep_words'].apply(lambda x: post_bigram_model[x])

        return pre_df, post_df


    def compute_coherence_values(self, corpus, dictionary, k, alpha, beta, texts, coherence='u_mass'):

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=k, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=alpha,
                                            eta=beta)
        
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, 
                                                            texts=texts, 
                                                            dictionary=dictionary, 
                                                            coherence=coherence)
        
        return coherence_model_lda.get_coherence()

    def perform_hyperparameter_tuning(self, corpus, dictionary, texts, path=None, min_topic=2, max_topic=15, corpus_sets_cutoffs=[0.75, 1]):

        grid = {}
        grid['Validation_Set'] = {}

        # Topics range
        min_topics = min_topic
        max_topics = max_topic
        step_size = 1
        topics_range = range(min_topics, max_topics, step_size)

        # Alpha parameter
        alpha = list(np.arange(0.01, 1, 0.3))
        alpha.append('symmetric')
        alpha.append('asymmetric')

        # Beta parameter
        beta = list(np.arange(0.01, 1, 0.3))
        beta.append('symmetric')

        # Validation sets
        num_of_docs = len(corpus)
        corpus_sets = []
        corpus_title = []
        for cutoff in corpus_sets_cutoffs:
            corpus_sets.append(gensim.utils.ClippedCorpus(corpus, int(num_of_docs*cutoff)))
            corpus_title.append(f"{cutoff*100}% Corpus")
            
        model_results = {'Validation_Set': [],
                        'Topics': [],
                        'Alpha': [],
                        'Beta': [],
                        'Coherence': []
                        }

        iterations = len(topics_range) * len(alpha) * len(beta) * len(corpus_sets)

        if 1 == 1:
            pbar = tqdm.tqdm(total=iterations)
            
            # iterate through validation corpuses
            for i in range(len(corpus_sets)):
                # iterate through number of topics
                for k in topics_range:
                    # iterate through alpha values
                    for a in alpha:
                        # iterare through beta values
                        for b in beta:
                            # get the coherence score for the given parameters
                            cv = self.compute_coherence_values(corpus=corpus_sets[i], 
                                                        dictionary=dictionary, 
                                                        k=k, alpha=a, beta=b,
                                                        texts=texts,
                                                        coherence='u_mass')
                            # Save the model results
                            model_results['Validation_Set'].append(corpus_title[i])
                            model_results['Topics'].append(k)
                            model_results['Alpha'].append(a)
                            model_results['Beta'].append(b)
                            model_results['Coherence'].append(cv)
                            
                            pbar.update(1)
            df_result = pd.DataFrame(model_results)
            pbar.close()

        df_result['Alpha'] = df_result['Alpha'].astype(str)
        df_result['Beta'] = df_result['Beta'].astype(str)
        
        if path is not None:
            df_result.to_csv(path, index=False)

        return df_result


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
