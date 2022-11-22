import re
import os
import json
import glob
import tqdm
import pandas as pd
import numpy as np
import plotly.express as px
import gensim
import spacy
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from gensim.parsing.preprocessing import STOPWORDS
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

tqdm.tqdm.pandas()

class Processor:

    def __init__(self, config_file_path):


        splitted_path = config_file_path.split('/')
        initial_path = "/".join(splitted_path[:-1])
        fintech_path = f"{initial_path}/fintech/"
        fintech_folders = glob.glob(f"{fintech_path}*/", recursive=True)
        self.fintech = [path.split('/')[-2] for path in fintech_folders]

        nbfc_path = f"{initial_path}/nbfc/"
        nbfc_folders = glob.glob(f"{nbfc_path}*/", recursive=True)
        self.nbfc = [path.split('/')[-2] for path in nbfc_folders]

        with open(config_file_path) as f:
            self.config_data = json.load(f)
        
        self._fintech_stopwords = []
        for comp in self.fintech:
            self._fintech_stopwords = self._fintech_stopwords + self.config_data["stopwords"]["fintech"][comp]
        
        self._nbfc_stopwords = []
        for comp in self.nbfc:
            self.config_data["stopwords"]["nbfc"][comp]
            self._nbfc_stopwords = self._nbfc_stopwords + self.config_data["stopwords"]["nbfc"][comp]

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


    def lemmatization(self, tokens):
        """Function to lemmatized tokenied sentence"""
        doc = nlp(" ".join(tokens))
        return " ".join([token.lemma_ for token in doc])

    def preprocess(self, df):

        # Drop duplicates
        df.drop_duplicates('tweet', inplace=True)

        # Select tweets with english language
        df = df[df['language'] == "en"]

        # Segregate the gate
        pre_df = df[df['date'] < '2020-02-01']
        post_df = df[df['date'] >= '2020-02-01']

        # Preprocess tweets
        pre_df.loc[:, 'preprocessed'] = pre_df['tweet'].apply(self.preprocess_tweet)
        post_df.loc[:, 'preprocessed'] = post_df['tweet'].apply(self.preprocess_tweet)

        # Remove empty texts
        pre_df = pre_df[pre_df['preprocessed'].map(bool)]
        post_df = post_df[post_df['preprocessed'].map(bool)]

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


    def create_topic_df(self, df, alpha, beta):
        topics_score = []
        for i in df['Topics'].value_counts().index:
            data = []
            data.append(i)
            temp = df[(df['Topics'] == i) & (df['Alpha'] == alpha) & (df['Beta'] == beta)]
            max_value = temp['Coherence'].max()
            data.append(max_value)
            topics_score.append(data)
        
        ts_df = pd.DataFrame(topics_score)
        ts_df.columns = ['topics', 'coherence']

        ts_df.sort_values('topics', inplace=True)

        return ts_df.copy()

    def dominant_topics(self, ldamodel, corpus, texts, tweets):
        sent_topics_df = pd.DataFrame()
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j==0:
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True
                    )
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        contents = pd.Series(tweets)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        sent_topics_df.rename(columns={0: "Text"}, inplace=True)
        return sent_topics_df.copy()

    def plot_topic_distribution(self, df):
        dist_df = pd.DataFrame(df['Dominant_Topic'].value_counts()).reset_index()
        dist_df.columns = ['Topic_Numbers', 'Document_Counts']

        fig = px.bar(dist_df, x='Topic_Numbers', y='Document_Counts', title='Topics Distribution')
        return dist_df, fig

    def word_distribution(self, text, find):
        text_tokens = word_tokenize(text)
        text = self.lemmatization(text_tokens)
        if find in text:
            return True
        else:
            return False
    
    def words_distribution(self, df, words):
        result = pd.DataFrame()
        rdf = pd.DataFrame()
        for word in tqdm.tqdm(words, position=1):
            rdf[word] = df['tweet'].progress_apply(self.word_distribution, find=word)
            temp = pd.DataFrame(rdf[word].value_counts()).reset_index()
            if len(temp) == 2:
                result = result.append(
                    pd.Series([word, temp.iloc[0][word], temp.iloc[1][word]]), ignore_index=True
                )
            else:
                if False in temp['index'].values.tolist():
                    result = result.append(
                        pd.Series([word, temp.iloc[0][word], 0]), ignore_index=True
                    )
                else:
                    result = result.append(
                        pd.Series([word, 0, temp.iloc[0][word]]), ignore_index=True
                    )
        result.columns=['words', 'false_cnt', 'true_cnt']
        result['false_percent'] = result['false_cnt'] * 100 / (result['false_cnt'] + result['true_cnt'])
        result['true_percent'] = result['true_cnt'] * 100 / (result['false_cnt'] + result['true_cnt'])
        return result
