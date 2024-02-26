import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import gensim
from gensim import corpora
from gensim.models import TfidfModel

import string
from string import punctuation
from pathlib import Path
from pprint import pprint

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import nltk.tokenize as tk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pyLDAvis
import pyLDAvis.gensim_models

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

from nltk.util import ngrams
from IPython.display import display
import webbrowser  

# from bertopic import BERTopic
# from umap import UMAP

from django.shortcuts import render
import base64
import io

stop_words = stopwords.words('english')
new_stop = ['know','think','like','also','really','well','would','right','us','one','actually','people','city','year','question','yeah','going',
            'lot','choose','feel','perhaps','gon','say','na']
stop_words.extend(new_stop)

    
class LDA():
            
    def add_columns(df):
        df['cleaned_corpus'] = ''
        df['POS_news'] = ''
        df['lemmed'] = ''
        df['cleaned_tokenized'] = ''
        return df

    def tokenizer(df):
        for i in df.index:
            tokenize = tk.word_tokenize(df['cleaned_corpus'][i])
            df['cleaned_tokenized'][i] = tokenize
        return df
    
    def lemma(df):
        for i in df.index:
            lemmatized = []
            for j in df['cleaned_tokenized'][i]:
                lemmatized.append(lemmatizer.lemmatize(j,pos = 'n'))
                df['lemmed'][i] = lemmatized
        return df
    
    def tfidf(df):
        list = []
        for i in df.index:
            list.append()

    def preprocessing(corpus):
            tokens = word_tokenize(corpus)

            tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            tokens = [token for token in tokens if not token.isnumeric()]
            tokens = [token for token in tokens if token not in string.punctuation]
            tokens = [token for token in tokens if token.isalnum()]
            tokens = [stemmer.stem(token) for token in tokens]
            
            cleaned_corpus = ' '.join(tokens)
            return cleaned_corpus
    

    def tagPOS(text):
        tokenized = sent_tokenize(text)
        tagged_text = []

        for i in tokenized: 
            
            wordsList = nltk.word_tokenize(i)
            # removing stop words from wordList 
            wordsList = [w for w in wordsList if not w in stop_words]  

            #  Using a Tagger. Which is part-of-speech tagger or POS-tagger.  
            tagged = nltk.pos_tag(wordsList) 
            filtered = [word[0] for word in tagged if word[1] in ['NN']]
            tagged_text.extend(filtered)
        
        combined_words = ' '.join(tagged_text)
        return combined_words  
        
        
    # def tagger():
    #     def findtags(tag_prefix, tagged_text, n):
    #         cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
    #                                     if tag.startswith(tag_prefix))
    #         return dict((tag, cfd[tag].most_common(n)) for tag in cfd.conditions())

    #     # find the top 5 adjective in the first news
    #     tagged_text = df['POS_News'][0]
    #     tagdict = findtags('NN', tagged_text, 5)
    #     for tag in sorted(tagdict):
    #         print(tag, tagdict[tag])
        
    def gen_words(texts):
        final = []
        for text in texts:
            new = gensim.utils.simple_preprocess(text, deacc=True)
            final.append(new)
        return(final)

    def bigrams(text):
        bigrams_phrases = gensim.models.Phrases(text, min_count=5, threshold=100)
        trigram_phrases = gensim.models.Phrases(bigrams_phrases[text], threshold=100)

        bigram = gensim.models.phrases.Phraser(bigrams_phrases)
        trigram = gensim.models.phrases.Phraser(trigram_phrases)

        def make_bigrams(texts):
            return(bigram[texts])

        def make_trigrams(texts):
            return(trigram[bigram[texts]])

        data_bigrams = make_bigrams(text)
        data_bigrams_trigrams = make_trigrams(data_bigrams)
        
        return data_bigrams_trigrams
    
            
    def fitting(data_bigrams_trigrams):
    
        id2word = corpora.Dictionary(data_bigrams_trigrams)
        texts = data_bigrams_trigrams

        corpus = [id2word.doc2bow(text) for text in texts]
        
        return id2word, corpus
    
    def modelling(corpus, id2word):
        topic_num = 5
        word_num = 5
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(corpus, num_topics = topic_num, id2word = id2word, passes=20, random_state=42)
        
        pprint(ldamodel.print_topics(num_topics=topic_num, num_words=word_num))
        return ldamodel

    def wordcloud(df):
        word_list = []
        for i in df.index:
            for j in df["lemmed"][i]:
                word_list.append(j)
        word_list

        freq_dist = nltk.FreqDist(word_list)
        # sorted_freqdist = sorted(freq_dist, key = freq_dist.__getitem__, reverse=True)

        # plt.figure()
        plt.ioff()
        wcloud = WordCloud().generate_from_frequencies(freq_dist)
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('topicmodel\static\images\cleaned_cloud.png')
        # plt.show()
    
    def freq_dist(df):
        word_list = []
        for i in df.index:
            for j in df["lemmed"][i]:
                word_list.append(j)
        word_list

        freq_dist = nltk.FreqDist(word_list)
        
        fig = plt.figure(figsize = (10,4))
        plt.gcf().subplots_adjust(bottom=0.15) # to avoid x-ticks cut-off
        words = dict([(k,v) for k,v in freq_dist.items() if len(k)>3])
        filtered_freq_dist = nltk.FreqDist(words)
        filtered_freq_dist.plot(20,cumulative=False) 
        plt.show() 
        fig.savefig('topicmodel\\static\\images\\freq_dist.png', bbox_inches = 'tight') 
           
        
    def perplexity(corpus, id2word):
        topic_counter = 15
        topic_num = 5

        stats_df = pd.DataFrame()

        for item in np.arange(1, topic_counter,2):
            passes = item
            Lda = gensim.models.ldamodel.LdaModel
            ldamodel = Lda(corpus, num_topics = topic_num, id2word = id2word, passes=passes, random_state=42) 

            per_var = ldamodel.log_perplexity(corpus)
            temp_df = pd.DataFrame([[item,per_var]],columns = ['Topic_num','Perplexity'])
            stats_df = pd.concat([stats_df, temp_df])
        
        plt.figure()
        plt.plot(stats_df.Topic_num, stats_df.Perplexity)
        plt.xlabel('Number of Passes')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs. Number of Passes')
        plt.savefig('topicmodel\static\images\Perplexity.png')
        plt.show()
            
    def plotting(ldamodel, corpus, id2word):
        vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, id2word)
        pyLDAvis.save_html(vis, 'lda_visualization.html')
        webbrowser.open('lda_visualization.html')
        return vis


    def segment(df,corpus,id2word):
        # Initialize variables
        segments = []
        current_segment = {'Speaker': '', 'cleaned_corpus': ''}
        
        # Iterate over the rows
        for index, row in df.iterrows():
            # Get the segment value
            segment = row['Segment']
            
            # Check if it's a new segment
            if segment == 'Question':
                # Add the previous segment to the list
                if current_segment['cleaned_corpus']:
                    segments.append(current_segment)
                
                # Start a new segment
                current_segment = {'Speaker': row['Speaker'], 'cleaned_corpus': row['cleaned_corpus']}
            else:
                # Append the text to the current segment
                current_segment['cleaned_corpus'] += ' ' + row['cleaned_corpus']

        # Add the last segment to the list
        if current_segment['cleaned_corpus']:
            segments.append(current_segment)

        # Apply LDA model using Gensim
        num_topics = 5  # Define the number of topics
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=20,random_state=42)

        for i, segment in enumerate(segments):
            topic_dist = lda[corpus[i]]
            dominant_topic_idx = max(topic_dist, key=lambda item: item[1])[0]
            dominant_topic_prob = max(topic_dist, key=lambda item: item[1])[1]
            top_words = [word for word, _ in lda.show_topic(dominant_topic_idx, topn=5)]
            
            print(f"Segment {i+1}:")
            print(f"Speaker: {segment['Speaker']}")
            print(f"Text: {segment['cleaned_corpus']}")
            print(f"Dominant Topic: {dominant_topic_idx}")
            print(f"Dominant Topic Probability: {dominant_topic_prob:.4f}")
            print(f"Top Words in Dominant Topic:")
            print(top_words)
            print()
            
    def main(request):
        df = pd.read_csv('C:\\Users\\Zude Ang\\VSC\\IND_TopicModellingOV-1\\Geopolitics.csv')
        df = LDA.add_columns(df)
        df['cleaned_corpus'] = df['Text'].apply(LDA.preprocessing)
        df['POS_news'] = df['cleaned_corpus'].apply(LDA.tagPOS) #filters out all words other than NN nouns
        df = LDA.tokenizer(df)
        df = LDA.lemma(df)
        cleaned = df['cleaned_corpus']
        # print(df['POS_news'])
        wordcloud = LDA.wordcloud(df)
        freq_dist = LDA.freq_dist(df)
        data_words = LDA.gen_words(cleaned)
        data_bigrams_trigrams = LDA.bigrams(data_words)
        id2word, corpus = LDA.fitting(data_bigrams_trigrams)
        ldamodel = LDA.modelling(corpus, id2word)
        segments = LDA.segment(df,corpus,id2word)
        # BERT = LDA.BERT(df)
        # print(ldamodel)
        # perplex = LDA.perplexity(corpus, id2word)
        # print(perplex)
        # ldaplot = LDA.plotting(ldamodel, corpus, id2word)
        # print(ldaplot)
        
        return render(request, 'topicmodel/index.html',{'wordcloud': wordcloud,'freq_dist':freq_dist})
            

    # def BERT(df):
    #     umap_model = UMAP(n_neighbors=15, n_components=5, 
    #                       min_dist=0.0, metric= 'cosine',
    #                       random_state=100)
    #     topic_model = BERTopic(umap_model=umap_model,top_n_words= 4, language='english',calculate_probabilities=True)
    #     topics, probabilities = topic_model.fit_transform(df['POS_news'])

    #     print(topic_model.get_topic_info())
    

