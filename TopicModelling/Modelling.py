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
from collections import defaultdict

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
from sklearn.feature_extraction.text import CountVectorizer

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
from pprint import pprint
from nltk import ne_chunk
nltk.download('words')
nltk.download('maxent_ne_chunker')

from nltk.util import ngrams
from IPython.display import display
import webbrowser  

# from bertopic import BERTopic
# from umap import UMAP

from django.shortcuts import render

stop_words = stopwords.words('english')
new_stop = ['know','think','like','also','really','well','would','right','us','one','actually','people','city',
            'year','question','yeah','going','lot','choose','feel','perhaps','gon','say','na','um','let','makes',
            'slip','meaning','sometimes','based','becomes','thinks','whatever','strong','wtf','works']
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
    
    def create_bag_of_words(df, cleaned_corpus):
        # Create a defaultdict to store the word frequency counts
        word_counts = defaultdict(int)

        # Iterate through each row in the DataFrame
        for index , row in df.iterrows():
            # Get the preprocessed text from the specified column
            preprocessed_text = row[cleaned_corpus]
            
            # Split the preprocessed_text into words
            words = preprocessed_text.split()
            
            # Update the word_counts with the frequency of each word in the row
            for word in words:
                word_counts[word] += 1

        # Convert the word_counts dictionary to a DataFrame
        bag_of_words_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Frequency'])
        
        return bag_of_words_df
    
    def generate_bar_chart(bow_df):
        # Sort the DataFrame in descending order based on the 'Frequency' column
        sorted_bow_df = bow_df.sort_values(by='Frequency', ascending=False)

        # Take the top 5 words and their frequencies
        top_5_words = sorted_bow_df.head(5)

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(top_5_words['Word'], top_5_words['Frequency'])
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 5 Words and Their Frequencies')
        plt.xticks(rotation=45)
        plt.show()

    # def build_bow_features(df, cleaned_corpus):
    # # Build the dictionary
    #     mydict = corpora.Dictionary(df[cleaned_corpus])
    #     vocab_len = len(mydict)

    #     def get_bow_features(df, cleaned_corpus, mydict, vocab_len):
    #         test_features = []
    #         for index, row in df.iterrows():
    #             # Converting the tokens into the format that the model requires
    #             features = gensim.matutils.corpus2csc([mydict.doc2bow(row[cleaned_corpus])], num_terms=vocab_len).toarray()[:, 0]
    #             test_features.append(features)
    #         return test_features

    #     header = ",".join(str(mydict[ele]) for ele in range(vocab_len))

    #     bow_features = pd.DataFrame(get_bow_features(df, cleaned_corpus, mydict, vocab_len),
    #                 columns=header.split(','), index=df.index)
    
    #     return bow_features
    

    def calculate_tfidf(df, cleaned_corpus):
        # Create a list of documents (preprocessed texts) from the 'cleaned_corpus' column
        documents = df[cleaned_corpus].tolist()

        # Initialize the TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Calculate TF-IDF values
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        # Convert the TF-IDF matrix to a DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        
        # Calculate mean TF-IDF scores for each word
        mean_tfidf_scores = tfidf_df.mean(axis=0)

        # Sort the DataFrame based on the mean TF-IDF scores in ascending order
        sorted_tfidf_df = mean_tfidf_scores.sort_values()

        # Extract the top 20 words with the lowest TF-IDF scores
        top_20_lowest_tfidf_words = sorted_tfidf_df.head(50)    

        return top_20_lowest_tfidf_words
    
    def preprocessing(corpus):
            tokens = word_tokenize(corpus)

            tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalnum()]   
            tokens = [token for token in tokens if not re.match(r'^\d+(st|nd|rd|th|g|s)?$', token)]     
            # tokens = [stemmer.stem(token) for token in tokens]
            tokens = [token for token in tokens if len(token) > 3]
            
            cleaned_corpus = ' '.join(tokens)
            return cleaned_corpus
    
    def tagPOS(text):
        tokenized = sent_tokenize(text) 
        tagged_text = []

        for i in tokenized: 
            # Word tokenizers is used to find the words  
            # and punctuation in a string 
            wordsList = nltk.word_tokenize(i) 

            # removing stop words from wordList 
            wordsList = [w for w in wordsList if not w in stop_words]  

            #  Using a Tagger. Which is part-of-speech tagger or POS-tagger.  
            tagged = nltk.pos_tag(wordsList) 
            tagged_text.extend(tagged)

        return tagged_text
        
    def NER(text):
        combined_text = ' '.join(text)
        doc = nlp(combined_text)
        named_entities = [X.text for X in doc.ents]
        return named_entities
    
    def NER_x(text):
        doc = nlp(text)
        tokens_with_ner = []
        current_ner = []

        for token in doc:
            if token.ent_type_:
                current_ner.append(token.text)
            else:
                if current_ner:
                    tokens_with_ner.append(' '.join(current_ner))
                    current_ner = []
                tokens_with_ner.append(token.text)

        if current_ner:
            tokens_with_ner.append(' '.join(current_ner))

        return tokens_with_ner
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
        
    # def gen_words(texts):
    #     final = []
    #     for text in texts:
    #         new = gensim.utils.simple_preprocess(text, deacc=True)
    #         final.append(new)
    #     return(final)

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

        plt.figure()
        wcloud = WordCloud().generate_from_frequencies(freq_dist)
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('cleaned_cloud.png')
        plt.show()     
    
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
        fig.savefig('freq_dist.png', bbox_inches = 'tight') 
           
        
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
        plt.xlabel('Number of Topics')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs. Number of Topics')
        plt.savefig('Perplexity.png')
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
        tfidf_vectorizer = TfidfVectorizer()  # Initialize the TF-IDF vectorizer
    
    # Split the TF-IDF representation based on the segment value
        tfidf_segments = tfidf_vectorizer.fit_transform(df['cleaned_corpus'])
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
            segment_tfidf = tfidf_segments[i]
            top_n_words = 5
            top_tfidf_indices = segment_tfidf.toarray().argsort()[0][-top_n_words:][::-1]
            top_words_tfidf = [tfidf_vectorizer.get_feature_names_out()[idx] for idx in top_tfidf_indices]
            
            # segment_bow = LDA.create_bag_of_words(df[df['Speaker'] == segment['Speaker']], 'cleaned_corpus')
            # segment_bow['Frequency'] = segment_bow['Frequency'].astype(int)
            # top_n_words = 5
            # top_words_bow = segment_bow.nlargest(top_n_words, 'Frequency')['Word'].tolist()       
            
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
            print(f"Top {top_n_words} Important Words (TF-IDF):")
            print(top_words_tfidf)
            # print(f"Top {top_n_words} Most Frequent Words (Bag of Words):")
            # print(top_words_bow)
            print()
            
            

    # def BERT(df):
    #     umap_model = UMAP(n_neighbors=15, n_components=5, 
    #                       min_dist=0.0, metric= 'cosine',
    #                       random_state=100)
    #     topic_model = BERTopic(umap_model=umap_model,top_n_words= 4, language='english',calculate_probabilities=True)
    #     topics, probabilities = topic_model.fit_transform(df['POS_news'])

    #     print(topic_model.get_topic_info())
    
