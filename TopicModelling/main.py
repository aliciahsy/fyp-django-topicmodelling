from Modelling import LDA

import pandas as pd

def main():
    df = pd.read_csv('C:\\Users\\Zude Ang\\VSC\\IND_TopicModellingOV-1\\Geopolitics.csv')
    df = LDA.add_columns(df)
    df['cleaned_corpus'] = df['Text'].apply(LDA.preprocessing)
    df['POS_news'] = df['cleaned_corpus'].apply(LDA.tagPOS) #filters out all words other than NN nouns
    df = LDA.tokenizer(df)
    df = LDA.lemma(df)
    NER = LDA.NER(df['cleaned_corpus'])
    cleaned = df['cleaned_corpus']
    df['NER'] = df['cleaned_corpus'].apply(LDA.NER_x)
    nn = df['NER'].tolist()
    print(nn)
    # boW = LDA.create_bag_of_words(df, cleaned_corpus = 'cleaned_corpus')
    # tfidf = LDA.calculate_tfidf(df, 'cleaned_corpus')
    # print(tfidf)
    # LDA.generate_bar_chart(boW)
    
# Visualisations
    # wordcloud = LDA.wordcloud(df)
    # freq_dist = LDA.freq_dist(df)
    
#Fittings
    # data_words = LDA.gen_words(cleaned)
    # data_bigrams_trigrams = LDA.bigrams(nn)
    # id2word, corpus = LDA.fitting(data_bigrams_trigrams)
    
#LDA model
    # ldamodel = LDA.modelling(corpus, id2word)
    # segments = LDA.segment(df,corpus,id2word)
    # print(ldamodel)
    # perplex = LDA.perplexity(corpus, id2word)
    # print(perplex)
    # ldaplot = LDA.plotting(ldamodel, corpus, id2word)
    # print(ldaplot)
    
    # # BERT = LDA.BERT(df)
main()  

        