# Import statements
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import string
import re
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud
import nltk.tokenize as tk
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.models import Phrases
from django.shortcuts import render, redirect, get_object_or_404
from bertopic import BERTopic
from .forms import VideoForm
from .models import Video, VideoResult
from django.http import HttpResponseServerError
import numpy as np

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Text Preprocessing
stop_words = set(stopwords.words('english'))
new_stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'us', 'think', 'know', 'right', 'like', 'well', 'lot', 'also', 'going', 'one', 'actually', 'really', 'see', 'many', 'feel', 'okay', 'back', 'thats', 'yeah', 'dont', 'want', 'go', 'theres', 'today', 'mean', 'even', 'youth', 'say', 'im', 'would', 'bit', 'terms', 'sort', 'youre', 'thank', 'among', 'take']
stop_words.update(new_stop)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocessing(corpus):
        tokens = word_tokenize(corpus)
        tokens = re.findall(r'\b\w+\b', corpus.lower())
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
        tokens = [token for token in tokens if token not in stop_words and not token.isnumeric()]
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if len(token) > 4]
        # tokens = [stemmer.stem(token) for token in tokens]
        
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

def create_bag_of_words(df, cleaned_corpus):
    # Create a defaultdict to store the word frequency counts
    word_counts = defaultdict(int)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
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

def calculate_tfidf(df, cleaned_corpus):
    # Create a list of documents (preprocessed texts) from the 'cleaned_corpus' column
    documents = df[cleaned_corpus].tolist()

    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Calculate TF-IDF values
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    return tfidf_df

def gen_words(texts):
        final = []
        for text in texts:
            new = gensim.utils.simple_preprocess(text, deacc=True)
            final.append(new)
        return(final)

from gensim.models import Phrases

def bigrams(texts):
    bigram_phrases = Phrases(texts, min_count=5, threshold=100)
    trigram_phrases = Phrases(bigram_phrases[texts], threshold=100)

    def make_bigrams(texts):
        return [bigram_phrases[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_phrases[bigram_phrases[doc]] for doc in texts]

    data_bigrams = make_bigrams(texts)
    data_bigrams_trigrams = make_trigrams(data_bigrams)
    
    return data_bigrams_trigrams

# VISUALISATION CHARTS -------------------------------------------------------------------

def add_columns(df):
        df['cleaned_text'] = ''
        df['POS'] = ''
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

def plot_bar_chart(main_topic, texts):
    topics = [topic for topic, _ in main_topic]
    scores = [score for _, score in main_topic]

    # Create a bar chart
    plt.bar(topics, scores)
    plt.xlabel('Topics')
    plt.ylabel('Scores')
    plt.title('Main Topics and Scores')
    plt.xticks(rotation=45)

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the buffer to base64 string
    chart_data = base64.b64encode(buffer.read()).decode()

    return chart_data

def plot_word_cloud(texts):
    text_combined = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text_combined)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the buffer to base64 string
    chart_data = base64.b64encode(buffer.read()).decode()

    return chart_data

def plot_pie_chart(main_topic):
    topics = [topic for topic, _ in main_topic]
    scores = [score for _, score in main_topic]

    plt.figure()
    plt.pie(scores, labels=topics, autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title('Main Topic Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the buffer to base64 string
    chart_data = base64.b64encode(buffer.read()).decode()

    return chart_data

# VIDEO RECORDS -------------------------------------------------------------------
def home(request):
    return render(request, "home.html")

def dashboard(request):
    return render(request, "dashboard.html")

def create_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('view_video')
    else:
        form = VideoForm()
    return render(request, 'create_video.html', {'form': form})

def view_video(request):
    videos = Video.objects.all()
    return render(request, 'view_video.html', {'videos': videos})

def update_video(request, pk):
    video = get_object_or_404(Video, pk=pk)
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES, instance=video)
        if form.is_valid():
            form.save()
            return redirect('view_video')
    else:
        form = VideoForm(instance=video)
    return render(request, 'update_video.html', {'form': form})

def delete_video(request, pk):
    video = get_object_or_404(Video, pk=pk)
    if request.method == 'POST':
        video.delete()
        return redirect('view_video')
    return render(request, 'delete_video.html', {'video': video})

# OLD VIDEO_DETAILS VIEW WITHOUT SEGMENTATION BUT WITH BAG OF WORDS AND TF_IDF
# def video_details(request, pk):
#     video = get_object_or_404(Video, pk=pk)

#     try:
#         video_result = VideoResult.objects.filter(video=video).first()

#         if video_result:
#             # If the results exist, fetch them from the database
#             main_topic = eval(video_result.main_topic)  # Convert JSON-like string to Python list
#             texts = eval(video_result.texts)  # Convert JSON-like string to Python list
#             chart_data = video_result.chart_data
#             word_cloud_data = video_result.word_cloud_data
#             pie_chart_data = video_result.pie_chart_data
#             # Fetch bag of words (bow_data) and tf-idf (tfidf_data) if needed
#         else:
#             df = pd.read_csv(video.VideoTranscript.path)
#             df = add_columns(df)
#             df['cleaned_text'] = df['Text'].apply(preprocessing)
#             df['POS'] = df['cleaned_text'].apply(tagPOS)
#             topic_model = BERTopic()
#             topic_ids, _ = topic_model.fit_transform(df['POS'])

#             main_topic_texts = {}
#             for text, topic_id in zip(df['Text'], topic_ids):
#                 main_topic = topic_model.get_topic(topic_id)
#                 main_topic_tuple = tuple(main_topic)
#                 if main_topic_tuple not in main_topic_texts:
#                     main_topic_texts[main_topic_tuple] = []
#                 main_topic_texts[main_topic_tuple].append(text)

#             results = []
#             for main_topic, texts in main_topic_texts.items():
#                 chart_data = plot_bar_chart(main_topic, texts)
#                 word_cloud_data = plot_word_cloud(texts)
#                 pie_chart_data = plot_pie_chart(main_topic)
#                 # bow_data = create_bag_of_words(df, 'cleaned_text')
#                 # tfidf_data = calculate_tfidf(df, 'cleaned_text')
#                 results.append((main_topic, texts, chart_data, word_cloud_data, pie_chart_data))

#             video_result = VideoResult.objects.create(
#                 video=video,
#                 main_topic=str(main_topic),  # Convert Python list to JSON-like string
#                 texts=str(texts),  # Convert Python list to JSON-like string
#                 chart_data=chart_data,
#                 word_cloud_data=word_cloud_data,
#                 pie_chart_data=pie_chart_data,
#                 # Save bag of words (bow_data) and tf-idf (tfidf_data) if needed
#             )

#         results = [(main_topic, texts, chart_data, word_cloud_data, pie_chart_data)]

#         return render(request, 'video_details.html', {'video': video, 'results': results})

#     except TypeError as e:
#         if "'numpy.float64' object cannot be interpreted as an integer" in str(e):
#             # Handle the specific TypeError here
#             error_message = "Model is busy, please try again later."
#             return HttpResponseServerError(error_message)
#         else:
#             # If it's a different TypeError, re-raise the exception
#             raise

# CURRENT VIDEO_DETAILS VIEW
def video_details(request, pk):
    video = get_object_or_404(Video, pk=pk)

    try:
        video_result = VideoResult.objects.filter(video=video).first()

        if video_result:
            # If the topic modelling results exist, retrieve them from the database
            main_topic = eval(video_result.main_topic)
            texts = eval(video_result.texts)
            chart_data = video_result.chart_data
            word_cloud_data = video_result.word_cloud_data
            pie_chart_data = video_result.pie_chart_data
            segmented_transcript = eval(video_result.segmented_transcript)

        else:
            # Generate topic modelling results
            df = pd.read_csv(video.VideoTranscript.path)
            df = add_columns(df)
            df['cleaned_text'] = df['Text'].apply(preprocessing)
            df['POS'] = df['cleaned_text'].apply(tagPOS)
            topic_model = BERTopic()
            topic_ids, _ = topic_model.fit_transform(df['cleaned_text'])
            segmented_transcript = segment_topics(df)
            print(segmented_transcript)
            main_topic_texts = {}
            for text, topic_id in zip(df['Text'], topic_ids):
                main_topic = topic_model.get_topic(topic_id)
                main_topic_tuple = tuple(main_topic)
                if main_topic_tuple not in main_topic_texts:
                    main_topic_texts[main_topic_tuple] = []
                main_topic_texts[main_topic_tuple].append(text)

            results = []
            for main_topic, texts in main_topic_texts.items():
                chart_data = plot_bar_chart(main_topic, texts)
                word_cloud_data = plot_word_cloud(texts)
                pie_chart_data = plot_pie_chart(main_topic)

                

                results.append((main_topic, texts, chart_data, word_cloud_data, pie_chart_data, segmented_transcript))

            # Save to database
            video_result = VideoResult.objects.create(
                video=video,
                main_topic=str(main_topic),
                texts=str(texts),
                chart_data=chart_data,
                word_cloud_data=word_cloud_data,
                pie_chart_data=pie_chart_data,
                segmented_transcript=str(segmented_transcript)
            )
        
        results = [(main_topic, texts, chart_data, word_cloud_data, pie_chart_data, segmented_transcript)]
        
        # Render the results and pass them to the template
        
        return render(request, 'video_details.html', {'video': video, 'results': results})

    except TypeError as e:
        if "'numpy.float64' object cannot be interpreted as an integer" in str(e):
            # Handle this specific TypeError as it occurs when BERTopic model is busy
            error_message = "Model is busy, please try again later."
            return HttpResponseServerError(error_message)
        else:
            # If it's a different TypeError, re-raise the exception
            raise

def video_cards(request):
    videos = Video.objects.all()

    for video in videos:
        try:
            video_result = VideoResult.objects.get(video=video)
            main_topics = eval(video_result.main_topic)
            video.main_topics = main_topics
        except VideoResult.DoesNotExist:
            video.main_topics = []

    return render(request, 'video_cards.html', {'videos': videos})

#Segmentation of transcript
def segment_topics(df):
        # Initialize variables
        segments = []
        output_messages = []
        current_segment = {'Speaker': '', 'Text': ''}
        topic_model = BERTopic()
        # Iterate over the rows
        for index, row in df.iterrows():
            # Get the segment value
            segment = row['Segment']
            
            # Check if it's a new segment
            if segment == 'Question':
                # Add the previous segment to the list
                if current_segment['Text']:
                    segments.append(current_segment)
                
                # Start a new segment
                current_segment = {'Speaker': row['Speaker'], 'Text': row['Text']}
            else:
                # Append the text to the current segment
                current_segment['Text'] += ' ' + row['Text']

        # Add the last segment to the list
        if current_segment['Text']:
            segments.append(current_segment)
        
        
        for i, segment in enumerate(segments):
            # Fit BERTopic model for the current segment
            segment_text = segment['Text']
            segment_texts = [segment_text]  # You can also use a list of texts if there are multiple texts per segment
            # topics, _ = topic_model.fit_transform(segment_texts)
            output_message = {
                    "Segment" : i+1 ,
                    "Speaker": segment['Speaker'],
                    "Text" : segment['Text'],
                    # "Dominant_Topic": topics,
            }
        
            output_messages.append(output_message)
            
        return output_messages

def user_videodetails(request, video_id):
    video = get_object_or_404(Video, VideoID=video_id)
    df = pd.read_csv(video.VideoTranscript.path)
    transcripts = list(zip(df['TranscriptTime'], df['Text']))
    
    context = {'video': video, 'transcripts': transcripts}
    return render(request, 'user_videodetails.html', context)