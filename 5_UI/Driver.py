from sys import path as syspath
from os import path as osPath, getcwd
import pandas as pd
import gensim
import numpy as np
from tweepy import API, OAuthHandler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\1_Data_Collection')
# noinspection PyUnresolvedReferences
from tweets_TimeLine_Collector import TweetTimeLineCollector
# noinspection PyUnresolvedReferences
from user_id_collector import UserIDCollector

syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\4_Model')

# noinspection PyUnresolvedReferences
from Hawkes_Process import get_user_timeline_hawkes
# noinspection PyUnresolvedReferences
from LDA import get_LDA_trained_Models,LstmClassifierModel,LDAClassifierModel
# noinspection PyUnresolvedReferences
from all_models_for_api import get_all_models, get_all_predictions
# noinspection PyUnresolvedReferences
from Combined_Models import CombinedClassifierModel
# noinspection PyUnresolvedReferences
from Temporal_models import TemporalClassifierModel
syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\2_Cleaning_Visualization')
# noinspection PyUnresolvedReferences
from text_cleaning import text_clean

uidc = UserIDCollector(" ")

API_KEY = "YOUR_API_KEY"
API_SECRETE_KEY = "YOUR_API_SECRETE_KEY"
CONSUMER_API_KEY = "YOUR_CONSUMER_API_KEY"
CONSUMER_API_SECRETE_KEY = "YOUR_CONSUMER_API_SECRETE_KEY"

auth = OAuthHandler(API_KEY, API_SECRETE_KEY)
auth.set_access_token(CONSUMER_API_KEY, CONSUMER_API_SECRETE_KEY)
api = API(auth)

LDA = None
Dict = None
Corpus = None
tweets_data = None
tokenized = None
df = None
columns = ['tweet_id', 'tweet_text', 'user_id',
           'name', 'screen_name', 'location',
           'description', 'url', 'is_protected',
           'followers_count', 'friends_count', 'created_at',
           'favourites_count', 'verified', 'statuses_count']


def get_all_data():
    global LDA, Dict, Corpus, tokenized, tweets_data,df

    _, _, _, _, Dict, Corpus, LDA = get_LDA_trained_Models()

    dataset_path = "C:/Users/Sampad/Desktop/Projects/Capstone/Implimentation/Code/0_DataSet/"
    df = pd.read_csv(dataset_path + "CompleteAnnotated.csv")
    df.tweet_text = df.tweet_text.apply(text_clean)

    tweets_data = df.tweet_text


def get_topic_vector(tweet_text, num_topics, Dict, LDA):
    d2b = Dict.doc2bow(tweet_text)

    topic_vector_sparse = LDA.get_document_topics(d2b)

    topic_vector = np.zeros(num_topics)

    for pair in topic_vector_sparse:
        topic_vector[pair[0]] = pair[1]

    return topic_vector


def extract_data_from_url(url):
    user_df = pd.DataFrame(columns=columns)
    tweet_id = url.split('/')[-1]
    tweet = api.get_status(tweet_id, tweet_mode="extended")
    data = uidc.getUserData(tweet)
    user_df.loc[len(user_df.index)] = data.values()
    ttl = TweetTimeLineCollector([data['user_id']])
    tweets_timelines = ttl.TweetsDataFrame(data['user_id'], ttl.getAllTweetsData(data['user_id'])[1])

    return user_df, tweets_timelines


def preprocessed_data(user_df, timeline_df):
    user_df.tweet_text = user_df.tweet_text.apply(text_clean)
    bl, adj = get_user_timeline_hawkes(timeline_df)
    return bl, adj


def getRetrainedLDAandDict(new_text, corpus, dict, LDA):
    new_text = new_text.split()
    dict.add_documents([new_text])
    corpus.append(dict.doc2bow(new_text))
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=10)
    return dict, lda


# old_tweets_data = df.tweet_text
def getUpdatedBOW(old_tweets_text, new_text):
    print(new_text)
    if new_text == None:
        word_size = 7000
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(old_tweets_text)
        tokenized = tokenizer.texts_to_sequences(old_tweets_text)
        return tokenized
    word_size = 7000
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(old_tweets_text + new_text)
    tokenized = tokenizer.texts_to_sequences(old_tweets_text + new_text)
    return tokenized


def prepare_data_to_models(user_df, tweets_df):
    global LDA, Corpus, Dict, tokenized, tweets_data
    baseline, adj = preprocessed_data(user_df, tweets_df)
    tweet_text = user_df.tweet_text.item()

    tokenized = getUpdatedBOW(tweets_data, tweet_text)

    sequence_size = 18
    padded = pad_sequences(tokenized, maxlen=sequence_size, padding='pre', truncating='pre')

    Dict, LDA = getRetrainedLDAandDict(tweet_text, Corpus, Dict, LDA)

    tempo_only_data = np.array([baseline, adj])
    tempo_only_data = tempo_only_data.reshape(1, -1)
    text_LDA_Topic_vec = get_topic_vector(tweet_text.split(), 10, Dict, LDA)
    text_LDA_Topic_vec = text_LDA_Topic_vec.reshape(1, -1)

    text_BOW_vec = padded[-1]
    text_BOW_vec = text_BOW_vec.reshape(1, 1, -1)
    combined_topic_vecs = get_topic_vector(tweet_text.split(), 12, Dict, LDA)
    combined_topic_vecs[-1] = baseline
    combined_topic_vecs[-2] = adj
    combined_topic_vecs = combined_topic_vecs.reshape(1, -1)

    datas = [tempo_only_data, tempo_only_data, tempo_only_data, tempo_only_data,
             text_LDA_Topic_vec, text_LDA_Topic_vec, text_LDA_Topic_vec, text_LDA_Topic_vec,
             text_BOW_vec, text_BOW_vec, combined_topic_vecs, combined_topic_vecs,
             combined_topic_vecs, combined_topic_vecs]
    return datas


if __name__ == '__main__':
    get_all_data()
    user_df, tweets_df = extract_data_from_url("https://twitter.com/MollyJongFast/status/1449749185276878852")
    datas = prepare_data_to_models(user_df, tweets_df)
    models = get_all_models()

    preds = get_all_predictions(datas, models)

    print(preds)
