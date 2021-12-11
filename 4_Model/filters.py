import pandas as pd
import os
import numpy as  np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sys import path as syspath
from os import path as osPath, getcwd

syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\2_Cleaning_Visualization')

# noinspection PyUnresolvedReferences
from text_cleaning import text_clean


data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '\\0_DataSet'
user_file = data_path + '\\Users.csv'
userWiseDataFolder = os.path.normpath(data_path + '\\UserWiseData')
userWiseDataFiles = next(os.walk(userWiseDataFolder), (None, None, []))[2]

users_df = pd.read_csv(user_file)




def get_number_of_users():
    return len(userWiseDataFiles)


def dateTimeCreator(s):
    s = s.split(' ')
    year, month, day = s[0].split('-')
    hours, minutes, seconds = s[1].split(':')
    return datetime(int(year), int(month), int(day), int(hours), int(minutes), int(seconds))

def dateTimeCreator_new(s):
    s = s.split(' ')
    day, month, year = s[0].split('-')
    hours, minutes, seconds = s[1].split(':')
    return datetime(int(year), int(month), int(day), int(hours), int(minutes), int(seconds))

def get_retweet_df(fileNumber):
    fileName = userWiseDataFiles[fileNumber]
    file_path = userWiseDataFolder + '\\' + fileName
    tweet_df = pd.read_csv(file_path)
    tweet_df.drop(tweet_df[tweet_df['tweet_text'].str[:2] == 'RT'].index, inplace=False)
    tweet_df = tweet_df.sort_values(by='created_at')
    return tweet_df


def get_timestamp(tweet_creation_time, origin_date):
    # print tweet_creation_time,"is creation time"

    tDelta = tweet_creation_time - origin_date
    return tDelta.seconds / 60

    # attbs = tweet_creation_time.split(' ')
    # print(attbs)
    # date = int(attbs[2])
    # hhmmss = attbs[3].split(':')
    # hhmmss = [int(val) for val in hhmmss]
    # # print date,hhmmss
    # minutes_passed = (date - origin_date) * 24 * 60 + (hhmmss[0]) * 60 + hhmmss[1] + hhmmss[2] / 60.0
    # return minutes_passed


def train_test_splitter(X, Y):
    return train_test_split(X, Y, test_size=0.20, random_state=42)


def generate_topic_vector(tweet_text, num_topics, dict_genuine, dict_fake, lda_genuine, lda_fake):

    xg = dict_genuine.doc2bow(tweet_text)
    xf = dict_fake.doc2bow(tweet_text)

    topic_vector_sparse_genuine = lda_genuine.get_document_topics(xg)
    topic_vector_sparse_fake = lda_fake.get_document_topics(xf)


    topic_vector = np.zeros(2 * num_topics + 2)

    for pair in topic_vector_sparse_genuine:
        topic_vector[pair[0]] = pair[1]

    for pair in topic_vector_sparse_fake:
        topic_vector[num_topics + pair[0]] = pair[1]

    return topic_vector


def get_users_dataframe():
    df =  pd.read_csv(data_path+'\\CompleteAnnotated.csv')
    df.tweet_text = df.tweet_text.apply(text_clean)
    return df


def get_topic_vector_for_unseen_data(text, dict_genuine, dict_fake, lda_genuine, lda_fake):
    xg = dict_genuine.doc2bow(text)
    xf = dict_fake.doc2bow(text)

    topic_vector = np.zeros(2 * num_topics)

    for pair in lda_genuine[xg]:
        topic_vector[pair[0]] = pair[1]


    for pair in lda_fake[xf]:
        topic_vector[num_topics + pair[0]] = pair[1]

    return topic_vector