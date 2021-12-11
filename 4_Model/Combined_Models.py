from Classifier import get_KNN_Model, get_accuracy_matric, get_lin_SVM_Model, get_NaiveBayes_Model
from Hawkes_Process import get_topic_vector, get_hawkes_model
from Topic_Modelling import LDA_main_driver
from filters import get_retweet_df, get_timestamp, get_number_of_users, dateTimeCreator, generate_topic_vector, \
    dateTimeCreator_new, train_test_splitter
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import torch
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sys import path as syspath
from os import path as osPath, getcwd
from os import listdir

syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\2_Cleaning_Visualization')

# noinspection PyUnresolvedReferences
from text_cleaning import text_clean

warnings.simplefilter(action='ignore')

x_train, x_test, y_train, y_test = None, None, None, None,
is_updated = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CombinedClassifierModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(CombinedClassifierModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.Linear(512, 256)

        self.layer6 = nn.Linear(256, 128)
        self.layer7 = nn.Linear(128, 64)
        self.layer8 = nn.Linear(64, 32)
        self.layer9 = nn.Linear(32, 16)
        self.layer10 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.layer1(inputs)
        out = self.relu(out)

        out = self.layer2(out)
        out = self.relu(out)

        out = self.layer3(out)
        out = self.relu(out)

        out = self.layer4(out)
        out = self.relu(out)

        out = self.layer5(out)
        out = self.relu(out)

        out = self.layer6(out)
        out = self.relu(out)

        out = self.layer7(out)
        out = self.relu(out)

        out = self.layer8(out)
        out = self.relu(out)

        out = self.layer9(out)
        out = self.relu(out)

        out = self.layer10(out)

        return out


def Validator(model, x_test, y_test):
    predicted = model(x_test).to(device)
    pred = torch.max(predicted.data, 1)[1]
    total_test = len(y_test)
    correct_pred = 0

    for i in range(total_test):
        if y_test[i] == pred[i]:
            correct_pred += 1

    return correct_pred / total_test


def get_topic_vector(tweet_text, num_topics, Dict, LDA):
    d2b = Dict.doc2bow(tweet_text)

    topic_vector_sparse = LDA.get_document_topics(d2b)

    topic_vector = np.zeros(num_topics+2)

    for pair in topic_vector_sparse:
        topic_vector[pair[0]] = pair[1]

    return topic_vector


def get_trainable_data():
    global x_train, x_test, y_train, y_test, is_updated
    df, dict_genuine, dict_fake, lda_genuine, lda_fake = LDA_main_driver()
    num_topics = 10
    user_topic_vectors, labels = get_topic_vector(df, dict_genuine, dict_fake, lda_genuine, lda_fake, 0)
    total_len = len(user_topic_vectors)
    X = np.array(user_topic_vectors)
    Y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_splitter(X, Y)
    is_updated = True


def get_trainable_data_2():
    global x_train, x_test, y_train, y_test, is_updated
    dataset_path = "C:/Users/Sampad/Desktop/Projects/Capstone/Implimentation/Code/0_DataSet/"

    df = pd.read_csv(dataset_path + "CompleteAnnotated.csv")

    df.tweet_text = df.tweet_text.apply(text_clean)
    complete_docs = []

    for i in range(df.shape[0]):
        complete_docs.append(df.iloc[i]['tweet_text'].split())

    complete_dict = Dictionary(complete_docs)
    complete_corpus = [complete_dict.doc2bow(text) for text in complete_docs]
    lda = LdaModel(complete_corpus, num_topics=10)

    num_topics = 10
    labels = []

    # LDA outputs
    user_topic_vectors = []
    print(
        '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Hawkes Process Started -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\n')
    for i in range(get_number_of_users()):
        # from 0_Datset folder get each user tweets timeline and filter retweets from them
        i_user_df = get_retweet_df(i)
        if i_user_df.shape[0] < 2:
            continue

        # String Date Time to python Datetime library
        i_user_df.created_at = i_user_df.created_at.apply(dateTimeCreator)

        # time at which first retweet was made
        min_date = min(i_user_df['created_at'])

        # time to minutes passed from the first retweet
        timestamps = i_user_df.created_at.apply(get_timestamp, origin_date=min_date).to_numpy()

        min_time = np.min(timestamps)
        max_time = np.max(timestamps)

        # Min-Max Scaler (0, 1)
        sorted_time = (np.sort(np.unique((timestamps - min_time) / max_time)))

        # fit the model and get the Hawkes Expression kernal model output
        BaseLine, adj_mat = get_hawkes_model(timestamps=[sorted_time])

        # append the results

        # baselines.append(BaseLine)
        if df[df.user_id == i_user_df['user_id'][0]]['Annotation'].item() == 1 or \
                df[df.user_id == i_user_df['user_id'][0]]['Tag'].item() == 1:
            labels.append(1)
        else:
            labels.append(0)
        # hawkes_user_ids.append(i_user_df['user_id'][0])
        # adjs.append(adj_mat)

        # get topic vector for tweet in users.csv file
        tweet_text_list = df[df.user_id == i_user_df['user_id'][0]]['tweet_text'].item().split()
        topic_vector = get_topic_vector(tweet_text_list, num_topics, complete_dict, lda)
        topic_vector[-1] = BaseLine
        topic_vector[-2] = adj_mat

        user_topic_vectors.append(topic_vector)

    print(
        '-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Hawkes Process Ended -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')

    labels = np.array(labels)
    user_topic_vectors = np.array(user_topic_vectors)

    x_train, x_test, y_train, y_test = train_test_splitter(user_topic_vectors, labels)
    is_updated = True
    return user_topic_vectors, labels


def get_knn_model():
    global x_train, x_test, y_train, y_test, is_updated

    if not is_updated:
        get_trainable_data()

    return get_KNN_Model(x_train, y_train)


def get_svm_model():
    global x_train, x_test, y_train, y_test, is_updated

    if not is_updated:
        get_trainable_data()

    return get_lin_SVM_Model(x_train, y_train)


def get_NB_model():
    global x_train, x_test, y_train, y_test, is_updated

    if not is_updated:
        get_trainable_data()

    return get_NaiveBayes_Model(x_train, y_train)


def get_nn_model():
    global x_train, x_test, y_train, y_test, is_updated

    if not is_updated:
        get_trainable_data()

    x_trn = torch.Tensor(x_train).to(device)
    y_trn = torch.Tensor(y_train).to(device)
    x_tst = torch.Tensor(x_test).to(device)
    y_tst = torch.Tensor(y_test).to(device)
    y_trn = y_trn.to(torch.long)

    input_size = 22
    output_size = 2
    hidden_size = 80
    learning_rate = 0.0001
    n_epochs = 500

    model = CombinedClassifierModel(input_size=input_size,
                                         output_size=output_size)
    model.to(device)

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    models = []
    acc = []

    for epoch in range(n_epochs):
        predicted = model(x_trn).to(device)

        loss = lossfn(predicted, y_trn)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        val_acc = Validator(x_tst, y_tst.to(torch.int))

        models.append(model)
        acc.append(val_acc)
        # print(f'Epoch [ {epoch + 1} / {n_epochs} ] Training-Loss = {loss.item():.4f} Training-Accuracy = {1 - loss.item()} Validation-Accuracy = {val_acc}')
    return models[acc.index(max(acc))]


def get_nn_model_2():
    global x_train, x_test, y_train, y_test, is_updated

    if "combined_nn.pt" in listdir("C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model"):
        return torch.load("C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\combined_nn.pt").to(device)

    if not is_updated:
        get_trainable_data_2()

    x_trn = torch.Tensor(x_train).to(device)
    y_trn = torch.Tensor(y_train).to(device)
    x_tst = torch.Tensor(x_test).to(device)
    y_tst = torch.Tensor(y_test).to(device)
    y_trn = y_trn.to(torch.long)


    input_size = 12
    output_size = 2
    hidden_size = 80
    learning_rate = 0.0001
    n_epochs = 500

    model = CombinedClassifierModel(input_size=input_size,
                                         output_size=output_size)
    model.to(device)

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    models = []
    acc = []

    for epoch in range(n_epochs):
        predicted = model(x_trn).to(device)

        loss = lossfn(predicted, y_trn)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        val_acc = Validator(model, x_tst, y_tst.to(torch.int))
        acc.append(val_acc)
        models.append(model)

        # print(f'Epoch [ {epoch + 1} / {n_epochs} ] Training-Loss = {loss.item():.4f} Training-Accuracy = {1 - loss.item()} Validation-Accuracy = {val_acc}')

    torch.save(models[acc.index(max(acc))],"C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\combined_nn.pt")
    return models[acc.index(max(acc))]
