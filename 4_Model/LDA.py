import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from sys import path as syspath
from os import path as osPath, getcwd
from Classifier import get_KNN_Model, get_lin_SVM_Model, get_NaiveBayes_Model
import numpy as np
import torch.nn as nn
import warnings
import torch

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from filters import train_test_splitter
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from os import listdir

warnings.simplefilter(action='ignore')

syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\2_Cleaning_Visualization')

# noinspection PyUnresolvedReferences
from text_cleaning import text_clean

num_topics = 10

x_train, x_test, y_train, y_test = None, None, None, None
df, dict_genuine, dict_fake, lda_genuine, lda_fake = None, None, None, None, None

is_LDA_trained = False
data_updated = False

LDA = None
corpus = None
Dict = None


class LstmClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LstmClassifierModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, inputs.size(1), self.hidden_size).to(device)

        out, _ = self.lstm(inputs, (h0, c0))
        out = out[:, -1, :]

        out = self.fc(out)

        return out


class LDAClassifierModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LDAClassifierModel, self).__init__()
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


def get_Device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = get_Device()


def get_topic_vector(tweet_text, num_topics, Dict, LDA):
    d2b = Dict.doc2bow(tweet_text)

    topic_vector_sparse = LDA.get_document_topics(d2b)

    topic_vector = np.zeros(num_topics)

    for pair in topic_vector_sparse:
        topic_vector[pair[0]] = pair[1]

    return topic_vector


def get_LDA_trained_Models():
    dataset_path = "C:/Users/Sampad/Desktop/Projects/Capstone/Implimentation/Code/0_DataSet/"

    df = pd.read_csv(dataset_path + "CompleteAnnotated.csv")

    df.tweet_text = df.tweet_text.apply(text_clean)

    all_docs_genuine = []
    all_docs_fake = []
    labels = []
    complete_docs = []

    for i in range(df.shape[0]):
        labels.append(df.iloc[i]['Annotation'])
        complete_docs.append(df.iloc[i]['tweet_text'].split())
        if df.iloc[i]['Annotation'] == 0:
            all_docs_genuine.append(df.iloc[i]['tweet_text'].split())
        else:
            all_docs_fake.append(df.iloc[i]['tweet_text'].split())
    complete_dict = Dictionary(complete_docs)

    complete_corpus = [complete_dict.doc2bow(text) for text in complete_docs]

    lda = LdaModel(complete_corpus, num_topics=10)

    X = np.array([get_topic_vector(tweet.split(), 10, complete_dict, lda) for tweet in df.tweet_text])
    Y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_splitter(X, Y)
    return x_train, x_test, y_train, y_test, complete_dict, complete_corpus, lda


def get_BOW_trainable_data():
    dataset_path = "C:/Users/Sampad/Desktop/Projects/Capstone/Implimentation/Code/0_DataSet/"

    df = pd.read_csv(dataset_path + "CompleteAnnotated.csv")

    df.tweet_text = df.tweet_text.apply(text_clean)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.tweet_text)
    counts = tokenizer.word_counts

    word_size = 7000
    vocab_size = word_size
    tokenizer = Tokenizer(num_words=word_size)

    tokenizer.fit_on_texts(df.tweet_text)
    tokenized = tokenizer.texts_to_sequences(df.tweet_text)

    sequence_size = 18
    padded = pad_sequences(tokenized, maxlen=sequence_size, padding='post', truncating='post')
    x_train, x_test, y_train, y_test = train_test_split(padded, df.Annotation.values, test_size=0.20)
    return x_train, x_test, y_train, y_test, tokenizer


def get_trainable_data():
    global x_train, x_test, y_train, y_test, data_updated, LDA, corpus, Dict
    x_train, x_test, y_train, y_test, Dict, corpus, LDA = get_LDA_trained_Models()
    data_updated = True
    return x_train, x_test, y_train, y_test


def get_text_knn_model():
    global x_train, x_test, y_train, y_test, data_updated
    if data_updated == False:
        x_train, x_test, y_train, y_test = get_trainable_data()
    return get_KNN_Model(x_train, y_train)


def get_text_svm_model():
    global x_train, x_test, y_train, y_test, data_updated
    if data_updated == False:
        x_train, x_test, y_train, y_test = get_trainable_data()
    return get_lin_SVM_Model(x_train, y_train)


def get_text_NB_model():
    global x_train, x_test, y_train, y_test, data_updated
    if data_updated == False:
        x_train, x_test, y_train, y_test = get_trainable_data()
    return get_NaiveBayes_Model(x_train, y_train)


def Validator(model, x_test, y_test):
    predicted = model(x_test).to(device)
    pred = torch.max(predicted.data, 1)[1]
    total_test = len(y_test)
    correct_pred = 0

    for i in range(total_test):
        if y_test[i] == pred[i]:
            correct_pred += 1

    return correct_pred / total_test


def get_text_nnLDA_model():
    global x_train, x_test, y_train, y_test, data_updated

    if "textual_simple_nn.pt" in listdir(
            "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model"):
        return torch.load(
            "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\textual_simple_nn.pt").to(
            device)

    if data_updated == False:
        x_train, x_test, y_train, y_test = get_trainable_data()

    x_tr = torch.Tensor(x_train).to(device)
    y_tr = torch.Tensor(y_train).to(device)
    x_ts = torch.Tensor(x_test).to(device)
    y_ts = torch.Tensor(y_test).to(device)
    y_tr = y_tr.to(torch.long)

    input_size = 10
    output_size = 2
    learning_rate = 0.0001
    n_epochs = 250

    model = LDAClassifierModel(input_size=input_size,
                               output_size=output_size)

    model.to(device)

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    models = []
    acc = []

    for epoch in range(n_epochs):
        predicted = model(x_tr).to(device)

        loss = lossfn(predicted, y_tr)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        val_acc = Validator(model, x_ts, y_ts.to(torch.int))

        acc.append(val_acc)
        models.append(model)

        # print(f'Epoch [ {epoch + 1} / {n_epochs} ] Training-Loss = {loss.item():.4f} Training-Accuracy = {1 - loss.item()} Validation-Accuracy = {val_acc}')
    model = models[acc.index(max(acc))]
    torch.save(model,
               "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\textual_simple_nn.pt")
    return model


def get_text_nnBOW_model():
    x_train, x_test, y_train, y_test, tokenizer = get_BOW_trainable_data()

    if "textual_lstm_nn.pt" in listdir(
            "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model"):
        return torch.load(
            "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\textual_lstm_nn.pt").to(
            device), tokenizer

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    x_test = torch.Tensor(x_test).to(device)
    y_test = torch.Tensor(y_test).to(device)
    y_train = y_train.to(torch.long)

    input_size = 18
    output_size = 2
    hidden_size = 256
    num_layers = 16
    learning_rate = 0.0001
    n_epochs = 250

    model = LstmClassifierModel(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                output_size=output_size)

    model.to(device)

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    models = []
    acc = []

    for epoch in range(n_epochs):
        predicted = model(x_train).to(device)

        loss = lossfn(predicted, y_train)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        val_acc = Validator(model, x_test, y_test.to(torch.int))
        acc.append(val_acc)
        models.append(model)

        # print(f'Epoch [{epoch + 1}/{n_epochs}] Training-Loss = {loss.item():.4f} Train-Accuracy = {1 - loss.item():.4f} Valid-Accuracy = {val_acc}')

    torch.save(models[acc.index(max(acc))],
               "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\textual_lstm_nn.pt")
    return models[acc.index(max(acc))], tokenizer


def get_text_bilstm_model():
    x_train, x_test, y_train, y_test, tokenizer = get_BOW_trainable_data()
    y_train = to_categorical(y_train, num_classes=2)

    word_vec_size = 20
    hidden_size = 128
    sequence_size = 18
    vocab_size = 7000

    if "textual_bilstm_nn.h5" in listdir(
            "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model"):
        return load_model(
            "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\textual_bilstm_nn.h5"), tokenizer

    model = Sequential()
    model.add(Input(shape=[sequence_size]))
    model.add(Embedding(vocab_size, word_vec_size, input_length=sequence_size))

    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(int(hidden_size / 2), return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(int(hidden_size / 2), return_sequences=True)))

    model.add(Flatten())
    model.output_shape
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    # model = keras.models.Model(X,Y)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    es = EarlyStopping(monitor='val_accuracy', mode='min', patience=6, verbose=1)

    hist = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, callbacks=[es])

    model.save(
        "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\textual_bilstm_nn.h5")
    return model, tokenizer
