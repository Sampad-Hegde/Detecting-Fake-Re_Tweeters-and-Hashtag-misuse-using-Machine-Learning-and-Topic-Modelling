from Hawkes_Process import get_topic_vector
from Topic_Modelling import LDA_main_driver
from filters import train_test_splitter
from Classifier import get_KNN_Model, get_lin_SVM_Model, get_NaiveBayes_Model
import torch.nn as nn
import numpy as np
import torch
from os import listdir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train, x_test, y_train, y_test = None, None, None, None
is_data_written = False


class TemporalClassifierModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TemporalClassifierModel, self).__init__()
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


def get_trainable_data():
    global x_train, x_test, y_train, y_test, is_data_written

    df, dict_genuine, dict_fake, lda_genuine, lda_fake = LDA_main_driver()
    user_topic_vectors, labels = get_topic_vector(df, dict_genuine, dict_fake, lda_genuine, lda_fake, 1)

    total_len = len(user_topic_vectors)
    X = np.array(user_topic_vectors)
    Y = np.array(labels)

    X = X[:, [20, 21]]

    x_train, x_test, y_train, y_test = train_test_splitter(X, Y)

    is_data_written = True


def get_tempo_knn_model():
    global x_train, x_test, y_train, y_test, is_data_written

    if not is_data_written:
        get_trainable_data()

    return get_KNN_Model(x_train, y_train)


def get_tempo_svm_model():
    global x_train, x_test, y_train, y_test, is_data_written

    if not is_data_written:
        get_trainable_data()

    return get_lin_SVM_Model(x_train, y_train)


def get_tempo_NB_model():
    global x_train, x_test, y_train, y_test, is_data_written

    if not is_data_written:
        get_trainable_data()

    return get_NaiveBayes_Model(x_train, y_train)


# noinspection PyTypeChecker
def Validator(model, x_test, y_test):
    predicted = model(x_test).to(device)
    pred = torch.max(predicted.data, 1)[1]
    total_test = len(y_test)
    correct_pred = 0

    for i in range(total_test):
        if y_test[i] == pred[i]:
            correct_pred += 1

    return correct_pred / total_test


# noinspection PyTypeChecker,PyArgumentList
def get_tempo_nn_model():
    global x_train, x_test, y_train, y_test, is_data_written

    if "temporal_nn.pt" in listdir("C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model"):
        return torch.load("C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\temporal_nn.pt").to(device)

    if not is_data_written:
        get_trainable_data()

    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    x_test = torch.Tensor(x_test).to(device)
    y_test = torch.Tensor(y_test).to(device)
    y_train = y_train.to(torch.long)


    input_size = 2
    output_size = 2
    learning_rate = 0.0001
    n_epochs = 500

    model = TemporalClassifierModel(input_size=input_size,
                                         output_size=output_size)
    model.to(device)

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    models = []
    acc = []
    for epoch in range(n_epochs):
        predicted = model(x_train).to(device)

        loss = lossfn(predicted, y_train)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        val_acc = Validator(model, x_test, y_test.to(torch.int))

        models.append((model))

        acc.append(val_acc)
        # print(f'Epoch [ {epoch + 1} / {n_epochs} ] Training-Loss = {loss.item():.4f} Training-Accuracy = {1 - loss.item()} Validation-Accuracy = {val_acc}')

    torch.save(models[acc.index(max(acc))], "C:\\Users\\Sampad\\Desktop\\Projects\\Capstone\\Implimentation\\Code\\4_Model\\saved_model\\temporal_nn.pt")

    return models[acc.index(max(acc))]
