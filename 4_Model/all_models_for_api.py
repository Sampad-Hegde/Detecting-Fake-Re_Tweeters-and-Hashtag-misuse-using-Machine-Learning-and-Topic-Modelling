from Temporal_models import get_tempo_knn_model, get_tempo_svm_model, get_tempo_NB_model, get_tempo_nn_model
from LDA import get_text_knn_model, get_text_svm_model, get_text_NB_model, get_text_nnLDA_model, get_text_nnBOW_model, get_text_bilstm_model
from Combined_Models import get_knn_model, get_svm_model, get_NB_model, get_nn_model_2, get_trainable_data_2
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_all_temporal_models():
    return get_tempo_knn_model(), get_tempo_svm_model(), get_tempo_NB_model(), get_tempo_nn_model()

def get_all_textual_models():
    return get_text_knn_model(), get_text_svm_model(), get_text_NB_model(), get_text_nnLDA_model(), get_text_nnBOW_model()[0], get_text_bilstm_model()[0]

def get_all_combined_models():
    get_trainable_data_2()
    return get_knn_model(), get_svm_model(), get_NB_model(), get_nn_model_2()


def get_all_models():
    models =  (get_all_temporal_models(), get_all_textual_models(), get_all_combined_models())
    m = []
    [ m.extend(list(x)) for x in models ]
    return m


def get_predicted_class_torch(model,data):
    global  device
    data = torch.Tensor(data).to(device)
    predicted = model(data).to(device)
    predicted = torch.max(predicted.data, 1)[1]
    return predicted



def get_predicted_class_tf(model,data):
    return np.argmax(model(data[0]), axis=-1)

def get_all_predictions(datas,models):
    res = []

    for i in range(len(models)):
        if i in [0, 1, 2, 4, 5, 6, 10, 11, 12]:
            res.append(int(models[i].predict(datas[i])[0]))
        elif i in [3,7,8,13]:
            res.append(int(get_predicted_class_torch(models[i], datas[i])[0]))
        else:
            res.append(int(get_predicted_class_tf(models[i], datas[i])[0]))
    return res