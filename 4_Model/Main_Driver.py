from Classifier import get_KNN_Model, get_accuracy_matric
from Hawkes_Process import get_topic_vector
from Topic_Modelling import LDA_main_driver
from filters import train_test_splitter
import numpy as np
import warnings

warnings.simplefilter(action='ignore')

if __name__ == '__main__':
    # Topic Modelling
    df, dict_genuine, dict_fake, lda_genuine, lda_fake = LDA_main_driver()
    num_topics = 10

    # Hawkes Process
    user_topic_vectors, labels = get_topic_vector(df, dict_genuine, dict_fake, lda_genuine, lda_fake, 0)

    total_len = len(user_topic_vectors)
    X = np.array(user_topic_vectors)
    Y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_splitter(X, Y)

    knn = get_KNN_Model(x_train, y_train)

    y_pred = knn.predict(x_test)

    print(get_accuracy_matric(y_pred, y_test, total_len))
