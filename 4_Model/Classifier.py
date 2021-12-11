from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_KNN_Model(X_train, Y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    return knn


def Predict_KNN(knn, X_test):
    return knn.predict(X_test)


def get_accuracy_matric(Y_pred, labels, length):
    a = accuracy_score(Y_pred, labels)
    p, r, f1, _ = precision_recall_fscore_support(Y_pred, labels, average='macro')
    print("Accuracy:", round(a, 3), "Precision:", round(p, 3), "Recall:", round(r, 3), "F1 Score:", round(f1, 3))
    return a, p, r, f1


# linearSVM
def get_lin_SVM_Model(X_train, Y_train):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, Y_train)
    return svclassifier


# polynomial
def get_poly_SVM_Model(X_train, Y_train):
    svclassifier = SVC(kernel='poly', degree=4)
    svclassifier.fit(X_train, Y_train)
    return svclassifier


# gaussian
def get_gauss_SVM_Model(X_train, Y_train):
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, Y_train)
    return svclassifier


# sigmoid
def get_sigmoid_SVM_Model(X_train, Y_train):
    svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(X_train, Y_train)
    return svclassifier


# naive_bayes
def get_NaiveBayes_Model(X_train, Y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    return gnb


# predict
def Predict_model(classifier, X_test):
    return classifier.predict(X_test)
