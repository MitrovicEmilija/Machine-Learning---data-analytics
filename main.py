import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from knnAlgorithm import Knn
from sklearn.utils import shuffle
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, auc

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

#df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
#df['target'] = cancer.target
#df.head()

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

classifier = Knn(n_neighbors=10)
classifier.fit_alg(X_train, y_train)
predict = classifier.predict_knn(X_test)

recall = recall_score(y_test, predict)
precision = precision_score(y_test, predict)
fscore = f1_score(y_test, predict)
auc = roc_auc_score(y_test, predict)
confusion_mat = confusion_matrix(y_test, predict)
a = confusion_mat[0][0] + confusion_mat[1][1]
b = confusion_mat[0][1] + confusion_mat[1][0]
accuracy = a / (a+b)
sensitivity = confusion_mat[0][0] / (confusion_mat[0][0] + confusion_mat[0][1])
specificity = confusion_mat[1][1] / (confusion_mat[1][1] + confusion_mat[1][0])

print(f'Confusion matrix: {confusion_mat}')
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'F-score: {fscore}')
print(f'Auc: {auc}')

fp, tp, thresholds = roc_curve(y_test, predict, pos_label=1)
plt.plot(fp, tp)
title = 'ROC Curve '
plt.title(title)
plt.ylabel('TP rate')
plt.xlabel('FP rate')
plt.show()
ax = sbn.heatmap(confusion_mat, annot=True, fmt='g')
matrixTitle = 'Confusion Matrix - train test split'
ax.set_title(matrixTitle)
ax.set_xlabel('Napovedani podatki')
ax.set_ylabel('Dejanski podatki')
plt.show()

# K-kratna navzkrizna validacija
X, y = load_breast_cancer(return_X_y=True)
X, y = shuffle(X, y, random_state=123)
y_true = y.copy()

skfolds = StratifiedKFold(n_splits=10)

for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)):
    count = 0
    #count = count + 1
    X_trainF = X[train_index]
    y_trainF = y[train_index]
    X_testF = X[test_index]
    y_testF = y[test_index]

    classify = Knn(n_neighbors=10)
    classify.fit_alg(X_trainF, y_trainF)
    prediction = classify.predict_knn(X_testF)

    recall_fold = recall_score(y_testF, prediction)
    precision_fold = precision_score(y_testF, prediction)
    fscore_fold = f1_score(y_testF, prediction)
    auc_fold = roc_auc_score(y_testF, prediction)
    conf_mat = confusion_matrix(y_testF, prediction)
    c = conf_mat[0][0] + conf_mat[1][1]
    d = conf_mat[0][1] + conf_mat[1][0]
    acc = c / (c + d)
    sens = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    specif = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0])

    print(f'------------FOLDS------------')
    print(f'Confusion matrix: {conf_mat}')
    print(f'Accuracy: {acc}')
    print(f'Recall: {recall_fold}')
    print(f'Precision: {precision_fold}')
    print(f'Sensitivity: {sens}')
    print(f'Specificity: {specif}')
    print(f'F-score: {fscore_fold}')
    print(f'Auc: {auc_fold}')

    fp, tp, thresholds = roc_curve(y_testF, prediction, pos_label=1)
    plt.plot(fp, tp)
    title = 'ROC Curve '
    plt.title(title)
    plt.ylabel('TP rate')
    plt.xlabel('FP rate')
    plt.show()
    ax = sbn.heatmap(conf_mat, annot=True, fmt='g')
    matrixTitle = 'Confusion Matrix - kfold'
    ax.set_title(matrixTitle)
    ax.set_xlabel('Napovedani podatki')
    ax.set_ylabel('Dejanski podatki')
    plt.show()

# Izracun povprecne vrednosti za folds
acc_avg = 0
sens_avg = 0
specif_avg = 0
recall_avg = 0
precision_avg = 0
fscore_avg = 0
auc_avg = 0

acc_avg += acc
sens_avg += sens
specif_avg += specif
recall_avg += recall_fold
precision_avg += precision_fold
fscore_avg += fscore_fold
auc_avg += auc_fold

print(f'------------AVERAGE------------')
print(f'Accuracy: {acc_avg/10}')
print(f'Recall: {recall_avg/10}')
print(f'Precision: {precision_avg/10}')
print(f'Sensitivity: {sens_avg/10}')
print(f'Specificity: {specif_avg/10}')
print(f'F-score: {fscore_avg/10}')
print(f'Auc: {auc_avg/10}')