#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn
import time
import itertools
from sklearn.ensemble import partial_dependence, GradientBoostingClassifier
from sklearn import metrics, model_selection
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


"""
Purpose: This script implements Gradient Boosting classification algorithm to predict risk
for strokes with SMOTE Oversampling technique

https://www.kaggle.com/qianchao/smote-with-imbalance-data
https://github.com/pprett/pydata-gbrt-tutorial/blob/master/gbrt-tutorial.ipynb
https://github.com/rempic/DC-GradientBoosting/blob/master/datachallenge_GradientBoostingClassifier.ipynb
https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py
"""

timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
print("Created on: {}".format(timestr))

__author__ = "Olga Lyashevska"
__email__ = "olga.lyashevska@gmit.ie"
__status__ = "Development"


# read in data
data = pd.read_csv('../data/data.csv', index_col=0)

# code head and no head
data['Head'] = data.Branch.map({'lcca': 1,
                                'rcca': 1,
                                'lsub': 0,
                                'rsub': 0,
                                'descending': 0})

# To normalise distribution we eliminate blood clots above 10 cm
data = data[data.Length < 10]

# split data into X and y
y = data.Head.ravel()
# select all columns except Head, thrombin and Branch
X = data.loc[:, data.columns != 'Head']

# drop Thrombin, Branch
X.drop(['Thrombin', 'Branch'], axis=1, inplace=True)

# select columns to convert
cols = list(X.select_dtypes(include=['object']).columns.values)
cols

# encode string class values as integers using LabelEncoder
# for inverse_transform and transform
# d = defaultdict(LabelEncoder)
# # encode the variable
# X[cols] = X[cols].select_dtypes(include=['object']).apply(lambda x: d[x.name].fit_transform(x))
# to inverse the encoded
#X[cols].apply(lambda x: d[x.name].inverse_transform(x))
# encode string class values as integers using One Hot encoder
# X[cols] = X[cols].select_dtypes(include=['object']).apply(lambda x: d[x.name].fit_transform(x))

one_hot = pd.get_dummies(X[cols], drop_first=True)

# drop original cols
X.drop(cols, axis=1, inplace=True)

# join the encoded
X = X.join(one_hot)

print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5,
                                                                    random_state=13)
print('X_train:', X_train.shape, "\n"
      'X_test:', X_test.shape, "\n"
      "y_train:", y_train.shape, "\n"
      "y_test:", y_test.shape, "\n")

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {}".format(sum(y_train == 0)))

# SMOT over_sampling
# SMOT finds the k-nearest-neighbors for minority class observations

sm = SMOTE(sampling_strategy=1,
           random_state=None,
           k_neighbors=5,
           n_jobs=1
           )

X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of X_train: {}'.format(X_train_sm.shape))
print('After OverSampling, the shape of y_train: {}'.format(y_train_sm.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_sm == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_sm == 0)))
# equal number of 1 and 0
X_test_sm, y_test_sm = sm.fit_sample(X_test, y_test.ravel())

print('After OverSampling, the shape of X_test: {}'.format(X_test_sm.shape))
print('After OverSampling, the shape of y_test: {}'.format(y_test_sm.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_test_sm == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_test_sm == 0)))

###############################################################################
# GradientBoostingClassifier
###############################################################################

# tuning the hyperparameter of an estimator
param_tune = {'max_depth': list(np.arange(1, 11)),
              'learning_rate': list(np.arange(start=0.001, stop=0.5, step=0.05))
              }

est_tune = GradientBoostingClassifier(n_estimators=80, verbose=0)

# exhaustive grid search generates candidates from a grid of parameter values
gs = model_selection.GridSearchCV(estimator=est_tune,
                                  param_grid=param_tune,
                                  cv=5,
                                  n_jobs=4,
                                  verbose=0
                                  )

gs.fit(X_train_sm, y_train_sm)

print("Best model parameters are: {}".format(gs.best_params_))

# add extra parameters
gs.best_params_['subsample'] = 0.75
# gs.best_params_['n_estimators'] = 500


# Model fitting
est_tune.set_params(**gs.best_params_)

est_tune.fit(X_train_sm, y_train_sm)

# Print the score (R^2)
acc_train = est_tune.score(X_train_sm, y_train_sm)
# acc_test = est_tune.score(X_test_sm, y_test_sm)
acc_test = est_tune.score(X_test_sm, y_test_sm)


print('Accuracy:')
print('R^2 train: %.4f' % acc_train)
print('R^2 test: %.4f' % acc_test)

# mse = metrics.mean_squared_error(y_test, est_tune.predict(X_test))
# print('MSE: %.4f' % mse)

# compute test set deviance
test_score = np.zeros((est_tune.n_estimators,), dtype=np.float64)
# for i, y_pred in enumerate(est_tune.staged_predict(X_test_sm)):
#     test_score[i] = est_tune.loss_(y_test_sm, y_pred)
for i, y_pred in enumerate(est_tune.staged_predict(X_test)):
    test_score[i] = est_tune.loss_(y_test, y_pred)

plt.figure(figsize=(10, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(est_tune.n_estimators) + 1,
         est_tune.train_score_, 'b-',
         label='Train')
plt.plot(np.arange(est_tune.n_estimators) + 1,
         test_score, 'r-',
         label='Test')
plt.legend(loc='right')
plt.xlabel('Boosting Iterations')
plt.ylabel('MSE')
plt.savefig('../paper/figs/deviance.eps', format='eps')

# Feature importance

names = X.columns.values
fx_imp = pd.Series(est_tune.feature_importances_, index=names)
fx_imp /= fx_imp.max()  # normalize
fx_imp.sort_values(ascending=False).plot(kind='barh', figsize=(8, 5), color='g')
plt.savefig('../paper/figs/importance.eps', format='eps')

# partial dependence plot

# take top 3 features and produce combinations
# fx_imp_3 = list(fx_imp.sort_values(ascending=False)[0:3].index)
fx_imp.sort_values().index

fx_imp_3 = ['Length', 'Diameter', 'Flowrate_normal']

features = []
for i in range(1, len(fx_imp_3)):
    els = [list(x) for x in itertools.combinations(fx_imp_3, i)]
    features.extend(els)

fig, axs = partial_dependence.plot_partial_dependence(est_tune,
                                                      X_train_sm,
                                                      features,
                                                      feature_names=list(X.columns.values),
                                                      n_cols=3,
                                                      figsize=(10, 5)
                                                      )
fig.suptitle('Partial dependence plots')
plt.subplots_adjust(top=0.9)
plt.savefig('../paper/figs/interaction.eps', format='eps')

# Early stopping


#######################################
# confusion Matrix

"""
To describe performance of classification model we use confusion matrix
confusion matrix cas 2x2 dimensions, because it is a binaryclassification.
Non-diagonal element are innacurate predictions
"""


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1  # print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')


# confusin matrix train set

y_train_pre = est_tune.predict(X_train_sm)
cnf_matrix_tra = metrics.confusion_matrix(y_train_sm, y_train_pre)

print("Recall metric in the train dataset: {:.3f}%".format(
    100*cnf_matrix_tra[1, 1]/(cnf_matrix_tra[1, 0]+cnf_matrix_tra[1, 1])))

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra, classes=class_names, title='Confusion matrix')
plt.show()


# confusin matrix test set

y_pre_sm = est_tune.predict(X_test_sm)
cnf_matrix = metrics.confusion_matrix(y_test_sm, y_pre_sm)

print("Recall metric in the testing dataset: {}%".format(
    100*cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1])))

print("Precision metric in the testing dataset: {}%".format(
    100*cnf_matrix[0, 0]/(cnf_matrix[0, 0]+cnf_matrix[1, 0])))
# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig('../paper/figs/confusion.eps', format='eps')


# ROC curve

"""In a ROC curve the TP rate (Sensitivity) is plotted as function of the
FP rate (100-Specificity) for different cut-off points of a parameter.
The area under the ROC curve (AUC) is a measure of how well a parameter
can distinguish between two outcomes Stroke/No Stroke.
Receiver Operating Characteristic(ROC) curve is a plot of the true positive
rate against the false positive rate.
It shows the trade off between sensitivity and specificity.
"""

tmp = est_tune.fit(X_train_sm, y_train_sm.ravel())
y_pred_sample_score = tmp.decision_function(X_test_sm)

fpr, tpr, thresholds = metrics.roc_curve(y_test_sm, y_pred_sample_score)
roc_auc = metrics.auc(fpr, tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('../paper/figs/roc.eps', format='eps')


# Classification report
print('Classification report: \n', metrics.classification_report(
    y_test_sm, est_tune.predict(X_test_sm)))

"""
The precision is the ability of the classifier not to label a sample as
positive if it is negative.

In binary classification:
    recall of the positive class is also known as “sensitivity”
    recall of the negative class is “specificity”.
"""

print("Test accuracy {:.3f}".format(metrics.accuracy_score(y_test_sm, est_tune.predict(X_test_sm))))
print("Precision {:.3f}".format(metrics.precision_score(y_test_sm, est_tune.predict(X_test_sm))))
print("Recall {:.3f}".format(metrics.recall_score(y_test_sm, est_tune.predict(X_test_sm))))
