import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#plt.rcParams['figure.figsize'] = (10.0, 8.0)


def plot(train_sizes, train_scores, test_scores, invert_score=True, title='', ylim=None, description=''):

    plt.figure()
    plt.suptitle(description)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.grid(True)
    if invert_score:
        train_scores = 1-train_scores
        test_scores = 1-test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test error")
    plt.legend(loc="best")



def DecisionTree(CrossVal, x, y, x_test, y_test, TrainS, criterio, minSamplesSplit=2, minSamplesLeaf=1, maxDepth=None):

    estimator = tree.DecisionTreeClassifier(criterion=criterio, max_depth=maxDepth,
                                            min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf)

    trainSizes, trainScore, testScore = learning_curve(estimator , x, y, train_sizes=TrainS, cv=CrossVal, n_jobs=-1)
    descr=str('DecisionTree usando %s con maxDepth: %d , minSampleSplit: %d , minSampleLeaf: %d ' % (criterio, maxDepth, minSamplesSplit, minSamplesLeaf))
    plot(trainSizes, trainScore, testScore, title='DecisionTree', description=descr)
    #aggiungi predizione e misura accuratezza
    estimator.fit(x,y)
    y_pred = estimator.predict(x_test)
    accuracy= accuracy_score(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred)
    print '%.2f%% Accuratezza per DecisionTree usando %s con maxDepth: %d , minSampleSplit: %d , minSampleLeaf: %d ' % (accuracy, criterio, maxDepth, minSamplesSplit, minSamplesLeaf)
    print 'matrice : \n', matrix
    print 'Classification report: \n', classification_report(y_test, y_pred)


def Parameters(nSamples, nSplit, testSize):
    trainSizes=np.logspace(np.log10(0.05), np.log10(1.0), num=nSamples)
    cv=ShuffleSplit(n_splits=nSplit,test_size=testSize,random_state=0)
    return trainSizes, cv

def PerceptronClass( CrossVal, x, y, x_test, y_test, TrainS, maxIter=5):

    estimator =Perceptron(max_iter=maxIter)

    trainSizes, trainScore, testScore = learning_curve(estimator, x, y, train_sizes=TrainS, cv=CrossVal, n_jobs=-1)
    descr=str('Perceptron usando maxIter: %d ' % maxIter)
    plot(trainSizes, trainScore, testScore, title='Perceptron', description=descr)
    #aggiungi predizione e misure accuratezza
    y_pred=estimator.fit(x,y).predict(x_test)
    accuracy= 100*accuracy_score(y_test,y_pred)
    matrix=confusion_matrix(y_test,y_pred)
    print '%.2f%% Accuratezza per perceptron usando maxIter: %d ' % (accuracy, maxIter)
    print 'matrice: \n', matrix
    print 'Classification report: \n', classification_report(y_test, y_pred)
