from utils import mnist_reader
from OtherFunctions import *

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

train_sizes, cv = Parameters(6, 100, 0.2)

BestEstimatorCurve('decision tree', x_train, y_train, y_train, y_test, train_sizes, cv)

BestEstimatorCurve('perceptron', x_train, y_train, y_train, y_test, train_sizes, cv)

PlotDecisionTree(cv, x_train, y_train, y_train, y_test, train_sizes=train_sizes, criterio='entropy', min_sample_leaf=5, min_sample_split=2, maxDepth=20)
PlotDecisionTree(cv, x_train, y_train, y_train, y_test, train_sizes=train_sizes, criterio='entropy', min_sample_leaf=5, min_sample_split=2, maxDepth=5)

PlotPerceptron(cv, x_train, y_train, y_train, y_test, train_sizes=train_sizes, maxIter=50)
PlotPerceptron(cv, x_train, y_train, y_train, y_test, train_sizes=train_sizes, maxIter=5)
PlotPerceptron(cv, x_train, y_train, y_train, y_test, train_sizes=train_sizes, maxIter=2)

DecisionTreeGridSearch(cv)

PerceptronGridSearch(cv)

#inizio confronto

#trainSizes, cv = Parameters(6,100,0.3)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=2, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=2, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 2)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=5, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=5, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 5)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=10, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=10, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 10)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=15, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=15, minSamplesLeaf=3, minSamplesSplit=5)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=20, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=20, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 20)

#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 50)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 100)

#cambio parametri

#trainSizes, cv = Parameters(6,100,0.2)

#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 5)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=5, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=5, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 10)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=10, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=10, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 20)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=15, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=15, minSamplesLeaf=3, minSamplesSplit=5)



#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=20, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=20, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 50)



#trainSizes, cv = Parameters(6,100,0.1)

#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 5)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=5, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=5, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 10)

#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=10, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=10, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 20)


#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=15, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=15, minSamplesLeaf=3, minSamplesSplit=5)


#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'gini', maxDepth=20, minSamplesLeaf=3, minSamplesSplit=5)
#DecisionTree(cv, x_train, y_train, x_test, y_test, trainSizes, 'entropy', maxDepth=20, minSamplesLeaf=3, minSamplesSplit=5)
#PerceptronClass(cv, x_train, y_train, x_test, y_test, trainSizes, 50)


#Ricerca parametri ideali

#cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#parameter_grid = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,20)}
#estimator = DecisionTreeClassifier(random_state=0, min_sample_leaf=3, min_sample_split=5)
#grid_search = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1)
#grid_search.fit(x_train,y_train)
#print "Best Score DecisionTree: %.2f%%", 100*(grid_search.best_score_)
#print "Best params DecisionTree: {}".format(grid_search.best_params_)


#parameter_grid={'max_iter': range(1,50)}
#estimator = Perceptron()
#grid_search = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1)
#grid_search.fit(x_train,y_train)
#print "Best Score Perceptron: %.2f%%", 100*(grid_search.best_score_)
#print "Best params Perceptron: {}".format(grid_search.best_params_)


cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#parameter_grid = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,15), 'min_samples_split': range(2,10), 'min_samples_leaf':range(1,8)}
#estimator = DecisionTreeClassifier(random_state=0)
#grid_search = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1, scoring='accuracy')
#grid_search.fit(x_train,y_train)
#print "Best Score DecisionTree: %.2f%%", 100*(grid_search.best_score_)
#print "Best params DecisionTree: {}".format(grid_search.best_params_)
#means = grid_search.cv_results_['mean_test_score']
#stds = grid_search.cv_results_['std_test_score']
#print 'Accuracy scores on training set'
#for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#y_pred = grid_search.predict(x_test)
#print classification_report(y_test,y_pred)




parameter_grid={'max_iter': range(1,35), 'eta0': [0.0001, 0.001, 0.01, 0.1, 0.5, 1]}
estimator = Perceptron()
grid_search = GridSearchCV(estimator, param_grid=parameter_grid, cv=cv, n_jobs=-1, scoring='accuracy')
grid_search.fit(x_train,y_train)
print "Best Score Perceptron: %.2f%%", 100*(grid_search.best_score_)
print "Best params Perceptron: {}".format(grid_search.best_params_)
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
print 'Accuracy scores on training set'
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
y_pred = grid_search.predict(x_test)
print classification_report(y_test,y_pred)


#plt.show()
