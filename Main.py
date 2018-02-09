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

plt.show()
