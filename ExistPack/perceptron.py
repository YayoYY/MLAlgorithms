from sklearn import linear_model
import sys
sys.path.append('../')
from Process import UCIadult_process
import warnings
warnings.filterwarnings('ignore')

train = UCIadult_process.process('../data/UCI_adult/adult.data', None)
test = UCIadult_process.process('../data/UCI_adult/adult.test', [0])

del train['native_country_Holand-Netherlands']

X_train = train.drop(['salary'], axis=1).values
y_train = train.salary.values

X_test = test.drop(['salary'],axis=1).values
y_test = test.salary.values
y_test = y_test

clf = linear_model.Perceptron(max_iter=1000,shuffle=True).fit(X_train, y_train)
clf.score(X_test, y_test)
