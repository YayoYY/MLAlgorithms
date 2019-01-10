from sklearn import linear_model
import sys
sys.path.append('../')
from Process import UCIadult_process

train = UCIadult_process.process('../data/UCI_adult/adult.data', None)
test = UCIadult_process.process('../data/UCI_adult/adult.test', [0])

del train['native_country_Holand-Netherlands']

X_train = train.drop(['salary'], axis=1).as_matrix()
y_train = train.salary.as_matrix()

X_test = test.drop(['salary'],axis=1).as_matrix()
y_test = test.salary.as_matrix()
y_test = y_test

clf = linear_model.LogisticRegression(penalty='l2',verbose=2,solver='newton-cg').fit(X_train, y_train)
clf.score(X_test, y_test)