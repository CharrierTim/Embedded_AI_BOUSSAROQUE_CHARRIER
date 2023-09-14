from dataset_handling import Dataset
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def average_error(y_true, y_pred):
    sum = 0
    for y_t, y_p in zip(y_true, y_pred):
        sum += abs(y_t-y_p)
    return sum / y_true.shape[0]


dataset = Dataset('./dataset/winequalityN.csv')
X_train, X_test, Y_train, Y_test = dataset.split()

svm = SVC()
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_test)
err = average_error(Y_test, Y_pred)
print(f'SVM: {err}')

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)
err = average_error(Y_test, Y_pred)
print(f'GNB: {err}')

log = LogisticRegression(max_iter=1500)
log.fit(X_train, Y_train)
Y_pred = log.predict(X_test)
err = average_error(Y_test, Y_pred)
print(f'LOG: {err}')

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 3), random_state=1)
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)
err = average_error(Y_test, Y_pred)
print(f'MLP: {err}')