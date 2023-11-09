from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from dataset_handling import Dataset

from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error


def average_error(y_true, y_pred):
    """
    Computes the average error between two vectors.

    Attributes:
    -----------
    y_true : numpy.ndarray
        The true values.

    y_pred : numpy.ndarray
        The predicted values.

     Methods:
    --------
    average_error(y_true, y_pred) -> float
        Computes the average error between two vectors.
    """
    sum = 0
    for y_t, y_p in zip(y_true, y_pred):
        sum += abs(y_t-y_p)
    return sum / y_true.shape[0]

if __name__ == '__main__':
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

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(20, 10, 3), random_state=1)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    err = average_error(Y_test, Y_pred)
    print(f'MLP: {err}')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    err = average_error(Y_test, Y_pred)
    print(f'RF: {err}')

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    err = average_error(Y_test, Y_pred)
    print(f'KNN: {err}')

    model = Sequential()
    model.add(Dense(20, input_dim=12, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=mean_squared_error, optimizer='adam')
    model.fit(X_train, Y_train/10, epochs=15, batch_size=32,
            validation_data=(X_test, Y_test/10))
    Y_pred = model.predict(X_test)
    err = average_error(Y_test, Y_pred*10)
    print(f'NN: {err}')
