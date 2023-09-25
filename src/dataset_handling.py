import csv
import numpy as np

text_to_float = np.vectorize(lambda x: float(x == 'red'))
remove_empty = np.vectorize(lambda x: '0' if x == '' else x)
def label_to_vector(x: float) -> list:
    vector = 7*[0]
    vector[int(x)-3] = 1
    return vector

class Dataset:

    """
    A class to handle datasets.

    Attributes:
    -----------
    data : numpy.ndarray
        The dataset as a numpy array.

    Methods:
    --------
    __init__(filename: str) -> None
        Initializes the DatasetHandler object by reading the dataset from a file and formatting it.
    """

    def __init__(self, filename: str) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.data = [row for row in reader if reader.line_num != 1]
            self.data = np.array(self.data)
            self.format()

    def format(self) -> None:
        self.data[:, 0] = text_to_float(self.data[:, 0])
        self.data = remove_empty(self.data)
        self.data = self.data.astype('float32')

    def save(self, output_filename) -> None:
        np.save(output_filename, self.data)
        np.save(f'{output_filename}_X', self.data[:, :-1])
        labels_vectors = [label_to_vector(x) for x in self.data[:, -1]]
        np.save(f'{output_filename}_Y', labels_vectors)
    
    def split(self, test_proportion=0.15) -> tuple:
        np.random.shuffle(self.data)
        split_index = int((1-test_proportion) * self.data.shape[0])
        X_train, X_test = self.data[:split_index, :-1], self.data[split_index:, :-1]
        Y_train, Y_test = self.data[:split_index, -1], self.data[split_index:, -1]        
        return X_train, X_test, Y_train, Y_test


def saveDatasets(X, Y, filename):
    np.save(f'{filename}_X.npy', X)
    np.save(f'{filename}_Y.npy', Y)