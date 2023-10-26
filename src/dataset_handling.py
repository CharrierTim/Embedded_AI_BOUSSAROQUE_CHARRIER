import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

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
    format() -> None
        Formats the dataset by removing empty values and converting the labels to vectors.
    split(test_proportion: float) -> tuple
        Splits the dataset into training and testing sets.
    data_augmentation(noise: float) -> None
        Augments the dataset by adding noise to the data.
    """

    def __init__(self, filename: str, data_augmentation_noise=None) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.data = [row for row in reader if reader.line_num != 1]
            self.data = np.array(self.data)
            self.format(data_augmentation_noise)

    def format(self, data_augmentation_noise) -> None:
        self.data = remove_empty(self.data)
        self.data[:, 0] = text_to_float(self.data[:, 0])
        self.data = self.data.astype('float16')
        if data_augmentation_noise is not None:
            self.data_augmentation(data_augmentation_noise)

    def split(self, test_proportion=0.15) -> tuple:
        """
        Splits the dataset into training and testing sets.

        Args:
            test_proportion (float): The proportion of the dataset to use for testing.
                Defaults to 0.15.

        Returns:
            tuple: A tuple containing the training and testing sets, as well as their
                corresponding labels. The tuple is in the format (X_train, X_test, Y_train, Y_test).
        """
        np.random.shuffle(self.data)
        split_index = int((1-test_proportion) * self.data.shape[0])
        X_train, X_test = self.data[:split_index,
                                    :-1], self.data[split_index:, :-1]
        Y_train, Y_test = self.data[:split_index, -
                                    1], self.data[split_index:, -1]
        return X_train, X_test, Y_train, Y_test

    def data_augmentation(self, noise) -> None:
        factors = [1.19, 1.19, 1.19, 1.0, 2.22, 2.22, 2.22]
        samples = [self.data[self.data[:, -1] == i] for i in range(3, 10)]
        number_of_samples = [s.shape[0] for s in samples]
        samples_to_add = [int((f-1)*n)
                          for f, n in zip(factors, number_of_samples)]

        def augment_scalar(x, noise):
            return x + np.random.normal(0, noise)

        standard_deviations = [np.std(self.data[:, i]) for i in range(12)]
        standard_deviations = [
            s if s != np.inf else 0 for s in standard_deviations]
        for i, n in enumerate(samples_to_add):
            new_samples = np.zeros((n, 13))
            for j in range(n):
                base = samples[i][np.random.randint(0, number_of_samples[i])]
                for k in range(12):
                    new_samples[j, k] = max(augment_scalar(
                        base[k], noise*standard_deviations[k]), 0)
                new_samples[j, 12] = base[12]
            self.data = np.concatenate((self.data, np.array(new_samples)))


def plot_label_distribution(x, y_tr, y_te, color_tr="blue", color_te="red", title="Wine Quality Label Distribution"):
    """
    Plot a bar chart of the distribution of labels in the dataset using Matplotlib.

    Args:
        x (list): List of x-axis values.
        y_tr (list): List of y-axis values for the training set.
        y_te (list): List of y-axis values for the test set.
        color_tr (str): Color for the training set bars. Default is "blue."
        color_te (str): Color for the test set bars. Default is "red."
        title (str): Title for the plot. Default is "Label Distribution."

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y_tr, color=color_tr, label='Train', alpha=0.7)
    ax.bar(x, y_te, color=color_te, label='Test', alpha=0.7)

    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel('Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')

    # Add labels & counts above the bars
    for i, (tr, te) in enumerate(zip(y_tr, y_te)):
        ax.text(i, max(tr, te), f"{tr}/{te}",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.legend(fontsize=14, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.show()


def plot_history(history, color_loss="blue", color_val="red"):
    """
    Plot the training and validation accuracy and loss curves for a Keras model.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit().

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history.history['accuracy'],
            label='Train Accuracy', linewidth=2, color=color_loss)
    ax.plot(history.history['val_accuracy'],
            label='Validation Accuracy', linewidth=2, color=color_val)

    ax.set_title('Model Accuracy', fontsize=20, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')

    ax.legend(fontsize=14)
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


def saveDatasets(X, Y, filename) -> None:
    np.save(f'{filename}_X.npy', X)
    np.save(f'{filename}_Y.npy', Y)
