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

    def save(self, output_filename) -> None:
        """
        Saves the dataset to disk in the specified format.

        Args:
            output_filename (str): The name of the file to save the dataset to.

        Returns:
            None
        """
        np.save(output_filename, self.data)
        np.save(f'{output_filename}_X', self.data[:, :-1])
        labels_vectors = [label_to_vector(x) for x in self.data[:, -1]]
        np.save(f'{output_filename}_Y', labels_vectors)

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
        samples_to_add = [int((f-1)*n) for f, n in zip(factors, number_of_samples)]

        def augment_scalar(x, noise):
            return x + np.random.normal(0, noise)
        
        standard_deviations = [np.std(self.data[:, i]) for i in range(12)]
        standard_deviations = [s if s != np.inf else 0 for s in standard_deviations]
        for i, n in enumerate(samples_to_add):
            new_samples = np.zeros((n, 13))
            for j in range(n):
                base = samples[i][np.random.randint(0, number_of_samples[i])]
                for k in range(12):
                    new_samples[j, k] = max(augment_scalar(base[k], noise*standard_deviations[k]), 0)
                new_samples[j, 12] = base[12]
            self.data = np.concatenate((self.data, np.array(new_samples)))
                
            
            

    def plot_label_distribution(self, color="blue", title="Label Distribution") -> None:
        """
        Plot a histogram of the distribution of labels in the dataset using Matplotlib.

        Args:
            color (str): Color for the histogram bars. Default is "blue."
            title (str): Title for the plot. Default is "Label Distribution."

        Returns:
            None
        """
        # Extract labels from the dataset
        labels = self.data[:, -1]

        # Create a figure and axis with a larger size
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the distribution of labels
        bins = np.arange(3, 11) - 0.5  # Adjust bins for better alignment
        ax.hist(labels, bins=bins, color=color,
                edgecolor='black', alpha=0.7, rwidth=0.8)

        # Set the title and axis labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Label', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        # Customize x-axis for better readability
        ax.set_xticks(np.arange(3, 10))
        ax.set_xticklabels([str(i) for i in range(3, 10)])

        # Add labels & counts above the bars
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            ax.text(label, count, str(count),
                    ha='center', va='bottom', fontsize=12)

        # Add grid lines to both axes
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Show the plot
        plt.show()


def saveDatasets(X, Y, filename) -> None:
    np.save(f'{filename}_X.npy', X)
    np.save(f'{filename}_Y.npy', Y)
