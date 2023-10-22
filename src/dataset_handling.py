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

    def __init__(self, filename: str, enable_data_augmentation=False) -> None:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.data = [row for row in reader if reader.line_num != 1]
            self.data = np.array(self.data)
            self.format(enable_data_augmentation)

    def format(self, enable_data_augmentation) -> None:
        self.data[:, 0] = text_to_float(self.data[:, 0])
        self.data = remove_empty(self.data)
        self.data = self.data.astype('float16')
        if enable_data_augmentation:
            self.data_augmentation()

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

    def data_augmentation(self, noise=0.05) -> None:
        """
        Applies data augmentation to the dataset by adding new samples with random noise.

        Args:
            noise (float): the standard deviation of the normal distribution used to generate the noise.
                Defaults to 0.05.

        Returns:
            None
        """
        factors = [30, 7.5, 1, 1, 1.5, 8, 150]

        for quality in range(3, 10):
            if factors[quality-3] != 1:
                f = factors[quality-3]
                n_samples = int(
                    (f-1) * len(self.data[self.data[:, -1] == quality]))
                new_samples = np.zeros((n_samples, self.data.shape[1]))
                new_samples[:, -1] = quality
                for i in range(n_samples):
                    # Add random noise to the sample
                    sample = self.data[np.random.randint(
                        0, len(self.data[self.data[:, -1] == quality])), :-1]
                    noise_values = np.random.normal(1, noise, sample.shape)
                    new_samples[i, :-1] = sample * noise_values
                    new_samples[i, :-
                                1] = np.clip(new_samples[i, :-1], 0, np.inf)
                self.data = np.concatenate((self.data, new_samples), axis=0)

    def plot_label_distribution(self, color="blue", title="Label Distribution") -> None:
        """
        Plot a histogram of the distribution of labels in the dataset.

        Args:
            color (str): Color for the histogram bars. Default is "blue."
            title (str): Title for the plot. Default is "Label Distribution."

        Returns:
            None
        """
        # Set the style to a white background with gridlines
        sns.set_style("whitegrid")

        # Extract labels from the dataset
        labels = self.data[:, -1]

        # Create a figure and axis with a larger size
        plt.figure(figsize=(10, 6))

        # Plot the distribution of labels
        sns.histplot(data=labels, bins=7, color=color, kde=False)

        # Set the title and axis labels
        plt.title(title, fontsize=16)
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Customize x-axis for better readability
        plt.xticks(np.arange(3, 10), labels=[str(i) for i in range(3, 10)])

        # Add labels & counts above the bars
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            plt.text(label, count, str(count),
                     ha='center', va='bottom', fontsize=12)

        # Add grid lines to both axes
        plt.grid(True, linestyle='--', alpha=0.6)

        # Show the plot
        plt.show()


def saveDatasets(X, Y, filename) -> None:
    np.save(f'{filename}_X.npy', X)
    np.save(f'{filename}_Y.npy', Y)
