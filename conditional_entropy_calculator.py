
import numpy as np
import pandas as pd

def conditional_entropy(y, *features):
    """
    Calculate the conditional entropy H(Y|X1, X2, ..., Xn) for a binary label Y 
    given values of n feature variables X1, X2, ..., Xn.

    Parameters
    ----------
    y : numpy array
        A numpy array containing the binary label data.
    *features : numpy arrays
        One or more numpy arrays containing the feature data, where each array 
        represents one feature variable X1, X2, ..., Xn.

    Returns
    -------
    float
        The conditional entropy H(Y|X1, X2, ..., Xn) as a floating-point number.

    Example
    -------
    >>> y = np.array([0, 1, 1, 0, 1])
    >>> x1 = np.array([1, 2, 3, 4, 5])
    >>> x2 = np.array([5, 4, 3, 2, 1])
    >>> conditional_entropy(y, x1, x2)
    1.9219280948873623
    """
    joint_prob, _ = np.histogramdd([*features, y], bins=[np.unique(var) for var in [*features, y]], density=True)
    joint_prob = joint_prob.flatten()

    features_prob, _ = np.histogramdd(features, bins=[np.unique(var) for var in features], density=True)
    features_prob = features_prob.flatten()

    conditional_prob = joint_prob / (features_prob + np.finfo(float).eps)

    return -np.sum(joint_prob * np.log2(conditional_prob + np.finfo(float).eps))

# Read the data from the CSV file
data = pd.read_csv('data.csv')

# Extract the label data (assumes label is in column M+1)
label = data.iloc[:, -1].to_numpy()

# Define the list of N features to use (e.g., [0, 1, 3] for the first, second, and fourth features)
selected_features = [0, 1, 3]

# Extract the selected feature data
feature_data = [data.iloc[:, feature].to_numpy() for feature in selected_features]

# Compute the conditional entropy of the label given the selected features
conditional_entropy_label_given_features = conditional_entropy(label, *feature_data)
print("Conditional entropy H(Label|Selected Features):", conditional_entropy_label_given_features)
