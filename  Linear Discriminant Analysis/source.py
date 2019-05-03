import numpy as np
from sklearn import datasets

# Configuration
np.set_printoptions(precision=4)

# Variables
CLASS_NUM = 3
ATTRIBUTES_NUM = 4
EXAMPLE_PER_CLASS_NUM = 50


def lda():
    """
    Linear Discriminant Analysis.

    Linear Discriminant Analysis (LDA) is similar to the
    PCA (Principle Component Analysis), but it focuses on
    maximizing the separability among known categories,
    while PCA looks at the categories with most variation.

    LDA dimensionality reduction and maximum separability is
    acquired by creating nex axis among existing ones using
    information from all categories. Then, data is projected
    onto the new axis in the way to maximize the separation
    between categories.

    Arguments:

    Returns:

    """
    pass


def create_new_axis():
    """
    For 2 categories:
    0. Create new axis.
    1. Maximize distance between means of categories.
    2. Minimize the variation (scatter) within each category.

    From 1. and 2. => (d ^ 2) / (s1 ^ 2 + s2 ^ 2)
      - d (distance) = m1 - m2
      - Squares are used in order to prevent negative values.
      - Ideal case is where distance is large, and scatter small.

    For 3 categories:
    0. Create 2 new axis (2 points define a line, while 3 points
    define a plane - this case).
    1. Maximize distance between means of categories and the
    central point between.
    2. Minimize the variation (scatter) within each category.

    From 1. and 2. => (d1 ^ 2 + d2 ^ 2 + d3 ^ 2) / (s1 ^ 2 + s2 ^ 2 + s3 ^ 2)

    Arguments:

    Returns:

    """
    pass


def main():
    # Load dataset
    iris = datasets.load_iris()

    # Calculate mean vector
    mean_by_class = []
    for classes in range(CLASS_NUM):
        mean_by_class.append(np.mean(iris.data[iris.target == classes], axis=0))

    mean_by_class = np.array(mean_by_class)

    # Scatter matrix (approximation of the covariance matrix)
    # # Within-Class
    scatter_within = np.zeros((ATTRIBUTES_NUM, ATTRIBUTES_NUM))
    for classes in range(CLASS_NUM):
        norm = iris.data[iris.target == classes] - mean_by_class[classes]
        norm = norm.T
        scatter_within += norm @ norm.T

    # # Between-Class
    scatter_between = np.zeros((ATTRIBUTES_NUM, ATTRIBUTES_NUM))
    overall_mean = np.mean(iris.data, axis=0)
    for classes in range(CLASS_NUM):
        norm = mean_by_class[classes, :] - overall_mean
        norm = norm.reshape((ATTRIBUTES_NUM, 1))
        scatter = norm @ norm.T
        scatter_between += EXAMPLE_PER_CLASS_NUM * scatter

    # Eigenvector & Eigenvalues

    return 0


if __name__ == '__main__':
    main()
