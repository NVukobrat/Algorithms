import numpy as np
from sklearn import datasets

# Configuration
np.set_printoptions(precision=4)

# Variables
CLASS_NUM = 3
ATTRIBUTES_NUM = 4
EXAMPLE_PER_CLASS_NUM = 50
EIG_ACCEPTANCE_PERCENTAGE = 0.995


def lda(X, y):
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
        X - Input dataset where row represent each example
        while columns represent attributes (dimensions).
        y - Input dataset classes where each row is
        corresponding to the X row and represent its class.

    Returns:
        New dataset with reduced dimensions.

    """
    # Calculate mean vector
    mean_by_class = []
    for classes in range(CLASS_NUM):
        mean_by_class.append(np.mean(X[y == classes], axis=0))

    mean_by_class = np.array(mean_by_class)

    # Scatter matrix (approximation of the covariance matrix)
    # # Within-Class
    scatter_within = np.zeros((ATTRIBUTES_NUM, ATTRIBUTES_NUM))
    for classes in range(CLASS_NUM):
        norm = X[y == classes] - mean_by_class[classes]
        norm = norm.T
        scatter_within += norm @ norm.T

    # # Between-Class
    scatter_between = np.zeros((ATTRIBUTES_NUM, ATTRIBUTES_NUM))
    overall_mean = np.mean(X, axis=0)
    for classes in range(CLASS_NUM):
        norm = mean_by_class[classes, :] - overall_mean
        norm = norm.reshape((ATTRIBUTES_NUM, 1))
        scatter = norm @ norm.T
        scatter_between += EXAMPLE_PER_CLASS_NUM * scatter

    # Get subspace axis matrix.
    eig_mtx = create_new_axis(scatter_within, scatter_between)

    # Transform data to new subspace.
    X_lda = X @ eig_mtx

    return X_lda


def create_new_axis(scatter_within, scatter_between):
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
        - scatter_within - Scatter matrix within classes.
        - scatter_between - Scatter matrix between classes.

    Returns:
        - Subspace eig matrix with most relevant Eigenvectors.

    """
    # Eigenvector & Eigenvalues
    # They provide as with information about distortion of a linear transformation.
    # Eigenvector represents direction of the distortion.
    # Eigenvalues represents the scaling factor of the Eigenvectors.
    # Eigenvectors will determine the new axis of the new feature space.
    eig_val, eig_vec = np.linalg.eig(np.linalg.inv(scatter_within).dot(scatter_between))

    # Eig check - it assert pass, eig vec and val are correct
    for i in range(len(eig_val)):
        eigv = eig_vec[:, i].reshape(4, 1)
        np.testing.assert_array_almost_equal(np.linalg.inv(scatter_within).dot(scatter_between).dot(eigv),
                                             eig_val[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)

    # Selecting linear discriminants for future space
    # Eigenvectors forms new axes. Because LDA needs to reduce dimensions,
    # only N wanted best scoring eigenvectors are selected. Because Eigenvectors
    # only define direction, Eigenvalues will  represent reference for choosing
    # right axes.
    # Sorting Eigenvectors by Eigenvalues.
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Choose most relevant eigenvectors. To reduce dimension only most relevant
    # eigenvectors should be chosen.
    eig_val_sum = sum(eig_val)
    eig_acc_per = 0
    eig_acc_count = 0
    for i, j in enumerate(eig_pairs):
        eig_acc_per += j[0] / eig_val_sum
        eig_acc_count += 1

        if eig_acc_per >= EIG_ACCEPTANCE_PERCENTAGE:
            break

    eig_mtx = []
    for i in range(eig_acc_count):
        eig_mtx.append(eig_pairs[i][1])

    eig_mtx = np.array(eig_mtx).T

    return eig_mtx


def main():
    # Load dataset
    X = datasets.load_iris()

    X_lda = lda(X.data, X.target)

    return 0


if __name__ == '__main__':
    main()
