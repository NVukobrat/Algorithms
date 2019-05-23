import numpy as np


def svm():
    """
    Support Vector Machines (SVM) comes from the family
    of the supervised machine learning algorithms. It
    represents a classifier which works by defining
    hyperplane which separates classes from the dataset.

    The hyperplane represents a line that separates two
    or more classes in the multidimensional space. In the
    two dimensional space, a hyperplane is in the simple
    line shape. When there are three, or more dimensions,
    this hyperplane can form curved/circular and more
    complex shapes that better separates data.

    To establish a best-fit hyperplane, data should be
    augmented (mapped) to the higher dimensions. This is
    an expensive process and because of that, a trade-off
    must be made. To make a mentioned trade-off, SVM has
    tuning parameters: kernel, regularization, gamma,
    and margin.
    """
    pass


def kernel(kernel_type, x1, x2, dimension=2.0, sigma=1.0):
    """
    Kernel play a role in transforming data into higher
    dimensions. When there is a case of the linear problem,
    the linear kernel does this transformation. It uses
    linear algebra to map vectors (data samples) to linear
    future space.

    From the other side, polynomial kernel (or exponential)
    transforms data to higher dimensions, and by that,
    most of the time gives better separation between data
    points. A polynomial kernel is also called kernel trick.
    """
    if kernel_type == 'linear':
        return __linear_kernel(x1, x2)
    elif kernel_type == 'polynomial':
        return __polykernel(x1, x2, dimension)
    elif kernel_type == 'gauss':
        return __gauss_kernel(x1, x2, sigma)
    else:
        raise Exception("Invalid kernel type: {}".format(kernel_type))


def __linear_kernel(x1, x2):
    """
    Linear kernel.

    Linear kernel applies linear function. Using this kernel, only
    linear models could be trained.
    """
    return np.dot(x1, x2.T)


def __polykernel(x1, x2, dimension):
    """
    Polynomial kernel.

    The Polynomial kernel maps inputs to the higher dimensions. It
    allows learning a non-linearity separations. 
    """
    return (__linear_kernel(x1, x2)) ** dimension


def __gauss_kernel(x1, x2, sigma):
    """
    Gaussina (or Radial Basis Function, RBF) kernel.

    The Gaussian kernel computed with a support vector is an
    exponentially decaying function in the input feature space,
    the maximum value of which is attained at the support vector
    and which decays uniformly in all directions around the
    support vector, leading to hyper-spherical contours of the
    kernel function.

    K(x,y) = exp(− (x − y) / 2 * sigma ^ 2)
    """
    return np.exp(- (np.linalg.norm(x1 - x2, 2)) ** 2 / (2 * sigma ** 2))


def regularization():
    """
    The regularization parameter represents the degree
    of importance of misclassified samples.

    When regularization value grows, hyperplane allows
    less misclassified samples. In this case, the
    optimization will choose a smaller margin so less
    wrongly classified samples are allowed. On the other
    hand, when regularization value tends to be lower,
    the optimizer will look for the larger margin, even
    when there are a lot of misclassified samples.
    """
    pass


def gamma():
    """
    The gamma parameter defines how each sample can
    influence in the process of defining hyperplane.

    With low values of gamma, samples that are far
    away will have an influence on the hyperplane.
    This means that a larger number of samples will
    influence on the separation. Conversely, high
    gamma values, use only samples that are closest
    to the current hyperplane. By this, a small number
    of samples influence the separation.
    """
    pass


def margin():
    """
    The margin represents the separation of the
    hyperplane from the closest sample points.

    The main point of SVM is to find margin that
    will give the best separation between classes.
    All of the mentioned parameters have their
    unique influence on the final size of the
    margin, and by that on the final position of
    the hyperplane.
    """
    pass
