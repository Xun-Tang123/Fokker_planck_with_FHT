import numpy as np

def fourier_function(x, deg):
    """
    Generate an array based on the input x and degree deg as described.

    Parameters:
    - x (numpy.ndarray): a batch of real scalar inputs with shape [batch_size]
    - deg (int): a degree parameter

    Returns:
    - numpy.ndarray: an array with shape [batch_size, 2*deg + 1] containing
                     [1, sin(pi*x), cos(pi*x), ..., sin(deg*pi*x), cos(deg*pi*x)] for each input in x
    """

    # Pre-allocate result array
    result = np.empty((x.shape[0], 2 * deg + 1))
    result[:, 0] = 1/np.sqrt(2)  # Setting the first column to one

    values = np.arange(1, deg + 1)

    # Compute the sines and cosines
    sines = np.sin(np.pi * x[:, np.newaxis] * values[np.newaxis, :])
    cosines = np.cos(np.pi * x[:, np.newaxis] * values[np.newaxis, :])

    # Fill the result array without reshaping by slicing
    result[:, 1::2] = sines
    result[:, 2::2] = cosines

    return result


def fourier_custom(y_column, fourier_modes):
    """
    Compute a custom Fourier sketching on a given column of data based on specified modes.

    Parameters:
    - y_column (numpy.ndarray): A 2D numpy array representing a column of data to be transformed.
                                Shape: (N, P), where N is the number of samples, and P is the dimensionality of each sample.
    - fourier_modes (numpy.ndarray): A 2D numpy array specifying the Fourier modes. Shape: (M, P),
                                     where P denotes the modes applied to each dimension of y_column, and M is the number
                                     of Fourier polynomials that are considered.

    Returns:
    - numpy.ndarray: The transformed y_column according to the given Fourier modes.

    Notes:
    - The Fourier sketching is customized based on the values in `fourier_modes`. Some basic fact:
        1. If a mode is 0, the transformation returns ones.
        2. If a mode is odd, a sine transformation is applied.
        3. If a mode is even (and non-zero), a cosine transformation is applied.
        4. The Fourier degree is based on the magnitude of fourier_modes.
    """

    # Determine frequency and transformation type
    deg = np.where(fourier_modes == 0, 0, (fourier_modes + 1) // 2)
    if_sin = np.where(fourier_modes == 0, -1, (fourier_modes % 2) == 1)

    # Initialize result as ones with the same shape as y_column
    result = np.ones([y_column.shape[0], fourier_modes.shape[0]])
    c_mode = np.ones(fourier_modes.shape[0])  # Collecting the coefficient

    # Apply the custom Fourier transformation
    for dim in range(y_column.shape[1]):
        for mode in range(fourier_modes.shape[0]):
            if if_sin[mode, dim] == -1:
                c_mode[mode] *= 1/np.sqrt(2)
            else:
                transformation = np.sin if if_sin[mode, dim] else np.cos
                result[:, mode] *= transformation(np.pi * deg[mode, dim] * y_column[:, dim])

    for mode in range(fourier_modes.shape[0]):
        result[:, mode] *= c_mode[mode]
    return result