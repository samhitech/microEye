import numpy as np


def expand_image(image: np.ndarray, shape):
    '''
    Expand an image to a specified shape by padding with zeros.

    Parameters
    ----------
    image : np.ndarray
        Image to be expanded
    shape : tuple
        Desired shape

    Returns
    -------
    np.ndarray
        Expanded image
    '''
    if image.shape == shape:
        return image
    else:
        ret = np.zeros(shape)
        ret[:image.shape[0], :image.shape[1]] = image
        return ret

def match_shape(image1: np.ndarray, image2: np.ndarray):
    '''
    Match the shape of two images by expanding the smaller one.

    Parameters
    ----------
    image1 : np.ndarray
        First image
    image2 : np.ndarray
        Second image
    '''
    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape))

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = expand_image(image1, shape)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = expand_image(image2, shape)

    return image1, image2

def checker_pairs(image: np.ndarray):
    '''
    Extracts checkerboard pairs from an image.

    Parameters
    ----------
    image : np.ndarray
        Image to extract checkerboard pairs from
    '''
    shape = image.shape
    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    odd = image[odd_index[0], :][:, odd_index[1]]
    even = image[even_index[0], :][:, even_index[1]]
    odd, even = match_shape(odd, even)

    oddeven = image[odd_index[0], :][:, even_index[1]]
    evenodd = image[even_index[0], :][:, odd_index[1]]
    oddeven, evenodd = match_shape(oddeven, evenodd)

    return odd, even, oddeven, evenodd
