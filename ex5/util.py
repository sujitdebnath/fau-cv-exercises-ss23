import numpy as np

def computeBilinerWeights(q):

    ## TODO 2.1
    ## - Compute bilinear weights for point q
    ## - Entry 0 weight for pixel (x, y)
    ## - Entry 1 weight for pixel (x + 1, y)
    ## - Entry 2 weight for pixel (x, y + 1)
    ## - Entry 3 weight for pixel (x + 1, y + 1)

    weights = [1, 0, 0, 0]

    return weights

def computeGaussianWeights(winsize, sigma):

    ## TODO 2.2
    ## - Fill matrix with gaussian weights
    ## - Note, the center is ((winSize.width - 1) / 2,winSize.height - 1) / 2)


    return np.array(weights)

def invertMatrix2x2(A):

    ## TODO 2.3
    ## - Compute the inverse of the 2 x 2 Matrix A


    return invA

