# Reservoir weights generation methods
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs


def erdos_renyi1(W_shape, sparseness, W_seeds):
    """Create the reservoir weights matrix according to Erdos-Renyi network
    Args:
        seeds: a list of seeds for the random generators;
            one for the connections, one for the uniform sampling of weights
    Returns:
        W: sparse matrix containing reservoir weights
    """
    # set the seeds
    rnd0 = np.random.RandomState(W_seeds[0])  # connection rng
    rnd1 = np.random.RandomState(W_seeds[1])  # sampling rng

    # initialize with zeros
    W = np.zeros(W_shape)

    # generate a matrix sampled from the uniform distribution (0,1)
    W_connection = rnd0.rand(W_shape[0], W_shape[1])

    # generate the weights from the uniform distribution (-1,1)
    W_weights = rnd1.uniform(-1, 1, W_shape)

    # replace the connections with the weights
    W = np.where(W_connection < (1 - sparseness), W_weights, W)
    # 1-sparseness is the connection probability = p,
    # after sampling from the uniform distribution between (0,1),
    # the probability of being in the region (0,p) is the same as having probability p
    # (this is equivalent to drawing from a Bernoulli distribution with probability p)

    W = csr_matrix(W)

    # find the spectral radius of the generated matrix
    # this is the maximum absolute eigenvalue
    rho_pre = np.abs(sparse_eigs(W, k=1, which="LM", return_eigenvectors=False))[0]

    # first scale W by the spectral radius to get unitary spectral radius
    W = (1 / rho_pre) * W

    return W


def erdos_renyi2(W_shape, sparseness, W_seeds):
    prob = 1 - sparseness

    # set the seeds
    rnd0 = np.random.RandomState(W_seeds[0])  # connection rng
    rnd1 = np.random.RandomState(W_seeds[1])  # sampling rng

    # initialize with zeros
    W = np.zeros(W_shape)
    for i in range(W_shape[0]):
        for j in range(W_shape[1]):
            b = rnd0.random()
            if (i != j) and (b < prob):
                W[i, j] = rnd1.random()

    W = csr_matrix(W)

    # find the spectral radius of the generated matrix
    # this is the maximum absolute eigenvalue
    rho_pre = np.abs(sparse_eigs(W, k=1, which="LM", return_eigenvectors=False))[0]

    # first scale W by the spectral radius to get unitary spectral radius
    W = (1 / rho_pre) * W
    return W
