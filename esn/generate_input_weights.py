# Input weights generation methods
import numpy as np
from scipy.sparse import lil_matrix


def sparse_random(W_in_shape, W_in_seeds):
    """Create the input weights matrix
    Inputs are not connected, except for the parameters

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_input_bias + N_param_dim)
        seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])
    rnd2 = np.random.RandomState(W_in_seeds[2])

    # make W_in
    for j in range(W_in_shape[0]):
        rnd_idx = rnd0.randint(0, W_in_shape[1])
        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx] = rnd1.uniform(-1, 1)

    # input associated with system's bifurcation parameters are
    # fully connected to the reservoir states

    W_in = W_in.tocsr()

    return W_in


def sparse_grouped(W_in_shape, W_in_seeds):
    # The inputs are not connected but they are grouped within the matrix

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * (W_in_shape[1]) / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    W_in = W_in.tocsr()
    return W_in


def dense(W_in_shape, W_in_seeds):
    # The inputs are all connected

    rnd0 = np.random.RandomState(W_in_seeds[0])
    W_in = rnd0.uniform(-1, 1, W_in_shape)
    return W_in
