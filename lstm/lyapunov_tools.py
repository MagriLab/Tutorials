import numpy as np
import tensorflow as tf
import einops
import time
import h5py
import itertools
import math
from typing import Optional, List, Union, Tuple
import scipy

import sys
sys.path.append('../')
from dynamicalsystems.tools import qr_factorization

def compute_lyapunov_exp_lstm(test_window: np.ndarray, tf_lstm,
                         N: int, dim: int, le_dim: Optional[int] = None, dt=0.01, calc_clvs=False) -> np.ndarray:
    """Compute the Lyapunov exponents for a given lstm.

    Args:
        test_window (np.ndarray): starting point of computation
        lstm (lstm): The lstm to use for the computation.
        args (Dict[str, Any]): Dictionary of lstm parameters.
        N (int): Number of samples to use for the computation.
        dim (int): Dimension of the data.
        le_dim (int): Dimension of the Lyapunov exponent.
        idx_lst (List[int], optional): List of indices to use for the computation. If not specified, all indices will be used.
        save_path (Path, optional): If specified, the Lyapunov exponents will be saved to this path.

    Returns:
        np.ndarray: A list of Lyapunov exponents, one for each index in `idx_lst` (or all indices if `idx_lst` is not specified).
    """

    global window_size, sysdim #temporary workaround
    
    if le_dim == None:
        le_dim = dim
    n_cell = tf_lstm.hidden_size
    window_size = tf_lstm.window_size
    sysdim =dim
    upsampling = 1 
    if le_dim > n_cell:
        le_dim=n_cell
        
    lstm = tf_lstm.model
    norm_time = 1
    Ntransient = max(int(N/100), window_size+2)
    N_test = N - Ntransient
    Ttot = np.arange(int(N_test/norm_time)) * (upsampling*dt) * norm_time
    N_test_norm = int(N_test/norm_time)

    # Lyapunov Exponents timeseries
    LE = np.zeros((N_test_norm, le_dim))
    # q and r matrix recorded in time
    qq_t = np.zeros((n_cell+n_cell, le_dim, N_test_norm))
    rr_t = np.zeros((le_dim, le_dim, N_test_norm))
    np.random.seed(1)
    delta = scipy.linalg.orth(np.random.rand(n_cell+n_cell, le_dim))
    q, r = qr_factorization(delta)
    delta = q[:, :le_dim]

    # initialize lstm and test window
    u_t = test_window[:, 0, :]
    h = tf.Variable(lstm.layers[0].get_initial_state(test_window)[0], trainable=False)
    c = tf.Variable(lstm.layers[0].get_initial_state(test_window)[1], trainable=False)
    pred = np.zeros(shape=(N, dim))
    pred[0, :] = u_t
    # prepare h,c and c from first window
    for i in range(1, window_size+1):
        u_t = test_window[:, i-1, :]
        u_t, h, c = tf_lstm.lstm_step(u_t, h, c)
        pred[i, :] = u_t
    i = window_size
    jacobian, u_t, h, c = step_and_jac(u_t, h, c, tf_lstm, lstm)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)
    q, r = qr_factorization(delta)
    delta = q[:, :le_dim]
    # compute delta on transient
    for i in range(window_size+1, Ntransient):
        jacobian, u_t, h, c = step_and_jac_analytical(u_t, h, c, lstm, i)
        pred[i, :] = u_t
        delta = np.matmul(jacobian, delta)

        if i % norm_time == 0:
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

    print('Finished on Transient')
    # compute lyapunov exponent based on qr decomposition
    start_time = time.time()
    for i in range(Ntransient, N):
        jacobian, u_t, h, c = step_and_jac_analytical(u_t, h, c, lstm, i)
        indx = i-Ntransient
        pred[i, :] = u_t
        delta = np.matmul(jacobian, delta)
        if i % norm_time == 0:
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

            rr_t[:, :, indx] = r
            qq_t[:, :, indx] = q
            LE[indx] = np.abs(np.diag(r[:le_dim, :le_dim]))

            if i % 10000 == 0:
                print(f'Inside closed loop i = {i}, Time: {time.time()-start_time}')
                start_time = time.time()
                if indx != 0:
                    lyapunov_exp = np.cumsum(np.log(LE[1:indx]), axis=0) / np.tile(Ttot[1:indx], (le_dim, 1)).T
                    print(f'Lyapunov exponents: {lyapunov_exp[-1] } ')


    lyapunov_exp = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (le_dim, 1)).T
    print(f'Final Lyapunov exponents: {lyapunov_exp[-1] } ')
    if calc_clvs=True:
        thetas_clv, il, D = CLV_calculation(qq_t, rr_t, le_dim, n_cell*2, dt=0.01, fname=None)
        return thetas_clv, lyapunov_exp
    return lyapunov_exp



def step_and_jac(u_t_in, h, c, tf_lstm, lstm):
    """advances LSTM by one step and computes the Jacobian

    Args:
        u_t_in (tf.EagerTensor): differential equation at time t
        h (tf.EagerTensor): LSTM hidden state at time t
        c (tf.EagerTensor): LSTM cell state at time t
        lstm (keras.Sequential): trained LSTM
        idx (int): index of current iteration

    Returns:
        u_t_in (tf.EagerTensor): coupled Jacobian at time t
        u_t_out (tf.EagerTensor): LSTM prediction at time t+1
        h_new (tf.EagerTensor): LSTM hidden state at time t+1
        c_new (tf.EagerTensor): LSTM cell state at time t+1
    """
    cell_dim = lstm.layers[1].get_weights()[0].shape[0]
    with tf.GradientTape(persistent=True) as tape_h:
        tape_h.watch(h)
        with tf.GradientTape(persistent=True) as tape_c:
            tape_c.watch(c)
            u_t_out, h_new, c_new = tf_lstm.lstm_step(u_t_in, h, c)
            Jac_c_new_c = tf.reshape(tape_c.jacobian(c_new, c), shape=(cell_dim, cell_dim))
            Jac_h_new_c = tf.reshape(tape_c.jacobian(h_new, c), shape=(cell_dim, cell_dim))
        Jac_h_new_h = tf.reshape(tape_h.jacobian(h_new, h), shape=(cell_dim, cell_dim))
        Jac_c_new_h = tf.reshape(tape_h.jacobian(c_new, h), shape=(cell_dim, cell_dim))

    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                    tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)

    return Jac, u_t_out, h_new, c_new



def step_and_jac_analytical(u_t, h, c, lstm,  idx):
    """advances LSTM by one step and computes the Jacobian

    Args:
        u_t (tf.EagerTensor): differential equation at time t
        h (tf.EagerTensor): LSTM hidden state at time t
        c (tf.EagerTensor): LSTM cell state at time t
        lstm (keras.Sequential): trained LSTM
        idx (int): index of current iteration

    Returns:
        u_t (tf.EagerTensor): coupled Jacobian at time t
        u_t_out (tf.EagerTensor): LSTM prediction at time t+1
        h_new (tf.EagerTensor): LSTM hidden state at time t+1
        c_new (tf.EagerTensor): LSTM cell state at time t+1
    """
    n_cell = lstm.layers[1].get_weights()[0].shape[0]
    cell_dim = n_cell
    if idx > window_size:  # for correct Jacobian, must multiply W in the beginning
        u_t = tf.reshape(tf.matmul(h, lstm.layers[1].get_weights()[
            0]) + lstm.layers[1].get_weights()[1], shape=(1, sysdim))
        u_t_temp = u_t
       
    z = tf.keras.backend.dot(u_t, lstm.layers[0].cell.kernel)
    z += tf.keras.backend.dot(h, lstm.layers[0].cell.recurrent_kernel)
    z = tf.keras.backend.bias_add(z, lstm.layers[0].cell.bias)

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c_tilde = tf.tanh(z2)
    i_c_tilde = i * c_tilde
    c_new = f * c + i_c_tilde
    o = tf.sigmoid(z3)

    h_new = o * tf.tanh(c_new)

    Jac_z_h = tf.transpose(
        tf.matmul(lstm.layers[1].get_weights()[0],
                  lstm.layers[0].cell.kernel) + lstm.layers[0].cell.recurrent_kernel)
    Jac_i_z = einops.rearrange(tf.linalg.diag(i*(1-i)), '1 i j -> i j')
    Jac_i_h = tf.matmul(Jac_i_z, Jac_z_h[:cell_dim, :])
    Jac_f_h = tf.matmul(einops.rearrange(tf.linalg.diag(f*(1-f)), '1 i j -> i j'), Jac_z_h[cell_dim:2*cell_dim, :])
    Jac_o_h = tf.matmul(einops.rearrange(tf.linalg.diag(o*(1-o)), '1 i j -> i j'), Jac_z_h[3*cell_dim:4*cell_dim, :])
    Jac_c_t_h = tf.matmul(
        tf.reshape(tf.linalg.diag(1 - c_tilde ** 2),
                   shape=(cell_dim, cell_dim)),
        Jac_z_h[2 * cell_dim: 3 * cell_dim, :])
    Jac_i_c_tilde = (Jac_c_t_h * tf.transpose(i) + Jac_i_h*tf.transpose(c_tilde))
    Jac_c_new_c = tf.reshape(tf.linalg.diag(f), shape=(cell_dim, cell_dim))
    Jac_h_new_c = tf.reshape(tf.linalg.diag(o * (1 - tf.tanh(c_new)**2)), shape=(cell_dim, cell_dim)) * Jac_c_new_c
    Jac_c_new_h = Jac_i_c_tilde + Jac_f_h * tf.transpose(c)
    Jac_h_new_h = (tf.matmul(einops.rearrange(tf.linalg.diag(1 - tf.tanh(c_new)**2), '1 i j -> i j'),
                   Jac_c_new_h) * tf.transpose(o) + Jac_o_h*tf.transpose(tf.tanh(c_new)))
    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                     tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)
    return Jac, u_t_temp, h_new, c_new

def normalize(M):
    ''' Normalizes columns of M individually '''
    nM = np.zeros(M.shape) # normalized matrix
    nV = np.zeros(np.shape(M)[1]) # norms of columns

    for i in range(M.shape[1]):
        nV[i] = scipy.linalg.norm(M[:,i])
        nM[:,i] = M[:,i] / nV[i]

    return nM, nV

def timeseriesdot(x,y, multype):
    tsdot = np.einsum(multype,x,y.T) #Einstein summation. Index i is time.
    return tsdot

def CLV_angles(clv, NLy):
    #calculate angles between CLVs
    thetas_num = int(np.math.factorial(NLy) / (np.math.factorial(2) * np.math.factorial(NLy-2)))
    costhetas = np.zeros((clv[:,0,:].shape[1],thetas_num))
    count = 0
    for subset in itertools.combinations(np.arange(NLy), 2):
        index1 = subset[0]
        index2 = subset[1]
        #For principal angles take the absolute of the dot product
        costhetas[:,count] = np.absolute(timeseriesdot(clv[:,index1,:],clv[:,index2,:],'ij,ji->j'))
        count+=1
    thetas = 180. * np.arccos(costhetas) / math.pi

    return thetas


def CLV_calculation(QQ, RR, NLy, n_cells_x2, dt=0.01, fname=None):
    """
    Calculates the Covariant Lyapunov Vectors (CLVs) using the Ginelli et al, PRL 2007 method.

    Args:
    - QQ (numpy.ndarray): matrix containing the timeseries of Gram-Schmidt vectors (shape: (n_cells_x2,NLy,tly))
    - RR (numpy.ndarray): matrix containing the timeseries of upper-triangualar R  (shape: (NLy,NLy,tly))
    - NLy (int): number of Lyapunov exponents
    - n_cells_x2 (int): dimension of the hidden states
    - dt (float): integration time step
    Returns:
    - nothing
    """
    tly = np.shape(QQ)[-1]
    su = int(tly / 10)
    sd = int(tly / 10)
    s  = su          # index of spinup time
    e  = tly+1 - sd  # index of spindown time
    tau = int(dt/dt)     #time for finite-time lyapunov exponents

    #Calculation of CLVs
    C = np.zeros((NLy,NLy,tly))  # coordinates of CLVs in local GS vector basis
    D = np.zeros((NLy,tly))  # diagonal matrix
    V = np.zeros((n_cells_x2,NLy,tly))  # coordinates of CLVs in physical space (each column is a vector)

    # FTCLE
    il  = np.zeros((NLy,tly+1)) #Finite-time lyapunov exponents along CLVs

    # initialise components to I
    C[:,:,-1] = np.eye(NLy)
    D[:,-1]   = np.ones(NLy)
    V[:,:,-1] = np.dot(np.real(QQ[:,:,-1]), C[:,:,-1])

    for i in reversed(range( tly-1 ) ):
        C[:,:,i], D[:,i]        = normalize(scipy.linalg.solve_triangular(np.real(RR[:,:,i]), C[:,:,i+1]))
        V[:,:,i]                = np.dot(np.real(QQ[:,:,i]), C[:,:,i])

    # FTCLE computations
    for j in np.arange(D.shape[1]): #time loop
        il[:,j] = -(1./dt)*np.log(D[:,j])
        

    #normalize CLVs before measuring their angles.
    timetot = np.shape(V)[-1]

    for i in range(NLy):
        for t in range(timetot-1):
            V[:,i,t] = V[:,i,t] / np.linalg.norm(V[:,i,t])
            

    thetas_clv = CLV_angles(V, NLy)

    '''Save the data to hdf5 file'''
    if fname is not None:
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('thetas_clv',       data=thetas_clv)
            hf.create_dataset('FTCLE',            data=il)
            
    return thetas_clv, il, D