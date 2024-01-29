import numpy as np
import scipy
import itertools
import math


def F_all(u, M, system, jacobian, params):
    ''' solves system. State vector is composed
        by the variables and the entries in the direct
        and adjoint mappings '''

    dqdt = system(u, params)
    J = jacobian(u, params)
    dMdt = np.dot(J, M)

    return dqdt, dMdt


def RK4var(x, dt, mat, system, jacobian,  params):

    K1, M1 = F_all(x, mat, system, jacobian, params)
    K2, M2 = F_all(x + dt*K1/2.0, mat + dt*M1/2.0, system, jacobian,  params)
    K3, M3 = F_all(x + dt*K2/2.0, mat + dt*M2/2.0, system, jacobian,  params)
    K4, M4 = F_all(x + dt*K3, mat + dt*M3, system, jacobian,  params)

    A = np.array(dt * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0)
    B = np.array(dt * (M1/2.0 + M2 + M3 + M4/2.0) / 3.0)
    return A, B


def qr_factorization(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q

        norm = np.linalg.norm(v)
        Q[:, j] = v / norm
        R[j, j] = norm
    return Q, R


def solve_ode_LEs(N, N_trans, dt, u0, system, jacobian, params, norm_time=1, dim=3, clv_angles=False):
    """
        Solves the ODEs for N time steps starting from u0.
        Additionally it computes the Lyapunov spectrum 

        Args:
            N: number of time steps
            N: number of time steps in transient
            dt: timestep
            u0: initial condition
            params: parameters for ODE
    """

    N_test = N - N_trans
    print('Number of timesteps', N)
    print('Number of timesteps for Lyapunov exponents', N_test)

    T = np.arange(N+1) * dt
    Ttest = np.arange(1, int(N_test)+1) * dt
    u = u0
    U = np.empty((T.size, u0.size))
    U[0] = u0

    N_test_norm = int(N_test/norm_time)

    # Lyapunov Exponents timeseries
    LE = np.zeros((N_test_norm, dim))
    # Instantaneous Lyapunov Exponents timeseries
    IBLE = np.zeros((N_test_norm, dim))

    qq_t = np.zeros((dim, dim, N_test_norm))
    rr_t = np.zeros((dim, dim, N_test_norm))
    # set random orthonormal Lyapunov vectors
    np.random.seed(0)
    delta = scipy.linalg.orth(np.random.rand(dim, dim))
    Q, R = qr_factorization(delta)
    delta = Q[:, :dim]

    for i in range(1, T.size):
        u_t, Mtemp = RK4var(u, dt, delta, system, jacobian, params)
        u += u_t
        delta += Mtemp
        Q, R = qr_factorization(delta)
        delta = Q[:, :dim]

        if i > N_trans:
            LE[i - N_trans-1] = np.abs(np.diag(R))
            Jacobian = jacobian(u, params)
            rr_t[:, :, i - N_trans-1] = R
            qq_t[:, :, i - N_trans-1] = Q

            for j in range(dim):
                IBLE[i - N_trans-1,
                     j] = np.dot(Q[:, j].T, np.dot(Jacobian, Q[:, j]))
        U[i] = u

    LEs = np.cumsum(np.log(LE[:]), axis=0) / np.tile(Ttest[:], (dim, 1)).T
    print(f'Lyapunov Exponents: {LEs[-1]}')
    if clv_angles:
        thetas_clv = CLV_calculation(qq_t, rr_t, N_test_norm, dim)
        return T, U, LEs, IBLE, thetas_clv
    return T, U, LEs, IBLE


def normalize(M):
    ''' Normalizes columns of M individually '''
    nM = np.zeros(M.shape)  # normalized matrix
    nV = np.zeros(np.shape(M)[1])  # norms of columns

    for i in range(M.shape[1]):
        nV[i] = scipy.linalg.norm(M[:, i])
        nM[:, i] = M[:, i] / nV[i]

    return nM, nV


def timeseriesdot(x, y, multype):
    tsdot = np.einsum(multype, x, y.T)  # Einstein summation. Index i is time.
    return tsdot


def CLV_angles(clv, NLy):
    # calculate angles between CLVs
    thetas_num = int(np.math.factorial(NLy) /
                     (np.math.factorial(2) * np.math.factorial(NLy-2)))
    costhetas = np.zeros((clv[:, 0, :].shape[1], thetas_num))
    count = 0
    for subset in itertools.combinations(np.arange(NLy), 2):
        index1 = subset[0]
        index2 = subset[1]
        # For principal angles take the absolute of the dot product
        costhetas[:, count] = np.absolute(timeseriesdot(
            clv[:, index1, :], clv[:, index2, :], 'ij,ji->j'))
        count += 1
    thetas = 180. * np.arccos(costhetas) / math.pi

    return thetas


def CLV_calculation(QQ, RR, NLy, delta_dim, sampling = 10):
    print('start CLV calculation')
    QQ = QQ[:, :, ::sampling]
    RR = RR[:, :, ::sampling]
    NLy = int(NLy/sampling)-1
    tly = np.shape(QQ)[-1]

    # Calculation of CLVs
    # coordinates of CLVs in local GS vector basis
    C = np.zeros((NLy, NLy, tly))
    D = np.zeros((NLy, NLy, tly))  # diagonal matrix
    # coordinates of CLVs in physical space (each column is a vector)
    V = np.zeros((delta_dim, NLy, tly))

    # initialise components to I
    C[:, :, -1] = np.eye(NLy)
    D[:, :, -1] = np.eye(NLy)
    V[:, :, -1] = np.dot(np.real(QQ[:, :, -1]), C[:, :, -1])

    for i in reversed(range(tly-1)):
        C[:, :, i], D[:, :, i] = normalize(
            scipy.linalg.solve_triangular(np.real(RR[:, :, i]), C[:, :, i+1]))
        V[:, :, i] = np.dot(np.real(QQ[:, :, i]), C[:, :, i])

    timetot = np.shape(V)[-1]
    for i in range(NLy):
        for t in range(timetot):
            V[:, i, t] = V[:, i, t] / np.linalg.norm(V[:, i, t])
    thetas_clv = CLV_angles(V, NLy)

    return thetas_clv
