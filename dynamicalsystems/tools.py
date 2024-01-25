import numpy as np
import scipy


def F_all(u, M, system, jacobian, params):

    ''' solves system. State vector is composed
        by the variables and the entries in the direct
        and adjoint mappings '''
    
    dqdt  = system(u, params)
    J     = jacobian(u, params)
    dMdt  = np.dot(J, M)

    return dqdt, dMdt

def RK4var(x, dt, mat, system, jacobian,  params):
    
    K1, M1 = F_all( x, mat, system, jacobian, params)        
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


def solve_ode_LEs(N, N_trans, dt, u0, system, jacobian, params, norm_time=1, dim=3):
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
    print('Number of timesteps',N)
    print('Number of timesteps for Lyapunov exponents',N_test)
    
        
    T = np.arange(N+1) * dt
    Ttest = np.arange(1,int(N_test)+1) * dt 
    u = u0
    U = np.empty((T.size, u0.size))
    U[0] = u0
    
    N_test_norm = int(N_test/norm_time)
    
    # Lyapunov Exponents timeseries
    LE   = np.zeros((N_test_norm,dim))
    # Instantaneous Lyapunov Exponents timeseries
    IBLE   = np.zeros((N_test_norm,dim))

    #set random orthonormal Lyapunov vectors 
    np.random.seed(0)
    delta = scipy.linalg.orth(np.random.rand(dim,dim))  
    Q, R = qr_factorization(delta)
    delta = Q[:,:dim]
    
    for i in range(1, T.size):
        u_t, Mtemp = RK4var(u, dt, delta, system, jacobian, params)        
        u += u_t
        delta += Mtemp             
        Q, R = qr_factorization(delta)
        delta = Q[:,:dim]

        if i > N_trans:
            LE[i- N_trans-1]       = np.abs(np.diag(R))
            Jacobian       = jacobian(u, params)
            for j in range(dim):
                    IBLE[i- N_trans-1, j] = np.dot(Q[:,j].T, np.dot(Jacobian,Q[:,j]))
        U[i] = u
    
    LEs = np.cumsum(np.log(LE[:]),axis=0) / np.tile(Ttest[:],(dim,1)).T
    print(f'Lyapunov Exponents: {LEs[-1]}')
    return T, U, LEs, IBLE