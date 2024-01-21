import numpy as np


def roessler(u, params):
    a, b, c = params
    x, y, z = u
    return np.array([-y - z, x + a*y, b + z*(x-c)])


def rosjac(u, params):
    a, b, c = params  # Unpack the constants vector
    x, y, z = u  # Unpack the state vector

    # Jacobian
    J = np.array([[0, -1,  -1],
                  [1,  a,   0],
                  [z,  0, x-c]])
    return J


def lorenz63(u, params):
    beta, rho, sigma = params
    x, y, z = u
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])


def lorenz63jac(u, params):
    beta, rho, sigma = params  # Unpack the constants vector
    x, y, z = u  # Unpack the state vector

    # Jacobian
    J = np.array([[-sigma, sigma,     0],
                  [rho-z,    -1,    -x],
                  [y,     x, -beta]])
    return J


def lorenz96(x, p):
    return np.roll(x,1) * (np.roll(x,-1) - np.roll(x,2)) - x + p

def lorenz96jac(x, p):
    D = len(x)
    J = np.zeros((D,D), dtype='float')
    for i in range(D):
        J[i,(i-1)%D] =  x[(i+1)%D] - x[(i-2)%D]
        J[i,(i+1)%D] =  x[(i-1)%D]
        J[i,(i-2)%D] = -x[(i-1)%D]
        J[i,i] = -1.0
    return J

def RK4(ddt, u0, T, params):
    u = np.empty((T.size, u0.size))
    u[0] = u0
    der = np.empty((T.size, u0.size))
    for i in range(1, T.size):
        delta_t = (T[i] - T[i-1])
        K1 = ddt(u[i-1], params)
        K2 = ddt(u[i-1] + delta_t*K1/2.0, params)
        K3 = ddt(u[i-1] + delta_t*K2/2.0, params)
        K4 = ddt(u[i-1] + delta_t*K3, params)
        u[i:] = u[i-1] + np.array(delta_t * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0)
        der[i-1] = K1
    return u, der


def solve_rk4(function, N, dt, u0, params):
    T = np.arange(N+1) * dt
    U, rhs = RK4(function, u0, T, params)
    return T, U, rhs
