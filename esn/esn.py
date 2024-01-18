import numpy as np
from sklearn.linear_model import Ridge

import esn.generate_input_weights as generate_input_weights
import esn.generate_reservoir_weights as generate_reservoir_weights

class ESN:
    def __init__(
        self,
        reservoir_size,
        dimension,
        reservoir_connectivity=0,
        input_normalization=None,
        input_scaling=1.0,
        tikhonov=1e-9,
        spectral_radius=1.0,
        leak_factor=1.0,
        input_bias=np.array([]),
        output_bias=np.array([]),
        input_seeds=[None, None, None],
        reservoir_seeds=[None, None],
        verbose=True,
        r2_mode=False,
        input_only_mode=False,
        input_weights_mode="sparse_grouped",
        reservoir_weights_mode="erdos_renyi2",
    ):
        """Creates an Echo State Network with the given parameters
        Args:
            reservoir_size: number of neurons in the reservoir
            dimension: dimension of the state space of the input and output
                they must have the same size in order for the closed-loop to work
            reservoir_connectivity: connectivity of the reservoir weights,
                how many connections does each neuron have (on average)
            input_normalization: normalization applied to the input before activation
                tuple with (mean, norm) such that input u is updated as (u-mean)/norm
            input_scaling: scaling applied to the input weights matrix
            spectral_radius: spectral radius (maximum absolute eigenvalue)
                of the reservoir weights matrix
            leak_factor: factor for the leaky integrator
                if set to 1 (default), then no leak is applied
            input_bias: bias that is augmented to the input vector
            input_seeds: seeds to generate input weights matrix
            reservoir_seeds: seeds to generate reservoir weights matrix
        Returns:
            ESN object

        """
        self.verbose = verbose
        self.r2_mode = r2_mode
        self.input_only_mode = input_only_mode

        # Hyperparameters
        # these should be fixed during initialization and not changed since they affect
        # the matrix dimensions, and the matrices can become incompatible
        self.N_reservoir = reservoir_size
        self.N_dim = dimension
        self.leak_factor = leak_factor

        # Biases
        self.input_bias = input_bias
        self.output_bias = output_bias

        # Input normalization
        if not input_normalization:
            input_normalization = [None] * 2
            input_normalization[0] = np.zeros(self.N_dim)
            input_normalization[1] = np.ones(self.N_dim)

        self.input_normalization = input_normalization

        # Weights
        # the object should also store the seeds for reproduction
        # initialise input weights
        self.W_in_seeds = input_seeds
        self.W_in_shape = (
            self.N_reservoir,
            self.N_dim + len(self.input_bias)
        )
        # N_dim+length of input bias because we augment the inputs with a bias
        # if no bias, then this will be + 0
        self.input_weights_mode = input_weights_mode
        self.input_weights = self.generate_input_weights()
        self.input_scaling = input_scaling
        self.tikh = tikhonov
        # input weights are automatically scaled if input scaling is updated

        # initialise reservoir weights
        if not self.input_only_mode:
            self.reservoir_connectivity = reservoir_connectivity
            self.W_seeds = reservoir_seeds
            self.W_shape = (self.N_reservoir, self.N_reservoir)
            self.reservoir_weights_mode = reservoir_weights_mode
            valid_W = False
            while not valid_W:
                try:
                    self.reservoir_weights = self.generate_reservoir_weights()
                    valid_W = True
                except:
                    # perturb the seeds
                    for i in range(2):
                        if self.W_seeds[i]:
                            self.W_seeds[i] = self.W_seeds[i] + 1
                    valid_W = False
                    print("Not valid reservoir encountered, changing seed.")
            self.spectral_radius = spectral_radius
            # reservoir weights are automatically scaled if spectral radius is updated

        # initialise output weights
        self.W_out_shape = (self.N_reservoir + len(self.output_bias), self.N_dim)
        # N_reservoir+length of output bias because we augment the outputs with a bias
        # if no bias, then this will be + 0
        self.output_weights = np.zeros(self.W_out_shape)

    @property
    def reservoir_connectivity(self):
        return self.connectivity

    @reservoir_connectivity.setter
    def reservoir_connectivity(self, new_reservoir_connectivity):
        # set connectivity
        if new_reservoir_connectivity <= 0:
            raise ValueError("Connectivity must be greater than 0.")
        self.connectivity = new_reservoir_connectivity
        # regenerate reservoir with the new connectivity
        if hasattr(self, "W"):
            if self.verbose:
                print("Reservoir weights are regenerated for the new connectivity.")
            self.reservoir_weights = self.generate_reservoir_weights()
        return

    @property
    def leak_factor(self):
        return self.alpha

    @leak_factor.setter
    def leak_factor(self, new_leak_factor):
        # set leak factor
        if new_leak_factor < 0 or new_leak_factor > 1:
            raise ValueError("Leak factor must be between 0 and 1 (including).")
        self.alpha = new_leak_factor
   

    @property
    def tikhonov(self):
        return self.tikh

    @tikhonov.setter
    def tikhonov(self, new_tikhonov):
        # set tikhonov coefficient
        if new_tikhonov <= 0:
            raise ValueError("Tikhonov coefficient must be greater than 0.")
        self.tikh = new_tikhonov


    @property
    def input_normalization(self):
        return self.norm_in

    @input_normalization.setter
    def input_normalization(self, new_input_normalization):
        self.norm_in = new_input_normalization
        if self.verbose:
            print("Input normalization is changed, training must be done again.")

    @property
    def input_scaling(self):
        return self.sigma_in

    @input_scaling.setter
    def input_scaling(self, new_input_scaling):
        """Setter for the input scaling, if new input scaling is given,
        then the input weight matrix is also updated
        """
        if hasattr(self, "sigma_in"):
            # rescale the input matrix
            self.W_in = (1 / self.sigma_in) * self.W_in
        # set input scaling
        self.sigma_in = new_input_scaling
        if self.verbose:
            print("Input weights are rescaled with the new input scaling.")
        self.W_in = self.sigma_in * self.W_in
        return

    @property
    def spectral_radius(self):
        return self.rho

    @spectral_radius.setter
    def spectral_radius(self, new_spectral_radius):
        """Setter for the spectral_radius, if new spectral_radius is given,
        then the reservoir weight matrix is also updated
        """
        if hasattr(self, "rho"):
            # rescale the reservoir matrix
            self.W = (1 / self.rho) * self.W

        # set spectral radius
        self.rho = new_spectral_radius
        if self.verbose:
            print("Reservoir weights are rescaled with the new spectral radius.")
        self.W = self.rho * self.W
        return

    @property
    def input_weights(self):
        return self.W_in

    @input_weights.setter
    def input_weights(self, new_input_weights):
        # first check the dimensions
        if new_input_weights.shape != self.W_in_shape:
            raise ValueError(
                f"The shape of the provided input weights does not match with the network, {new_input_weights.shape} != {self.W_in_shape}"
            )

        # set the new input weights
        self.W_in = new_input_weights

        # set the input scaling to 1.0
        if self.verbose:
            print("Input scaling is set to 1, set it separately if necessary.")
        self.sigma_in = 1.0
        return

    @property
    def reservoir_weights(self):
        return self.W

    @reservoir_weights.setter
    def reservoir_weights(self, new_reservoir_weights):
        # first check the dimensions
        if new_reservoir_weights.shape != self.W_shape:
            raise ValueError(
                f"The shape of the provided reservoir weights does not match with the network,"
                f"{new_reservoir_weights.shape} != {self.W_shape}"
            )

        # set the new reservoir weights
        self.W = new_reservoir_weights

        # set the spectral radius to 1.0
        if self.verbose:
            print("Spectral radius is set to 1, set it separately if necessary.")
        self.rho = 1.0
        return

    @property
    def output_weights(self):
        return self.W_out

    @output_weights.setter
    def output_weights(self, new_output_weights):
        # first check the dimensions
        if new_output_weights.shape != self.W_out_shape:
            raise ValueError(
                f"The shape of the provided output weights does not match with the network,"
                f"{new_output_weights.shape} != {self.W_out_shape}"
            )
        # set the new reservoir weights
        self.W_out = new_output_weights
        return

    @property
    def input_bias(self):
        return self.b_in

    @input_bias.setter
    def input_bias(self, new_input_bias):
        self.b_in = new_input_bias
        return

    @property
    def output_bias(self):
        return self.b_out

    @output_bias.setter
    def output_bias(self, new_output_bias):
        self.b_out = new_output_bias
        return

    @property
    def sparseness(self):
        """Define sparseness from connectivity"""
        # probability of non-connections = 1 - probability of connection
        # probability of connection = (number of connections)/(total number of neurons - 1)
        # -1 to exclude the neuron itself
        return 1 - (self.connectivity / (self.N_reservoir - 1))

    def generate_input_weights(self):
        if self.input_weights_mode == "sparse_random":
            return generate_input_weights.sparse_random(
                self.W_in_shape, self.W_in_seeds
            )
        elif self.input_weights_mode == "sparse_grouped":
            return generate_input_weights.sparse_grouped(
                self.W_in_shape, self.W_in_seeds
            )
        elif self.input_weights_mode == "dense":
            return generate_input_weights.dense(self.W_in_shape, self.W_in_seeds)
        else:
            raise ValueError("Not valid input weights generator.")

    def generate_reservoir_weights(self):
        if self.reservoir_weights_mode == "erdos_renyi1":
            return generate_reservoir_weights.erdos_renyi1(
                self.W_shape, self.sparseness, self.W_seeds
            )
        if self.reservoir_weights_mode == "erdos_renyi2":
            return generate_reservoir_weights.erdos_renyi2(
                self.W_shape, self.sparseness, self.W_seeds
            )
        else:
            raise ValueError("Not valid reservoir weights generator.")

    def step(self, x_prev, u):
        """Advances ESN time step.
        Args:
            x_prev: reservoir state in the previous time step (n-1)
            u: input in this time step (n)
        Returns:
            x_next: reservoir state in this time step (n)
        """
        # normalise the input
        u_norm = (u - self.norm_in[0]) / self.norm_in[1]
        # we normalize here, so that the input is normalised
        # in closed-loop run too

        # augment the input with the input bias
        u_augmented = np.hstack((u_norm, self.b_in))

        # update the reservoir
        if self.input_only_mode:
            x_tilde = np.tanh(self.W_in.dot(u_augmented))
        else:
            x_tilde = np.tanh(self.W_in.dot(u_augmented) + self.W.dot(x_prev))

        # apply the leaky integrator
        x = (1 - self.alpha) * x_prev + self.alpha * x_tilde
        return x

    def open_loop(self, x0, U, P=None):
        """Advances ESN in open-loop.
        Args:
            x0: initial reservoir state
            U: input time series in matrix form (N_t x N_dim)

        Returns:
            X: time series of the reservoir states (N_t x N_reservoir)
        """
        N_t = U.shape[0]  # number of time steps

        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t + 1, self.N_reservoir))
        # N_t+1 because at t = 0, we don't have input

        # initialise with the given initial reservoir states
        X[0, :] = x0
        # X = [x0]
        # step in time
        for n in np.arange(1, N_t + 1):
            X[n] = self.step(X[n - 1, :], U[n - 1, :])

        return X

    def before_readout_r1(self, x):
        # augment with bias before readout
        return np.hstack((x, self.b_out))

    def before_readout_r2(self, x):
        # replaces r with r^2 if even, r otherwise
        x2 = x.copy()
        x2[1::2] = x2[1::2] ** 2
        return np.hstack((x2, self.b_out))

    @property
    def before_readout(self):
        if not hasattr(self, "_before_readout"):
            if self.r2_mode:
                self._before_readout = self.before_readout_r2
            else:
                self._before_readout = self.before_readout_r1
        return self._before_readout

    def closed_loop(self, x0, N_t):
        # @todo: make it an option to hold X or just x in memory
        """Advances ESN in closed-loop.
        Args:
            N_t: number of time steps
            x0: initial reservoir state
        Returns:
            X: time series of the reservoir states (N_t x N_reservoir)
            Y: time series of the output (N_t x N_dim)
        """
        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t + 1, self.N_reservoir))
        # create an empty matrix to hold the output states in time
        Y = np.empty((N_t + 1, self.N_dim))

        # initialize with the given initial reservoir states
        X[0, :] = x0

        # augment the reservoir states with the bias
        x0_augmented = self.before_readout(x0)

        # initialise with the calculated output states
        Y[0, :] = np.dot(x0_augmented, self.W_out)

        # step in time
        for n in range(1, N_t + 1):
            # update the reservoir with the feedback from the output
            X[n, :] = self.step(X[n - 1, :], Y[n - 1, :])

            # augment the reservoir states with bias
            x_augmented = self.before_readout(X[n, :])

            # update the output with the reservoir states
            Y[n, :] = np.dot(x_augmented, self.W_out)
        return X, Y

    def run_washout(self, U_washout):
        # Wash-out phase to get rid of the effects of reservoir states initialised as zero
        # initialise the reservoir states before washout
        x0_washout = np.zeros(self.N_reservoir)

        # let the ESN run in open-loop for the wash-out
        # get the initial reservoir to start the actual open/closed-loop,
        # which is the last reservoir state
        x0 = self.open_loop(x0=x0_washout, U=U_washout)[-1, :]
        return x0

    def open_loop_with_washout(self, U_washout, U):
        x0 = self.run_washout(U_washout)
        X = self.open_loop(x0=x0, U=U)
        return X

    def closed_loop_with_washout(self, U_washout, N_t):
        x0 = self.run_washout(U_washout)
        X, Y = self.closed_loop(x0=x0, N_t=N_t)
        return X, Y

    def solve_ridge(self, X, Y, tikh):
        """Solves the ridge regression problem
        Args:
            X: input data
            Y: output data
            tikh: weighing tikhonov coefficient that regularises L2 norm
        """
        reg = Ridge(alpha=self.tikh, fit_intercept=False)
        reg.fit(X, Y)
        W_out = reg.coef_.T
        return W_out

    def reservoir_for_train(self, U_washout, U_train):
        X_train = self.open_loop_with_washout(U_washout, U_train)

        # X_train is one step longer than U_train and Y_train, we discard the initial state
        X_train = X_train[1:, :]

        # augment with the bias
        N_t = X_train.shape[0]  # number of time steps

        if self.r2_mode:
            X_train2 = X_train.copy()
            X_train2[:, 1::2] = X_train2[:, 1::2] ** 2
            X_train_augmented = np.hstack((X_train2, self.b_out * np.ones((N_t, 1))))
        else:
            X_train_augmented = np.hstack((X_train, self.b_out * np.ones((N_t, 1))))

        return X_train_augmented

    def train(
        self,
        U_washout,
        U_train,
        Y_train,
        tikhonov=1e-12,
        train_idx_list=None,
    ):
        """Trains ESN and sets the output weights.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            Y_train: training output time series
            (list of time series if more than one trajectories)
            tikhonov: regularization coefficient
            train_idx_list: if list of time series, then which ones to use in training
                if not specified, all are used
        """
        # get the training input
        # this is the reservoir states augmented with the bias after a washout phase
        if isinstance(U_train, list):
            X_train_augmented = np.empty((0, self.W_out_shape[0]))
            if train_idx_list is None:
                train_idx_list = range(len(U_train))
            for train_idx in train_idx_list:
                X_train_augmented_ = self.reservoir_for_train(U_washout[train_idx], U_train[train_idx])
                X_train_augmented = np.vstack((X_train_augmented, X_train_augmented_))

            Y_train = [Y_train[train_idx] for train_idx in train_idx_list]
            Y_train = np.vstack(Y_train)
        else:
            X_train_augmented = self.reservoir_for_train(U_washout, U_train)

        # solve for W_out using ridge regression
        self.output_weights = self.solve_ridge(X_train_augmented, Y_train, self.tikh)
        return
