from functools import partial
from itertools import product
import numpy as np
import skopt
from skopt.space import Integer, Real

from esn.esn import ESN
from esn.utils import errors, reverse_scalers


def set_ESN(my_ESN, param_names, param_scales, params):
    # set the ESN with the new parameters
    for param_name in set(param_names):
        # get the unique strings in the list with set
        # now the indices of the parameters with that name
        # (because ESN has attributes that are set as arrays when there are more than one parameters
        # and not single scalars)
        param_idx_list = np.where(np.array(param_names) == param_name)[0]

        new_param = np.zeros(len(param_idx_list))
        for new_idx in range(len(param_idx_list)):
            # rescale the parameters according to the given scaling
            param_idx = param_idx_list[new_idx]
            reverse_scaler = getattr(reverse_scalers, param_scales[param_idx])
            new_param[new_idx] = reverse_scaler(params[param_idx])

        if len(param_idx_list) == 1:
            new_param = new_param[0]

        setattr(my_ESN, param_name, new_param)
    return

def create_search_grid(n_param, n_grid, grid_range):
    """Generates a grid for the given grid ranges and number of grid points
    Works for any number of parameters
    Needs return a list of lists for skopt minimize
    Args:
        n_grid: number of grid points for each parameter
        grid_range: search range for each parameter
    Returns:
        search_grid
    """
    # initialise list of np.arrays that holds the range of each parameter
    param_grid = [None] * n_param
    for param_idx in range(n_param):
        param_grid[param_idx] = np.linspace(*grid_range[param_idx], n_grid[param_idx])

    # create the search grid from the parameter grids
    search_grid = product(*param_grid, repeat=1)

    # itertools.product returns a list of tuples
    # turn it to a list of lists
    search_grid = [list(search_list) for search_list in search_grid]

    return search_grid


def create_search_space(n_param, grid_range, param_names):
    search_space = [None] * n_param
    for param_idx in range(n_param):
        if param_names[param_idx] == "tikhonov":
            search_space[param_idx] = Integer(
                *grid_range[param_idx], name=param_names[param_idx], dtype=np.int32
            )
        else:
            search_space[param_idx] = Real(
                *grid_range[param_idx], name=param_names[param_idx]
            )

    return search_space


def run_gp_optimization(val_fun, search_space, n_calls, n_initial_points, rand_state, search_grid=None):
    print(f"Input to run_gp:{ search_space, n_calls, n_initial_points, rand_state}")
    # Bayesian Optimization
    res = skopt.gp_minimize(
        val_fun,  # the function to minimize
        search_space,  # the bounds on each dimension of params
        n_calls=n_calls,  # total number of evaluations of f
        x0=search_grid,
        n_initial_points=n_initial_points,  # the number of additional random initialization points
        random_state=0,  # seed
        noise=1e-10,
        n_jobs=1,
    )
    return res


def loop(
    params,
    param_names,
    param_scales,
    ESN_dict,
    U_washout_train,
    U_train,
    U_val,
    Y_train,
    Y_val,
    n_folds,
    n_realisations,
    N_washout,
    N_val,
    N_trans=0,
    error_measure=errors.rmse,
):
    # initialize a base ESN object with unit input scaling and spectral radius
    # seeds not given, so the random generator creates a different seed each run
    global run_idx
    run_idx += 1
    print("--NEW RUN--", run_idx)

    realisation_error = np.zeros(n_realisations)
    for real_idx in range(n_realisations):
        print("Realisation:", real_idx)
        my_ESN = ESN(
            **ESN_dict,
            verbose=False,
        )

        for param_name, param, param_scale in zip(param_names, params, param_scales):
            reverse_scaler = getattr(reverse_scalers, param_scale)
            print(param_name, reverse_scaler(param))
            if not hasattr(my_ESN, param_name):
                raise ValueError(
                    f"Trying to set a non-existing hyperparameter, {param_name}"
                )

        print("\n")
        # avoid setting some non-existing hyperparameter
        # because setattr won't throw an error, but
        # will simply create a new attribute with that name

        # set the ESN with the given parameters
        set_ESN(my_ESN, param_names, param_scales, params)

        # train ESN
        my_ESN.train(
            U_washout_train,
            U_train,
            Y_train
        )

        # divide test set in intervals and predict
        if not isinstance(U_val, list):
            U_val = [U_val]
            Y_val = [Y_val]

        val_idx_list = range(len(U_val))

        val_error = np.zeros(len(val_idx_list))

        for val_idx_idx, val_idx in enumerate(val_idx_list):
            # set the time delay for rijke esn

            print("Val regime:", val_idx_idx)

            # validate with different folds
            fold_error = np.zeros(n_folds)

            for fold in range(n_folds):
                # select washout and validation
                # start_step = fold * (N_val-N_washout)
                start_step = np.random.randint(
                    len(U_val[val_idx]) - (N_washout + N_val) - N_trans
                )
                # print("Start_step:", start_step, len(U_val[val_idx]) , N_washout, N_val)
                U_washout_fold = U_val[val_idx][
                    start_step: start_step + N_washout
                ].copy()

                Y_val_fold = Y_val[val_idx][
                    start_step + N_washout: start_step + N_washout + N_val
                ].copy()
                # predict output validation in closed-loop
                _, Y_val_pred = my_ESN.closed_loop_with_washout(
                    U_washout=U_washout_fold,
                    N_t=N_val
                )
                Y_val_pred = Y_val_pred[1:, :]

                # compute error
                fold_error[fold] = error_measure(
                    Y_val_fold, Y_val_pred
                )
                # print("Fold Prediction", Y_val_fold[N_trans:N_trans+10], Y_val_pred[N_trans:N_trans+10])
                print("Fold:", fold, ", fold error: ", fold_error[fold])
            # average over intervals
            val_error[val_idx_idx] = np.mean(fold_error)
            print("Val regime error:", val_error[val_idx_idx])
        # sum over validation regimes
        realisation_error[real_idx] = np.sum(val_error)
        print("Realisation error:", realisation_error[real_idx])
        print("\n")
    # average over realisations
    error = np.mean(realisation_error)
    print("Run", run_idx, "error:", error)
    print("\n")
    return error


def validate(
    grid_range,
    param_names,
    param_scales,
    n_calls,
    n_initial_points,
    ESN_dict,
    U_washout_train,
    U_train,
    U_val,
    Y_train,
    Y_val,
    n_folds,
    n_realisations,
    N_washout_steps,
    N_val_steps,
    N_transient_steps=0,
    n_grid=None,
    random_seed=20,
    error_measure=errors.rmse,
):

    n_param = len(param_names)  # number of parameters

    # ranges for hyperparameters
    search_space = create_search_space(n_param, grid_range, param_names)

    # grid points to start the search
    if n_grid:
        search_grid = create_search_grid(n_param, n_grid, grid_range)
        n_calls = len(search_grid) + n_calls
    else:
        search_grid = None

    # initialize dictionary to hold the minimum parameters and errors
    n_top = 5
    min_dict = {
        "f": np.zeros(n_top),
    }
    for param_name in param_names:
        min_dict[param_name] = np.zeros(n_top)

    global run_idx
    run_idx = 0
    # create the validation function
    # skopt minimize takes functions with only parameters as args
    # we create a partial function passing our ESN and other params
    # which we can then access for training/validation
    val_fun = partial(
        loop,
        param_names=param_names,
        param_scales=param_scales,
        ESN_dict=ESN_dict,
        U_washout_train=U_washout_train,
        U_train=U_train,
        U_val=U_val,
        Y_train=Y_train,
        Y_val=Y_val,
        n_folds=n_folds,
        n_realisations=n_realisations,
        N_washout=N_washout_steps,
        N_val=N_val_steps,
        N_trans=N_transient_steps,
        error_measure=error_measure,
    )

    res = run_gp_optimization(
        val_fun, 
        search_space, 
        n_calls, 
        n_initial_points, 
        rand_state=random_seed,
        search_grid=search_grid,
    )
    # find the top 5 parameters
    min_idx_list = res.func_vals.argsort()[:n_top]

    # save the best parameters
    for j, min_idx in enumerate(min_idx_list):
        for param_idx, param_name in enumerate(param_names):
            # rescale the parameters according∂to the given scaling
            reverse_scaler = getattr(reverse_scalers, param_scales[param_idx])
            new_param = reverse_scaler(res.x_iters[min_idx][param_idx])
            min_dict[param_name][j] = new_param

        min_dict["f"][j] = res.func_vals[min_idx]
    print(min_dict)

    return min_dict
