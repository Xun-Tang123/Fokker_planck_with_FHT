import random

import numpy as np
from copy import deepcopy
from fht_utils import fourier_custom
import random
from scipy.linalg import qr


def einsum(subscripts, *operands):
    return np.einsum(subscripts, *operands)


def parent(node):
    # Return parent node of current node
    k, l = node
    if l == 0:
        print('Warning: query parent node of root node')
        return None
    else:
        return int(np.ceil(k / 2)), l - 1


def child(node, child_id):
    # Return child node of current node
    # As one node as multiple child, one has child id either 1 or 2
    k, l = node
    if child_id == 1:
        return 2 * k - 1, l + 1
    elif child_id == 2:
        return 2 * k, l + 1
    else:
        raise ValueError('Wrong child id')


def generate_vectors(d, r, N=float('inf')):
    # Randomly generate Fourier modes for sketching. Ordered by total fourier degree
    # Start with the zero vector
    queue = [[(0, None)] * d]
    visited = set()
    results = []

    # Keep track of the current sum (i.e., "size" of the vectors)
    current_sum = 0
    deg_mode = lambda x: sum([x[i][0] for i in range(len(x))])

    while queue and len(results) < r and len(results) <= N:
        vec = queue.pop(0)
        vec_sum = deg_mode(vec)  # Calculate the sum of the current vector

        if tuple(vec) in visited or vec_sum > r:
            continue

        visited.add(tuple(vec))

        # Check if we're moving to a higher sum (i.e., next level of vectors)
        if vec_sum > current_sum:
            current_sum = vec_sum
            # Shuffle the queue when transitioning to vectors with a higher sum
            random.shuffle(queue)

        results.append(vec)

        # Generate offspring by adding 1 to each dimension
        if vec_sum + 1 <= r:
            for i in range(d):
                offspring_c, offspring_s = vec.copy(), vec.copy()
                offspring_c[i], offspring_s[i] = (offspring_c[i][0] + 1, 'cos'), (offspring_c[i][0] + 1, 'sin')
                # Only add offspring that haven't been visited and don't exceed the limit
                for offspring in [offspring_c, offspring_s]:
                    if tuple(offspring) not in visited:
                        queue.append(offspring)

    modes_chosen = np.zeros((len(results), d))
    for i, vec in enumerate(results):
        for k in range(d):
            modes_chosen[i, k] = 2 * vec[k][0]
            if vec[k][1] == 'sin':
                modes_chosen[i, k] -= 1

    return modes_chosen


def right_compress_custom(mat_, rank_):
    rank_ = min(rank_, mat_.shape[1], mat_.shape[0])
    rank_ = int(rank_)
    # Custom svd api.
    svd_ = np.linalg.svd
    U, S, Vh = svd_(mat_)
    V = Vh.T
    cutoff_ = 1e-2
    if S[rank_ - 1] < cutoff_ * S[0]:
        new_rank = np.sum(S > cutoff_ * S[0])
        rank_ = new_rank
    V = V[:, range(rank_)]
    return V


def trim_custom(mat_, rank_):
    rank_ = min(rank_, mat_.shape[1], mat_.shape[0])
    rank_ = int(rank_)
    # Custom svd api.
    svd_ = np.linalg.svd
    U, S, Vh = svd_(mat_)
    V = Vh.T
    cutoff_ = 1e-2
    if S[rank_ - 1] < cutoff_ * S[0]:
        new_rank = np.sum(S > cutoff_ * S[0])
        rank_ = new_rank
    U = U[:, range(rank_)]
    S = S[range(rank_)]
    V = V[:, range(rank_)]
    return U @ np.diag(S), V


def compress_custom(mat_, rank_):
    rank_ = min(rank_, mat_.shape[1], mat_.shape[0])
    rank_ = int(rank_)
    # Custom svd api.
    svd_ = lambda x: np.linalg.svd(x)
    U, S, Vh = svd_(mat_)
    V = Vh.T
    cutoff_ = 1e-2
    if S[rank_ - 1] < cutoff_ * S[0]:
        new_rank = np.sum(S > cutoff_ * S[0])
        rank_ = new_rank
    U = U[:, range(rank_)]
    V = V[:, range(rank_)]
    S = S[range(rank_)]

    return U @ np.diag(S) @ V.T


def sketching_custom(y, arg, extra_function=None):
    """
    Generate a sketch for a given input tensor, `y`, using the specified Fourier modes and indices in `arg`.

    Parameters:
    - y (numpy.ndarray): A 2D tensor where each column corresponds to a different signal.
                         Its shape is (N, M) where N is the number of samples and M is the number of signals.
    - arg (list of tuples): Specifies which Fourier mode and which column of y to use. Each tuple is of the form
                            (fourier_mode, index) where:
                            - fourier_mode (int): The Fourier mode to apply.
                            - index (int): 1-indexed index specifying which column of y to process.

    - extra_function (list of functions) : Specifies added functions for extra sketching

    Returns:
    - numpy.ndarray: A tensor which is the result of applying the Fourier mode(s) to the specified columns
                     of y. The dimensionality of the result depends on the length of `arg`.

    Notes:
    - The function supports up to a three-way tensor, meaning `arg` can contain at most three tuples.
    - The function utilizes `fourier_custom` to evaluate each signal on the Fourier mode.
      This function (`fourier_custom`) is not defined in this snippet and is assumed to be available elsewhere in
      the codebase.

    Raises:
    - ValueError: If `arg` contains more than three tuples.
    """
    fourier_sketch = []

    for i, (fourier_modes, idx) in enumerate(arg):
        new_idx = [x - 1 for x in idx]  # Account for 1-indexing
        y_column = y[:, new_idx]
        fourier_sketch.append(fourier_custom(y_column, fourier_modes))  # Evaluate sample on Fourier mode

    if extra_function is not None:
        for i, func in enumerate(extra_function):
            if func is not None:
                fourier_sketch[i] = np.concatenate((fourier_sketch[i], func(y)), axis=1)

    # argument
    if len(arg) == 1:
        result = fourier_sketch[0]
    elif len(arg) == 2:
        result = einsum('na, nb->nab', fourier_sketch[0], fourier_sketch[1])
    elif len(arg) == 3:
        result = einsum('na, nb, nc->nabc', fourier_sketch[0], fourier_sketch[1], fourier_sketch[2])
    else:
        raise ValueError('Calling sketching for more than a three-way tensor.')
    return 1 / y.shape[0] * np.sum(result, axis=0)


def average_fourier(y, node, L, arg, deg=1, level=1):
    '''
    Custom function. Take average of sin(y) over specified region. Is used to
    :param y (np.ndarray): input
    :param node (tuple of int): current node (see documentation in hier_tensor_sketch)
    :param L (int): total level of hierarchy
    :param deg (int): polynomial degree of the output.
    :param level: (int) the number of hierarchical levels taken to break up indices within node. Useful only when arg == 'self
    :return: Evaluation of average_fourier
    '''
    k, l = node
    N = y.shape[0]
    indices = np.arange(2 ** (L - l) * (k - 1), 2 ** (L - l) * k)
    level = int(min(L - l, level))
    n_blocks = 2 ** level
    sin_y_indices = np.sin(np.pi*y[:, indices])
    sin_y_indices = np.mean(sin_y_indices.reshape((N, n_blocks, -1)), axis=2)
    output = np.zeros([N, n_blocks * (deg + 1)])
    for i in range(deg):
        output[:, i * n_blocks:(i + 1) * n_blocks] = np.power(sin_y_indices,
                                                              i + 1)  # Store average of sin raised to power i+1
    return output


def hier_tensor_sketch(y, L, d, deg, r, s=None, debug=False, nbhd_fun=None, nbhd_int_fun=None):
    # input:
    # - y: an N*d dimensional array. Stores the samples
    # - L: Hierarchical level
    # - d: dimension of the input
    # - r: a dictionary encoding internal bond dimension
    # - s: a dictionary encoding sketching size
    # - deg: an integer capping the maximal degree of a Fourier mode.
    # - nbhd_fun : a function that chooses dimensions for sketching on the parent direction
    # - nbhd_int_fun : a function chooses dimensions for sketching on the child direction
    # output:
    # - c : a dictionary holding the array to be used in the FunctionalHierarchicalTensorFourier class.
    # One desires the following to roughly hold:
    # output_htn.eval(a) \approx <probability density on a>.
    if d != 2 ** L:
        raise ValueError('Input dimension size wrong')
    c = dict()  # holding the result for c
    # change degree information and rank information to vector format
    if np.size(r) == 1 and type(r) != dict:
        r_scalar = deepcopy(r)
        r = dict()
        for l in reversed(range(0, L + 1)):
            for k in range(1, 2 ** l + 1):
                if l == L:
                    r[(k, l)] = [2 * deg + 1, r_scalar]
                elif l == 0:
                    r[(k, l)] = [r_scalar, r_scalar]
                else:
                    r[(k, l)] = [r_scalar, r_scalar, r_scalar]

    if s is None:
        s = dict()
        s_level = 4 + 2 * np.arange(L, 0, -1)
        for l in reversed(range(0, L + 1)):
            for k in range(1, 2 ** l + 1):
                if l == L:
                    s[(k, l)] = [2 * deg + 1, r[(k, l)][1] + s_level[L - 1]]
                elif l == 0:
                    s[(k, l)] = [r[(k, l)][0] + s_level[0], r[(k, l)][1] + s_level[0]]
                else:
                    s[(k, l)] = [r[(k, l)][0] + s_level[l - 1], r[(k, l)][1] + s_level[l], r[(k, l)][2] + s_level[l]]
    n = 2 * deg + 1  # formula for total number of functions.

    # Prepare sketching function
    S_dict = dict()
    T_dict = dict()
    I_v_dict = dict()  # number of up-sketching modes

    for l in reversed(range(0, L + 1)):
        for k in range(1, 2 ** l + 1):
            node = (k, l)

            if l == L:
                I_v_dict[node] = int(s[node][1])
                if nbhd_fun is None:
                    idx_nbhd = [k - 2, k - 1, k + 1, k + 2]
                else:
                    idx_nbhd = nbhd_fun((k, l), L)
                idx_nbhd = [x for x in idx_nbhd if 1 <= x <= 2 ** L]
                mode_nbhd = generate_vectors(len(idx_nbhd), I_v_dict[node])
                T_dict[node] = (mode_nbhd, idx_nbhd)
                mode_mid = np.arange(0, n).reshape(-1, 1)
                S_dict[node] = (mode_mid, [k])  # Not called for sketching, but used for message aggregation
            elif l == 0:
                I_v_dict[node] = None
                T_dict[node] = None
            else:
                I_v_dict[node] = int(s[node][0])
                if nbhd_fun is None:
                    idx_nbhd = [2 ** (L - l) * (k - 1) - 1, 2 ** (L - l) * (k - 1), 2 ** (L - l) * k + 1,
                                2 ** (L - l) * k + 2]
                    idx_nbhd += [2 ** (L - l) * (k - 3), 2 ** (L - l) * (k - 2), 2 ** (L - l) * (k + 1),
                                 2 ** (L - l) * (k + 2)]
                else:
                    idx_nbhd = nbhd_fun((k, l), L)
                idx_nbhd = [x for x in idx_nbhd if 1 <= x <= 2 ** L]

                mode_nbhd = generate_vectors(len(idx_nbhd), I_v_dict[node])
                T_dict[node] = (mode_nbhd, idx_nbhd)
                if nbhd_int_fun is None:
                    mode_1, idx_1 = S_dict[child(node, 1)]
                    mode_2, idx_2 = S_dict[child(node, 2)]
                    idx_combined = idx_1 + idx_2
                else:
                    idx_combined = nbhd_int_fun((k, l), L)

                idx_chosen = random.sample(idx_combined, min(2 + 4 * (L - l), len(idx_combined)))
                mode_child = generate_vectors(len(idx_chosen), I_v_dict[node])
                S_dict[node] = (mode_child, idx_chosen)

    # System forming
    A_dict = dict()  # Dictionary holding the LHS of system of the queried node's parent node
    W_dict = dict()  # Dictionary holding the trimmed LHS
    B_dict = dict()  # Dictionary holding the RHS of system of the queried node
    G_dict = dict()  # Dictionary holding the solved linear system of queries node
    V_dict = dict()  # Dictionary holding gauge correction from trimming
    for l in reversed(range(0, L + 1)):
        for k in range(1, 2 ** l + 1):
            node = (k, l)
            if l == L:
                mode_mid = np.arange(0, n).reshape(-1, 1)
                idx_mid = [k]
                B_dict[node] = sketching_custom(y, [(mode_mid, idx_mid), T_dict[node]])
                A_dict[node] = sketching_custom(y, [(mode_mid, idx_mid), T_dict[node]])
            elif l != 0:
                if L - l > 1:
                    extra_function_B = []
                    extra_function_B.append(None)
                    extra_function_B.append(
                        lambda x: average_fourier(x, child(node, 1), L, 'self', level=min([L - l - 1, 3]), deg=1))
                    extra_function_B.append(
                        lambda x: average_fourier(x, child(node, 2), L, 'self', level=min([L - l - 1, 3]), deg=1))
                else:
                    extra_function_B = None
                B_dict[node] = sketching_custom(y, [T_dict[node], S_dict[child(node, 1)], S_dict[child(node, 2)]],
                                                extra_function_B)

                extra_function_A = []
                extra_function_A.append(lambda x: average_fourier(x, node, L, 'self', level=min([L - l, 3]), deg=1))
                extra_function_A.append(None)
                A_dict[node] = sketching_custom(y, [S_dict[node], T_dict[node]], extra_function_A)
            else:
                extra_function_B = []
                extra_function_B.append(
                    lambda x: average_fourier(x, child(node, 1), L, 'self', level=min([L - l, 3]), deg=1))
                extra_function_B.append(
                    lambda x: average_fourier(x, child(node, 2), L, 'self', level=min([L - l, 3]), deg=1))
                B_dict[node] = sketching_custom(y, [S_dict[child(node, 1)], S_dict[child(node, 2)]], extra_function_B)
                A_dict[node] = None

    for l in reversed(range(0, L + 1)):
        for k in range(1, 2 ** l + 1):
            node = (k, l)
            if A_dict[node] is None:
                V_dict[node] = None
            else:
                if l == L:
                    W_dict[node], V_dict[node] = trim_custom(A_dict[node], rank_=r[node][1])
                else:
                    W_dict[node], V_dict[node] = trim_custom(A_dict[node], rank_=r[node][0])

    # Solving for tensor sketching equations
    for l in reversed(range(0, L + 1)):
        for k in range(1, 2 ** l + 1):
            node = (k, l)
            if l == L:
                G_dict[node] = B_dict[node]
            elif l != 0:
                inv1 = np.linalg.pinv(W_dict[child(node, 1)])
                inv2 = np.linalg.pinv(W_dict[child(node, 2)])
                G_dict[node] = einsum('ij, ajk, lk -> ail', inv1, B_dict[node], inv2)
            else:
                inv1 = np.linalg.pinv(W_dict[child(node, 1)])
                inv2 = np.linalg.pinv(W_dict[child(node, 2)])
                G_dict[node] = einsum('ij, jk, lk -> il', inv1, B_dict[node], inv2)

    # Postprocessing to get tensor cores
    for l in reversed(range(0, L + 1)):
        for k in range(1, 2 ** l + 1):
            node = (k, l)
            if l == L:
                c[node] = einsum('ba, nb -> na', V_dict[node], G_dict[node])
            elif l != 0:
                c[node] = einsum('ba, bij -> aij', V_dict[node], G_dict[node])
            else:
                c[node] = G_dict[node]

    if debug:
        # Return intermediate sketch results for debugging
        return c, V_dict, G_dict, A_dict, B_dict, S_dict, T_dict
    else:
        return c
