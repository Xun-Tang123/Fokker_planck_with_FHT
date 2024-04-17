import numpy as np
from copy import deepcopy
from fht_utils import fourier_function


def einsum(subscripts, *operands):
    # For optimal performance, consider using oe.einsum from package opt_einsum
    return np.einsum(subscripts, *operands)


def is_root(node):
    k, l = node
    return l == 0


def parse_none_message(msg1, msg2):
    if msg1 == 'None' and msg2 == 'None':
        return 'None'
    elif msg1 == 'None' and (msg2 != 'None'):
        return msg1
    elif msg2 == 'None' and (msg1 != 'None'):
        return msg2
    else:
        print(f'Warning: This function should only be used when either msg is None.')
        raise ValueError('Wrong usage of PARSE_NONE_MESSAGE.')


class FunctionalHierarchicalTensorFourier:
    def __init__(self, d, L, deg, c, ghost_pt=None):
        # Initialize Hierarchical tensor
        # --input:
        #  - d : integer (Number of nodes)
        #  - L : integer (Number of levels)
        #  - ghost_pt : list, optional (The list contains ghost points which are not included)
        #  - deg : integer (Maximum fourier degree)
        #  - c : dictionary of tensor cores
        #  Note:
        #  The horizontal node index is 1-indexed, instead of 0-indexed.
        #  This indexing makes notation simplified.
        #  Convention of internal non-root node core indexing is {output bond, child1 bond, child2 bond}
        #  Convention of internal root node core indexing is {child1 bond, child2 bond}
        #  Convention of physical node is {Fourier bond, output bond}
        #  For ghost point, assume that parent node with ghost point node descendants acts as np.eye(r).
        self.d_true = d  # Input number for tensor
        self.L = L
        self.d = np.power(2, L)
        if ghost_pt is None:
            self.generate_ghost_pt()
        else:
            self.ghost_pt = ghost_pt
        self.ghost_pt.sort()
        self.node_dict = dict()
        self.create_node_dict()

        if len(ghost_pt) + self.d_true != self.d:
            raise ValueError('Ghost point size plus input dimension should equal 2**L.')

        self.n = 2 * deg + 1  # core dimension on node level
        self.deg = deg  # maximal polynomial degree of Fourier mode
        self.c = deepcopy(c)  # list of cores
        self.l2_msg = dict()
        self.P0 = np.eye(self.n)  # See documentation in get_contraction

    def generate_ghost_pt(self):
        # Generate ghost point
        # Use simple rule of right padding
        self.ghost_pt = np.arange(self.d_true + 1, self.d + 1)

    def create_node_dict(self):
        k_true_counter = 1
        for i in range(self.d):
            k = i + 1
            if not (k in self.ghost_pt):
                self.node_dict[k] = k_true_counter
                k_true_counter += 1

    def node_embedding(self, k):
        # Embedding of the index in [1, self.d] to [1, self.d_true].
        return self.node_dict[k]

    def check_node(self, k, l):
        # Check that node k agrees with level
        if k < 1 or k > 2 ** l:
            print(f'Value of k is {k} while the allowed range is {1}-{2 ** l}.')
            raise ValueError('Wrong Value input for node')
        if l < 0 or l > self.L:
            print(f'Value of l is {l} while the allowed range is {0}-{self.L}.')
            raise ValueError('Wrong Value input for node')

    def core_eval(self, x_n, i):
        # The indexing transform accounts for the 1-indexing and the ghost points
        return fourier_function(np.ravel(x_n[:, self.node_embedding(i) - 1]), self.deg)

    def parent(self, node):
        # Return parent node of current node
        k, l = node
        self.check_node(k, l)
        if l == 0:
            print('Warning: query parent node of root node')
            return None
        else:
            return int(np.ceil(k / 2)), l - 1

    def child(self, node, child_id):
        # Return child node of current node
        # As one node as multiple child, one has child id either 1 or 2
        k, l = node
        self.check_node(k, l)
        if l == self.L:
            print('Warning: query child node of leaf node')
            return None
        else:
            if child_id == 1:
                return 2 * k - 1, l + 1
            elif child_id == 2:
                return 2 * k, l + 1
            else:
                raise ValueError('Wrong child id')

    def get_eval_msg(self, x_n):

        eval_msg = dict()

        # Leaf to Root sweep
        orientation = 'L2R'
        for l in reversed(range(0, self.L + 1)):
            for k in range(1, 2 ** l + 1):
                node = (k, l)
                eval_msg = self.update_eval_msg(x_n, node, eval_msg, orientation)

        # Root to Leaf sweep
        orientation = 'R2L'
        for l in range(0, self.L + 1):
            for k in range(1, 2 ** l + 1):
                node = (k, l)
                eval_msg = self.update_eval_msg(x_n, node, eval_msg, orientation)
        return eval_msg

    def evaluate_marginal(self, x_n, mask):
        # Perform function space evaluation.
        # Mask is a list of dimenisons to take marginal over
        eval_msg = dict()

        # Leaf to Root sweep
        orientation = 'L2R'
        for l in reversed(range(0, self.L + 1)):
            for k in range(1, 2 ** l + 1):
                # print(eval_msg.keys())
                node = (k, l)
                if l == self.L and k in mask:
                    eval_msg = self.update_eval_msg_masked(x_n.shape[0], node, eval_msg)
                else:
                    eval_msg = self.update_eval_msg(x_n, node, eval_msg, orientation)
            if l == 0:
                node = (1, l)
                eval_msg = self.update_eval_msg(x_n, node, eval_msg, orientation)
        return eval_msg[((1, 0), None)]

    def evaluate(self, x_n):
        # Perform function space evaluation
        eval_msg = dict()

        # Leaf to Root sweep
        orientation = 'L2R'
        for l in reversed(range(0, self.L + 1)):
            for k in range(1, 2 ** l + 1):
                node = (k, l)
                eval_msg = self.update_eval_msg(x_n, node, eval_msg, orientation)
            if l == 0:
                node = (1, l)
                eval_msg = self.update_eval_msg(x_n, node, eval_msg, orientation)
        return eval_msg[((1, 0), None)]
        # return eval_msg

    def unpack_child_message(self, node, eval_msg):
        msg1 = eval_msg[(self.child(node, 1), node)]
        msg2 = eval_msg[(self.child(node, 2), node)]
        has_none_msg = msg1 == 'None' or msg2 == 'None'
        return msg1, msg2, has_none_msg

    def update_eval_msg(self, x_n, node, eval_msg, orientation):
        N = x_n.shape[0]
        k, l = node
        self.check_node(k, l)
        if orientation == 'L2R':
            # Leaf to Root sweep
            if self.is_leaf(node):
                if k in self.ghost_pt:
                    # Ghost point does not send messages
                    eval_msg[(node, self.parent(node))] = 'None'
                else:
                    P = np.einsum('na, ab -> nb', self.core_eval(x_n, k), self.c[node])
                    eval_msg[(node, self.parent(node))] = P.reshape((N, -1))
            elif self.is_middle(node):
                msg1, msg2, has_none = self.unpack_child_message(node, eval_msg)
                if has_none:
                    eval_msg[(node, self.parent(node))] = parse_none_message(msg1, msg2)
                else:
                    eval_msg[(node, self.parent(node))] = np.einsum('na, nb, cab -> nc', msg1, msg2, self.c[node])
            else:  # at root node
                msg1, msg2, has_none = self.unpack_child_message(node, eval_msg)
                if has_none:
                    eval_msg[(node, None)] = parse_none_message(msg1, msg2)
                else:
                    eval_msg[(node, None)] = np.einsum('na, nb, ab -> n', msg1, msg2, self.c[node])

        elif orientation == 'R2L':
            # Root to Leaf sweep
            if is_root(node):
                msg1, msg2, has_none = self.unpack_child_message(node, eval_msg)
                if msg1 == 'None':
                    eval_msg[(node, self.child(node, 1))] = eval_msg[(node, None)]
                    eval_msg[(node, self.child(node, 2))] = 'None'
                elif msg2 == 'None':
                    eval_msg[(node, self.child(node, 2))] = eval_msg[(node, None)]
                    eval_msg[(node, self.child(node, 1))] = 'None'
                else:
                    eval_msg[(node, self.child(node, 2))] = np.einsum('na, ab -> nb', msg1, self.c[node])
                    eval_msg[(node, self.child(node, 1))] = np.einsum('nb, ab -> na', msg2, self.c[node])
            elif self.is_middle(node):
                msg1, msg2, has_none = self.unpack_child_message(node, eval_msg)
                msg3 = eval_msg[(self.parent(node), node)]
                if msg1 == 'None':
                    eval_msg[(node, self.child(node, 2))] = msg3
                    eval_msg[(node, self.child(node, 1))] = 'None'
                elif msg2 == 'None':
                    eval_msg[(node, self.child(node, 1))] = msg3
                    eval_msg[(node, self.child(node, 2))] = 'None'
                else:
                    # By construction, msg cannot be None
                    if msg3 == 'None':
                        raise ValueError('msg3 cannot be None message.')
                    else:
                        eval_msg[(node, self.child(node, 1))] = np.einsum('nb, nc, cab -> na', msg2, msg3, self.c[node])
                        eval_msg[(node, self.child(node, 2))] = np.einsum('na, nc, cab -> nb', msg1, msg3, self.c[node])
            else:
                # No need to compute
                eval_msg[(node, None)] = eval_msg[((1, 0), None)]
        else:
            raise ValueError('update_eval_msg: Orientation argument must be either L2R or R2L.')
        return eval_msg

    def update_eval_msg_masked(self, N, node, eval_msg):
        k, l = node
        self.check_node(k, l)
        if k in self.ghost_pt:
            # Ghost point does not send messages
            eval_msg[(node, self.parent(node))] = 'None'
        else:
            P = np.outer(np.sqrt(2) * np.ones(N), self.c[node][0, :])
            eval_msg[(node, self.parent(node))] = P.reshape((N, -1))
        return eval_msg

    def is_leaf(self, node):
        k, l = node
        return l == self.L

    def is_middle(self, node):
        k, l = node
        return l != self.L and l != 0
