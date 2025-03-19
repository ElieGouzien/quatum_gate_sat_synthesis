#!/usr/bin/env python3
# coding: utf-8
"""
Module for gate set generation.

@author: Élie Gouzien
"""
from itertools import product, permutations
import warnings

from sympy import ImmutableMatrix, eye, zeros, diag, kronecker_product
from sympy import exp, log, sqrt, prod, arg, GramSchmidt, simplify
from sympy import I, pi, expand_complex

import numpy as np  # Only for faster symbolic tensor products

from algebraic_extension import algeb_class, ALGEB_CLASSES


# %% Elementary gates

X = ImmutableMatrix([[0, 1],
                     [1, 0]])
Y = ImmutableMatrix([[0, -I],
                     [I, 0]])
Z = ImmutableMatrix([[1,  0],
                     [0, -1]])


def rot_z(theta):
    """Rotation around Z with angle theta in rad (and a global phase)."""
    return ImmutableMatrix([[1, 0],
                            [0, exp(I*theta)]])


H = ImmutableMatrix([[1,  1],
                     [1, -1]]) / sqrt(2)

H_XY = (X + Y) / sqrt(2)
H_YZ = (Y + Z) / sqrt(2)

S = ImmutableMatrix([[1, 0],
                     [0, I]])
SQRT_X = ImmutableMatrix([[1+I, 1-I],
                          [1-I, 1+I]]) / 2
SQRT_Y = ImmutableMatrix([[1+I, -1-I],
                          [1+I, 1+I]]) / 2

C_XYZ = ImmutableMatrix([[1, -I],
                         [1, I]]) / sqrt(2)

T = rot_z(pi/4)
T_X = H * T * H
T_Y = H_YZ * T * H_YZ

TOFFOLI = (kronecker_product(diag(1, 1, 1, 0), eye(2))
           + kronecker_product(diag(0, 0, 0, 1), X))
CCZ = (kronecker_product(diag(1, 1, 1, 0), eye(2))
       + kronecker_product(diag(0, 0, 0, 1), Z))
CCCZ = (kronecker_product(diag(1, 1, 1, 1, 1, 1, 1, 0), eye(2))
        + kronecker_product(diag(0, 0, 0, 0, 0, 0, 0, 1), Z))

YCY = ImmutableMatrix([[1, -I, -I, 1],
                       [I,  1, -1, -I],
                       [I, -1, 1, -I],
                       [1, I, I, 1]]) / 2


GATES_EXAMPLE = {'X': X, 'Z': Z, 'sqrt(X)': SQRT_X, 'S': S}


# %% Extanding and restricting functions

def dag(gate):
    """Hermitian conjugaison."""
    return gate.conjugate().transpose()


def promote(nb_qubits, target, gate):
    """Promote the single qubit 'gate' to itself, on multi qubits."""
    if not 0 <= target < nb_qubits:
        raise IndexError("'target' must index one qubit.")
    return kronecker_product(*(eye(2) if i != target else gate
                               for i in range(nb_qubits)))


def controlled(nb_qubits, control, target, gate,
               state=ImmutableMatrix([0, 1])):
    """Build the matrix of the controlled gate, 1 control, 1 target."""
    one = state @ state.transpose().conjugate()
    zero = eye(2) - one
    gates0 = [eye(2) for _ in range(nb_qubits)]
    gates1 = [eye(2) for _ in range(nb_qubits)]
    gates0[control] = zero
    gates1[control] = one
    gates1[target] = gate
    return kronecker_product(*gates0) + kronecker_product(*gates1)


def controlled2(nb_qubits, control0, control1, target, gate,
                state0=ImmutableMatrix([0, 1]),
                state1=ImmutableMatrix([0, 1])):
    """Build the matrix of the controlled gate, 2 control, 1 target."""
    one0 = state0 @ state0.transpose().conjugate()
    zero0 = eye(2) - one0
    one1 = state1 @ state1.transpose().conjugate()
    zero1 = eye(2) - one1
    gates00 = [eye(2) for _ in range(nb_qubits)]
    gates01 = [eye(2) for _ in range(nb_qubits)]
    gates10 = [eye(2) for _ in range(nb_qubits)]
    gates11 = [eye(2) for _ in range(nb_qubits)]
    gates00[control0] = zero0
    gates00[control1] = zero1
    gates01[control0] = zero0
    gates01[control1] = one1
    gates10[control0] = one0
    gates10[control1] = zero1
    gates11[control0] = one0
    gates11[control1] = one1
    gates11[target] = gate
    return (kronecker_product(*gates00) + kronecker_product(*gates01)
            + kronecker_product(*gates10) + kronecker_product(*gates11))


def test_controlled2():
    """Vérifie vite fait qu'on retrouve bien la Toffoli."""
    assert controlled2(3, 0, 1, 2, X) == TOFFOLI


def fix_inputs_zero(mat, fixed_qubits, nb_qubits=None, strict=True):
    """Given a matrix, remove parts corresponding to input |1> on some qubits.

    Fixed qubits are igored if their index is larger that the number of qubits
    corresponding to the given matrix. This allows to introduce ancillary
    qubits that are supposed to be post-selected at the end (a warning is
    given).

    Inefficient function, but no need for a fast one.
    """
    if nb_qubits is None:
        nb_qubits = int(log(mat.shape[0], 2))
    if not mat.shape[0] == mat.shape[1] == 2**nb_qubits:
        if strict:
            raise ValueError("Unconsistant sizes")
        warnings.warn("Unconsistant sizes")
    if not all(x < nb_qubits for x in fixed_qubits):
        # raise ValueError("Fixed_qubits must contain indexes of qubits "
        #                  "compatible with the size of the matrix!")
        # TODO: gérer ça mieux.
        warnings.warn("Index trop grands ignorés "
                      "(permet de gérer les auxiliaires)")
        fixed_qubits = [x for x in fixed_qubits if x < nb_qubits]
    keep_indexes = []
    for i, state in enumerate(product((0, 1), repeat=nb_qubits)):
        if all(state[q] == 0 for q in fixed_qubits):
            keep_indexes.append(i)
    return mat[:, keep_indexes]


def fix_output_postselect(mat, post_select: dict, nb_qubits=None):
    """Remove lines of matrix to keep only the specified post-selections.

    post_select : dictionnary with as keys the index of the qubits to
    post-selection on, and value the value to keep (0 or 1).

    Ineficient function, but no need for a fast one.

    WARNING : this operation does not conserve normalisation of the matrix.
              To restore it, it can for instance been assumed a 1/2 probability
              to get the post-selected values for each qubit. Proper
              normalisation should be column by column, but is more complicated
              to handle with variable matrix.
    """
    if nb_qubits is None:
        nb_qubits = int(log(mat.shape[0], 2))
    if not mat.shape[0] == 2**nb_qubits:
        raise ValueError("Unconsistant sizes")
    # if not mat.shape[1] == 2**nb_qubits:
    #     warnings.warn("Non-square matrix !")
    if not all(x < nb_qubits for x in post_select):
        raise ValueError("post_select keys must be indexes of qubits "
                         "compatible with the size of the matrix!")
    keep_indexes = []
    for i, state in enumerate(product((0, 1), repeat=nb_qubits)):
        if all(state[q] == val for q, val in post_select.items()):
            keep_indexes.append(i)
    if isinstance(mat, ALGEB_CLASSES):
        return mat.mat_indexed[keep_indexes, :]
    return mat[keep_indexes, :]


def is_unitary(mat):
    """Check if matrix is unitary."""
    return simplify(dag(mat)*mat - eye(*mat.shape)) == zeros(*mat.shape)


def gram_schmidt(mat):
    """Apply sympy's GramSchmidt to the matrix."""
    res = ImmutableMatrix.hstack(*GramSchmidt(mat.columnspace(), True))
    if not is_unitary(res):
        raise RuntimeError("Échec de l'orthogonalisation.")
    return res


def ensure_unit_int(mat, deep, nb_phases=8):
    """Ensure the matrix unitary, such that 2**deep*mat is in correct ring."""
    # TODO: l'implémenter, au pire avec un solveur quadratique.
    raise NotImplementedError


def approx_mat_mul(*mats, deep=None, nb_phases=8):
    """Aproximate multipication, with precision 2**deep.

    Done to give a group structure to the gates generated from standard set and
    finite deepth.
    Example from arXiv:quant-ph/0306018:
        >>> res = approx_mat_mul(H,T, H, dag(T), H, T, H, T, H, T, H, dag(T),
        ...                      H, dag(T), H, T, H, T, H, dag(T), H, dag(T),
        ...                      H, T, H, dag(T), H, dag(T), H, dag(T), H,
        ...                      deep=5)
    """
    warnings.warn("Attention, le résultat n'est pas garanti d'être unitaire.")
    cls = algeb_class(nb_phases)
    if deep is None:
        raise TypeError("deep is mandatory")
    mat1, mat2, *reste = mats
    res = (cls.from_sympy_mat(2**deep*mat1) @ cls.from_sympy_mat(2**deep*mat2)
           ).div_round(2**deep).to_sympy()/(2**deep)
    if not reste:
        return res
    return approx_mat_mul(res, *reste, deep=deep, nb_phases=nb_phases)


# %% Gate sets

# TODO: generaliser l'extension à plein de qubits.

def h_cnot_t(nb_qubits):
    r"""Set with H, CNOT T, and T^\dag."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('H', 'T', 'T_DAG'), (H, T, dag(T))):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def pauli_h_cnot(nb_qubits):
    """Set with Pauli, H and CNOT gates."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('X', 'Y', 'Z', 'H'), (X, Y, Z, H)):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def h_s_cnot(nb_qubits):
    """Set with Pauli, H and CNOT gates."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('H', 'S'), (H, S)):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def pauli_h_s_cnot(nb_qubits):
    """Set with Pauli, H and CNOT gates."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('X', 'Y', 'Z', 'H', 'S'), (X, Y, Z, H, S)):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def min_clifford_t_gates(nb_qubits, minimalist=True):
    """Minimum set to find known implementations of Toffoli gates.

    Reference circuit:
    https://commons.wikimedia.org/wiki/File:Qcircuit_ToffolifromCNOT.svg
    """
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('H', 'T', 'T_DAG'), (H, T, dag(T))):
            if minimalist and name == 'H' and i != 2:
                continue
            if minimalist and name == 'T_DAG' and i == 0:
                continue
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j or (minimalist and i >= j):
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def min_clifford_t_gates_and(nb_qubits, targ=2, *, minimalist=True):
    """Minimum set to find known implementations of AND gate.

    Reference: arXiv:1805.03662
    """
    gates = {}
    # 1 qubit gates
    for i in [targ]:
        for name, gate in zip(('H', 'T', 'T_DAG', 'S_DAG'),
                              (H, T, dag(T), dag(S))):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j or (targ not in (i, j) and not minimalist) or (
                targ != j and minimalist):  # avant : if i >= j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def clifford_t_gates(nb_qubits):
    """Generate the gate set for one version of clifford+T gates."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        # TODO : ajouter les conjugées, multiples, et SQRT_X
        for name, gate in zip(('X', 'Y', 'Z', 'H', 'S', 'S_DAG', 'T', 'T_DAG'),
                              (X, Y, Z, H, S, dag(S), T, dag(T))):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in permutations(range(nb_qubits), 2):
        # TODO: ajouter d'autres portes Clifford à deux qubits.
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
        if i < j:  # CZ invariant on i <-> j
            gates[f"CZ_{i}_{j}"] = controlled(nb_qubits, i, j, Z)
    return gates


def clifford_t_gates_standardplus(nb_qubits):
    """Generate the gate set for one version of clifford+T gates.

    Here I take only H, T, T_DAG, CNOT (but no restriction on targets).
    """
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('H', 'T', 'T_DAG'),
                              (H, T, dag(T))):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in permutations(range(nb_qubits), 2):
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def clifford_toffoli_gates(nb_qubits):
    """Generate the gate set for one version of clifford+T gates."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        # TODO : ajouter les conjugées, multiples, et SQRT_X
        for name, gate in zip(('X', 'Y', 'Z', 'H', 'S', 'S_DAG'),
                              (X, Y, Z, H, S, dag(S))):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in permutations(range(nb_qubits), 2):
        # TODO: ajouter d'autres portes Clifford à deux qubits.
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
        if i < j:  # CZ invariant on i <-> j
            gates[f"CZ_{i}_{j}"] = controlled(nb_qubits, i, j, Z)
    # Toffoli gates
    for i, j, k in permutations(range(nb_qubits), 3):
        if i < j:  # the two controls are symetrical
            continue
        gates[f"TOFFOLI_{i}_{j}_{k}"] = controlled2(nb_qubits, i, j, k, X)
    return gates


def clifford_t_gates_reduced(nb_qubits):
    """Generate the gate set for one version of clifford+T gates.

    Only the last qubit can get the one qubit gates.
    """
    gates = {}
    # 1 qubit gates
    i = nb_qubits - 1
    # TODO : ajouter les conjugées, multiples, et SQRT_X
    for name, gate in zip(('X', 'Y', 'Z', 'H', 'S', 'S_DAG', 'T', 'T_DAG'),
                          (X, Y, Z, H, S, dag(S), T, dag(T))):
        gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        # TODO: ajouter d'autres portes Clifford à deux qubits.
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
        if i < j:  # CZ invariant on i <-> j
            gates[f"CZ_{i}_{j}"] = controlled(nb_qubits, i, j, Z)
    return gates


def gidney_problem_set(nb_qubits):
    """Generate the gate set for one version of clifford+T gates.

    https://x.com/CraigGidney/status/1379593866563743748
    """
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        for name, gate in zip(('H', 'H_XY'), (H, H_XY)):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
    return gates


def clifford_cs_gates(nb_qubits):
    """Generate the gate set for one version of clifford+CS gates."""
    gates = {}
    # 1 qubit gates
    for i in range(nb_qubits):
        # TODO : ajouter les conjugées, multiples, et SQRT_X
        for name, gate in zip(('X', 'Y', 'Z', 'H', 'S', 'S_DAG'),
                              (X, Y, Z, H, S, dag(S))):
            gates[name+'_'+str(i)] = promote(nb_qubits, i, gate)
    # 2 qubit gates
    for i, j in product(range(nb_qubits), repeat=2):
        if i == j:
            continue
        gates[f"CNOT_{i}_{j}"] = controlled(nb_qubits, i, j, X)
        if i < j:  # CZ and CS invariant on i <-> j
            gates[f"CZ_{i}_{j}"] = controlled(nb_qubits, i, j, Z)
            gates[f"CS_{i}_{j}"] = controlled(nb_qubits, i, j, S)
    return gates


# %% states, as gates (from |0>)
def ghz(nb_qubits):
    """Gate that apply identity, except on |0…0>, mapped to GHZ state.

    Warning, this is not unitary and only purposed to be fed to
    fix_inputs_zero().
    """
    gate = eye(2**nb_qubits)
    gate[0, 0] = 1/sqrt(2)
    gate[-1, 0] = 1/sqrt(2)
    # gate[:, 0] = 1/sqrt(2**nb_qubits) * ones(2**nb_qubits, 1)
    return gate


# %% Helper functions for interpreting results
def _find_nonzero(mat):
    """Find the coordinates of non-zero coefficients."""
    for coef in np.ndindex(mat.shape):
        if simplify(mat[coef]) != 0:
            return coef
    raise ValueError("Zéro matrix.")


def parse_result(gates, dvars, phases=None, *,
                 target=None, fixed_qubits=None, post_select=None,
                 verb=True, num_check=False):
    """Parse a solution into a readable format.

    dvars are given into the "left to right matrix product" order.
    (dvars[0] describes the last gate of the circuit)
    """
    # TODO: put everything on Chronological order.
    if post_select is None:
        post_select = {}
    if not len(dvars[0]) == len(gates):
        raise ValueError("dvars and gate set non consistents")
    gates_list = list(gates.items())

    # Retrieve gates
    if not all(sum(dvar) == 1 for dvar in dvars):
        raise ValueError("Only one gate by time-slice allowed.")
    res_gates = [gates_list[dvar.index(True)] for dvar in dvars]
    res_names, res_mats = zip(*res_gates)

    # Eventuelly, switch to numerical gates.
    if num_check:
        res_mats = [np.asarray(mat).astype(complex) for mat in res_mats]

    # Generate the reconstituted matrix
    # TODO: adapt to matrix or list
    if target is not None:
        if num_check:
            reconst = res_mats[0]
            for mat in res_mats[1:]:
                reconst = reconst @ mat
        else:
            reconst = expand_complex(prod(res_mats))

    # Retrieve phase
    if phases is not None:
        if not sum(phases) == 1:
            raise ValueError("Only one phase is allowed")
        phi = exp(2*pi*I/len(phases))**phases.index(True)
    elif target is not None:
        coord = _find_nonzero(target)
        phi = exp(I*arg(simplify(reconst[coord] / target[coord])))
    else:
        phi = None
    if num_check and phi is not None:
        phi = complex(phi)

    # Check reconstitution
    # TODO: adapt to matrix or list
    if target is not None and not hasattr(target, 'shape'):
        warnings.warn("Input/output vectors specification of the target "
                      "not yet supported")
    if target is not None and hasattr(target, 'shape'):
        phi_inv = phi**-1 if num_check else simplify(pow(phi, -1))
        phased_reconst = phi_inv * reconst
        if not num_check:
            phased_reconst = expand_complex(phased_reconst)
        if fixed_qubits:
            phased_reconst = fix_inputs_zero(phased_reconst, fixed_qubits)
            target = fix_inputs_zero(target, fixed_qubits)
        phased_reconst = fix_output_postselect(phased_reconst, post_select)
        # TODO: prendre en compte une meilleure normalisation (idem main code).
        phased_reconst *= sqrt(2)**len(post_select)
        equal = (np.all(phased_reconst == target) if num_check
                 else expand_complex(phased_reconst) == expand_complex(target))
        if equal:
            symbol = '∝'
        else:
            print("Matrice non parfaitement reconstituée !")
            print("Infidélité de la reconstruction :",
                  1 - (abs((dag(target) @ phased_reconst).trace().evalf())
                       / target.shape[0]))
            symbol = '≈'
    else:
        symbol = '=?='

    # Print result
    if verb:
        print("target", symbol, '*'.join(res_names))
        # Should be nice, but too large recursion.
        # with evaluate(False):
        #     pprint(Eq(symbols("target"), phi_inv * prod(res_mats)))

    return phi, res_gates
