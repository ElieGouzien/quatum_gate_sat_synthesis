#!/usr/bin/env python3
# coding: utf-8
"""
Solve the gate sythesis problem, by mapping to a SAT.

@author: Élie Gouzien
"""

import sys
import os
import time
import datetime
from functools import partial
import gzip
from itertools import combinations
from math import ceil

import z3
import numpy as np
import sympy as sp
from tqdm import tqdm

from algebraic_extension import algeb_class

from gates import Z, H, S, TOFFOLI, ghz, CCCZ
from gates import controlled, promote, dag
from gates import fix_inputs_zero, fix_output_postselect
from gates import (GATES_EXAMPLE, clifford_t_gates, min_clifford_t_gates,
                   min_clifford_t_gates_and, gidney_problem_set, pauli_h_cnot)
from gates import clifford_cs_gates, h_cnot_t, clifford_t_gates_reduced
from gates import clifford_toffoli_gates, clifford_t_gates_standardplus
from gates import parse_result


def _np_int_var(shape, name='', nb_bits=None, bound=None, solver=None):
    """Create integral vars and translate it into a numpy object array.

    if nb_bits is None, create Int, otherwise BitVec.
    """
    func = z3.Int if nb_bits is None else partial(z3.BitVec, bv=nb_bits)
    res = np.array([[func(name+f"_{i}_{j}") for j in range(shape[1])]
                    for i in range(shape[0])], dtype=object)
    if bound is not None:
        for coef in res.flat:
            solver.add(-bound <= coef)
            solver.add(coef <= bound)
    return res


def _algeb_re_var(nb_phases, shape, name='', nb_bits=None,
                  bound=None, solver=None):
    """Create a real algebraic integral var."""
    if nb_phases == 8:
        elem_names = ['int', 'sqrt2']
        bound_div = [1, np.sqrt(2)]
    elif nb_phases == 16:
        elem_names = ['int', 'sqrt2', 'sqrtbis', 'sqrt_sqrtbis']
        bound_div = [1, np.sqrt(2), np.sqrt(np.sqrt(2) + 2),
                     np.sqrt(2)*np.sqrt(np.sqrt(2)+2)]
    else:
        raise ValueError("nb_phases supported only for 8 or 16")
    return algeb_class(nb_phases, real=True)._make(
        _np_int_var(shape, name+'_'+t, nb_bits,
                    int(bound/div) if bound is not None else None, solver)
        for t, div in zip(elem_names, bound_div))


def _algeb_var(nb_phases, shape, name='', nb_bits=None,
               bound=None, solver=None):
    """Create an algebraic integral var."""
    return algeb_class(nb_phases, real=False)._make(
        _algeb_re_var(nb_phases, shape, name+'_'+t, nb_bits, bound, solver)
        for t in ['real', 'imag'])


@np.vectorize
def z3_simplify(expr):
    """Try to simplify with z3 or just return the input."""
    try:
        return z3.simplify(expr)
    except z3.Z3Exception:
        return expr


def export_dimacs(goal, filename, init_vars=(), verb=True, gates_dict=None):
    """Export a goal into CNF form, with preprocess.

    This fonction is not careful and if the conversion into CNF fails, you
    won't get a warning but a false result.
    """
    # Was done because the cnf.dimacs() from Z3 was bugged.
    # TODO remove this and just use cnf.dimacs() now that it's fixed!
    # We could use repr(var) to knwow if its var, Not(var) or Or(var1, var1)
    # instead, if var.num_args() == 0 -> var
    #          if var.num_args() == 1 -> Not
    #          if var.num_args() >= 1 -> Or
    # That's dangerous, be carefull with that.
    out = sys.stdout if verb else open(os.devnull, 'w', encoding='utf-8')

    print("Traduction du problème en SAT…", file=out)
    tactic = z3.Then('simplify', 'reduce-bv-size', 'bit-blast', 'pb2bv',
                     'tseitin-cnf')
    res = tactic(goal)
    assert len(res) == 1
    if len(res[0]) == 0:
        raise RuntimeError("Modèle faisable, rien à exporter !")
    if len(res[0]) == 1 and res[0][0].sexpr() == 'false':
        raise RuntimeError("Modèle infaisable, rien à exporter !")

    class AutoIndex(dict):
        """Dictionnary that index the values (starting at 1)."""

        def __missing__(self, key):
            val = len(self) + 1
            self[key] = val
            return val

    clauses = []
    var_index = AutoIndex()
    for var in init_vars:  # placement des bonnes variables en premier.
        var_index[var.sexpr()]

    def signed_var(var):
        """Find if not, and apply -."""
        # if repr(var).startswith('Not('):
        if var.num_args() == 1:
            return -var_index[var.arg(0).sexpr()]
        # assert var.num_args() == 0
        return var_index[var.sexpr()]

    print("Traduction des clauses canoniques en variables python…", file=out)
    # TODO : cette partie gagnerait à être codée en C.
    for clause in tqdm(res[0], file=out):
        # if repr(clause).startswith('Or('):
        if clause.num_args() >= 2:
            clauses.append([signed_var(var) for var in clause.children()])
        else:
            clauses.append([signed_var(clause)])

    print("Writing file…", file=out)
    with gzip.open(filename+'.gz', 'wt') as file:
        if gates_dict:
            print("c", gates_dict.keys(), file=file)
        print("c Variables in order :", file=file)
        # dict ordered and variables created in order
        print('c', ' '.join(v for v in var_index.keys()), file=file)
        print('c Indications pour approxmc :', file=file)
        print('c ind', ' '.join(str(var_index[v.sexpr()]) for v in init_vars),
              '0', file=file)
        print('p', 'cnf', len(var_index), len(clauses), file=file)
        for clause in clauses:
            print(*clause, 0, file=file)


def export_smt(solver, filename, shown_vars=()):
    """Write the smt file."""
    with open(filename, 'w') as file:
        print("(set-logic QF_LIA)", file=file)
        # print(solver.to_smt2(), file=file)
        print(solver.sexpr(), file=file)
        print("(check-sat)", file=file)
        for var in shown_vars:
            print(f"(get-value ({var}))", file=file)
        # print("(get-info :all-statistics)", file=file)
        # print("(get-model)", file=file)
        # print("( exit )", file=file)


def load_sat_sol(filename, gates, deep=None, nb_phases=8, *, verb=True):
    """Charge le fichier de solution du solver SAT, et donne le résultat."""
    if deep is None:
        deep = int(filename.partition("deep")[2].split('_')[0].split('.')[0])
    row_vars = []
    with (gzip.open(filename, 'rt') if filename.endswith('.gz')
          else open(filename)) as file:
        solved = False
        for line in file:
            if line.startswith('c'):
                continue
            elif line.startswith('s'):
                if line[2:].strip() == 'SATISFIABLE' and verb:
                    solved = True
                    print("SATISFIABLE")
                elif line[2:].strip() == 'UNSATISFIABLE':
                    if verb:
                        print('UNSATISFIABLE')
                    return None, None
                else:
                    raise RuntimeError("Unknown solution type.")
            elif line.startswith('v'):
                row_vars.extend(int(x) > 0 for x in line[2:].strip().split())
        if not solved:
            raise RuntimeError(
                f"File {filename} contains no solution. "
                "You should delete it and run again the solver.")
    phases = row_vars[:nb_phases]
    dvars = [row_vars[nb_phases+len(gates)*i:nb_phases+len(gates)*(i+1)]
             for i in range(deep)]
    dvars.reverse()  # SAT formulated in chronological order
    return phases, dvars


def read_metadata(filename):
    """Read the fixed_qubits and post_select variables from filename."""
    fixed_qubits = ()
    post_select = {}
    if "_fixed-" in filename:
        fixed_qubits = tuple(int(i) for i in filename.partition(
            "_fixed-")[2].partition('_')[0].partition('.')[0].split('-'))
    if "_postselect-" in filename:
        post_select = {
            int(s.split(':')[0]): int(s.split(':')[1])
            for s in filename.partition("_postselect-")[2].partition(
                    '_')[0].partition('.')[0].split('-')}
    return fixed_qubits, post_select


def exclude_2(solver, variables):
    """Prevent any two variables from the list to be simultaneously true."""
    for var1, var2 in combinations(variables, 2):
        solver.add(z3.Or(z3.Not(var1), z3.Not(var2)))


def z3_mat_solve(target, gates, deep, nb_phases=8, fixed_qubits=(),
                 post_select={}, *,
                 verb=True, sat=False, max_t=None, export=False,
                 strict_bound=False):
    """Build and solve the problem representing each intermediate, with z3.

    target : target definition. Two cases are possible :
        1. a matrix (anything with a shape), it can be squared or not to take
            into account post-selection.
        2. a list of tuple when whe specify the target as mapping some vectors
            to other vectors.

    It uses matrix expressions an general constraints to propagate the value
    of the matrix.

    Supported nb_phases : 8 and 16.

    strict_bound: use `ceil(1.5 d) + 2` if True, otherwise `d + 2`.
                  first one is proven, second seems to work.
    """
    # TODO: voir si on peut prendre en compte ça : arXiv:1206.0758
    out = sys.stdout if verb else open(os.devnull, 'w', encoding='utf-8')

    print("\tConversion des portes en semi-numérique (avec facteur 2)…",
          file=out)
    gates_shape = next(iter(gates.values())).shape
    assert all(gate.shape == gates_shape for gate in gates.values())
    formal_gates = [algeb_class(nb_phases).from_sympy_mat(2*gate)
                    for gate in gates.values()]
    # Taille des unitaires avec qubits fixés, mais pas encore de postselection.
    intermed_shape = (gates_shape[0], gates_shape[1]//(2**len(fixed_qubits)))
    if hasattr(target, 'shape'):  # Matrix case
        # HINT: one factor 2 is kept for multiplying the phase variable.
        # target assumed to have already the shape of post-selected.
        target = algeb_class(nb_phases).from_sympy_mat(2**(deep-1) * target)
        target = target.map(partial(fix_inputs_zero,
                                    fixed_qubits=fixed_qubits))
        target_shape = target.shape
    else:  # tuple of vectors case
        # TODO: consider rebuilding a matrix problem, to factorize code.
        nb_bits_targ = 2
        while True:
            try:
                target = [
                    (algeb_class(nb_phases).from_sympy_mat(
                        2**nb_bits_targ*t[0]),
                     algeb_class(nb_phases).from_sympy_mat(
                         2**(nb_bits_targ+deep-1)*t[1]))
                    for t in target]
            except ValueError:
                nb_bits_targ += 1
                continue
            break
        # HINT: no reshaping is done, supposed to already take into account
        #       fixed qubits and post-selection.
        target_shape = (target[0][1].shape[0], target[0][0].shape[0])

    print("Création du modèle et des variables…", file=out)
    # z3.set_param("parallel.enable", True)  # rend très lent
    if export and sat:
        solver = z3.Goal()
    elif sat:
        solver = z3.SolverFor("QF_FD")
    else:
        solver = z3.SolverFor("QF_LIA")
    if verb:
        z3.set_option(verbose=2)
    z3_phases = z3.BoolVector('phase', nb_phases)
    # dvars are in the chronological order, not as in the gate analysis
    z3_dvars = [z3.BoolVector(f'x{depth}', len(gates))
                for depth in range(deep)]

    # Intermediate matrices
    # HINT: not sure z3 internal solver benefits from using BitVec()
    bound_factor = 1.5 if strict_bound else 1
    nb_bits = (ceil(bound_factor*deep) + 2 if hasattr(target, 'shape')
               else ceil(bound_factor*(nb_bits_targ+deep)) + 2)
    if strict_bound:
        nb_bits += 2  # overbound: need room for no overflow.
    z3_mats = [_algeb_var(nb_phases, intermed_shape, name=f"mat_{i}",
                          nb_bits=nb_bits if sat else None,
                          bound=2**(ceil(bound_factor*i)+1) if sat else None,
                          solver=solver)
               for i in range(deep)]
    # bound on the phase multiplication: +2 is overestimation of *2sqrt(2)
    extra_bnd = 2 if strict_bound else 0
    if hasattr(target, 'shape'):
        phased_target = _algeb_var(nb_phases, target_shape,
                                   name="phased_target",
                                   nb_bits=nb_bits if sat else None,
                                   bound=2**(deep+extra_bnd) if sat else None,
                                   solver=solver)
    else:
        phased_targ_vects = [
            _algeb_var(nb_phases, (target_shape[0], 1),
                       name=f"phased_targ_vects[{i}]",
                       nb_bits=nb_bits if sat else None,
                       bound=2**(nb_bits_targ+deep+extra_bnd) if sat else None,
                       solver=solver) for i in range(len(target))]

    # Exclusion conditions : one phase, one gate by time slot
    if sat:
        exclude_2(solver, z3_phases)
    else:
        solver.add(z3.AtMost(*z3_phases, 1))
    solver.add(z3.Or(*z3_phases))
    for moment_dvars in z3_dvars:
        if sat:
            exclude_2(solver, moment_dvars)
        else:
            solver.add(z3.AtMost(*moment_dvars, 1))
        solver.add(z3.Or(*moment_dvars))

    # Very special condition : no more than max_t T or T^dag gates
    if max_t is not None:
        t_indexes = [i for i, g in enumerate(gates) if g.startswith('T')]
        t_dvars = [deep_vars[i] for deep_vars in z3_dvars for i in t_indexes]
        # solver.add(z3.Or(*t_dvars))  # Force number of T gates >= 1
        solver.add(z3.AtMost(*t_dvars, max_t))

    # Contraintes de propagation de l'information.
    print("Imposition des contraintes de succession…", file=out)
    # 1ere porte : trivial
    print(f"\tCouche 1/{deep}", file=out)
    for var, gate in zip(z3_dvars[0], formal_gates):
        for m1, m2 in zip(z3_mats[0].iter_elems(), gate.iter_elems()):
            m2 = fix_inputs_zero(m2, fixed_qubits)
            for coord in np.ndindex(intermed_shape):
                solver.add(z3.simplify(
                    z3.Implies(var, m1[coord] == m2[coord])))

    # portes suivantes
    for i in range(1, deep):
        print(f"\tCouche {i+1}/{deep}", file=out)
        for var, gate in zip(z3_dvars[i], formal_gates):
            pmat = gate @ z3_mats[i-1]
            pmat = pmat.map(z3_simplify)
            for m1, m2 in zip(z3_mats[i].iter_elems(), pmat.iter_elems()):
                for coord in np.ndindex(intermed_shape):
                    solver.add(z3.simplify(
                        z3.Implies(var, m1[coord] == m2[coord])))
    del var, gate

    # build phased target
    print("Phasage de la cible et contraintes d'objectif…", file=out)
    phi = sp.exp(2*sp.I*sp.pi/nb_phases)
    for k, var in enumerate(z3_phases):
        if hasattr(target, 'shape'):  # Matrix case
            # HINT: factor 2 "taken" from target
            pmat = algeb_class(nb_phases).from_sympy(2*phi**k) * target
            for m1, m2 in zip(phased_target.iter_elems(), pmat.iter_elems()):
                for coord in np.ndindex(target_shape):
                    solver.add(z3.simplify(
                        z3.Implies(var, m1[coord] == m2[coord])))
        else:  # Vector tuples case
            for targ, phased_vec in zip(target, phased_targ_vects):
                # HINT: factor 2 "taken" from target
                pvect = algeb_class(nb_phases).from_sympy(2*phi**k) * targ[1]
                for m1, m2 in zip(phased_vec.iter_elems(), pvect.iter_elems()):
                    for coord in np.ndindex((target_shape[0], 1)):
                        solver.add(z3.simplify(
                            z3.Implies(var, m1[coord] == m2[coord])))

    # Force result matrix value
    res_var_mat = fix_output_postselect(z3_mats[-1], post_select)
    # TODO: assumes here that probability of post-select is 1/2.
    #       it would be better to do a proper renormalisation, column by column
    res_var_mat *= algeb_class(nb_phases).new_sqrt2()**len(post_select)
    if hasattr(target, 'shape'):  # Matrix case
        for m1, m2 in zip(res_var_mat.iter_elems(),
                          phased_target.iter_elems()):
            for coord in np.ndindex(target_shape):
                solver.add(z3.simplify(m1[coord] == m2[coord]))
    else:
        for targ, phased_vec in zip(target, phased_targ_vects):
            for m1, m2 in zip((res_var_mat@targ[0]).iter_elems(),
                              phased_vec.iter_elems()):
                for coord in np.ndindex((target_shape[0], 1)):
                    solver.add(z3.simplify(m1[coord] == m2[coord]))

    # Nettoyage avant la conversion (qui prend beaucoup de RAM)
    del k, var, m1, m2, res_var_mat, pmat, moment_dvars, z3_mats, formal_gates
    del coord
    if hasattr(target, 'shape'):
        del phased_target
    else:
        del phased_targ_vects, phased_vec, targ, pvect
    del target

    if export:
        print("Exportation du problème…", file=out)
        basename = f"{export}_gates{len(gates)}_deep{deep}"
        if max_t is not None:
            basename += f"_{max_t}T"
        if fixed_qubits:
            basename += "_fixed-" + '-'.join(map(str, fixed_qubits))
        if post_select:
            basename += "_postselect-" + '-'.join(
                f"{i}={j}" for i, j in post_select.items())
        if sat:
            export_dimacs(solver, basename + '.cnf',
                          init_vars=z3_phases + sum(z3_dvars, []),
                          verb=verb, gates_dict=gates)
        else:
            export_smt(solver, basename + '.smt2',
                       z3_phases + sum(z3_dvars, []))
        return None, None

    print("Résolution du problème…", file=out)
    start_time = time.time()
    status = solver.check()
    print("\nTemps de résolution :", datetime.timedelta(
        seconds=time.time() - start_time), file=out)
    print("Solver stats :", solver.statistics(), file=out)
    print("Status :", status, file=out)

    # Parse result
    if status == z3.sat:
        model = solver.model()
        print(model, file=out)
        phases = [bool(model[p]) for p in z3_phases]
        dvars = [[bool(model[v]) for v in var] for var in z3_dvars]
        dvars.reverse()  # back to left to right product order
    else:
        phases, dvars = None, None
    return phases, dvars


# %% Main part
if __name__ == '__main__':
    # Frequent, and default options
    fixed_qubits, post_select = [], {}

    # H
    # targ_name, target = 'H', H
    # gates = GATES_EXAMPLE

    # CZ
    targ_name, target = 'CZ', controlled(2, 0, 1, Z)
    gates = clifford_t_gates(2)
    del gates['CZ_0_1']

    # S, as asked in https://twitter.com/CraigGidney/status/1379593866563743748
    # targ_name, target, fixed_qubits = 'S_0', promote(2, 0, S), [1]
    # gates = gidney_problem_set(2)

    # YCY, as asked in
    # https://twitter.com/CraigGidney/status/1432859006184275975
    # from gates import YCY, h_s_cnot
    # targ_name, target = 'YCY', YCY
    # gates = h_s_cnot(2)

    # AND
    # targ_name, target, fixed_qubits = 'AND', TOFFOLI, [2]
    # gates = clifford_t_gates(3)
    # gates = min_clifford_t_gates_and(3)

    # Toffoli
    # targ_name, target = 'TOFFOLI', TOFFOLI
    # gates = clifford_t_gates(3)
    # gates = min_clifford_t_gates(3)
    # gates = clifford_t_gates_standardplus(3)

    # Toffoli with ancillary
    # targ_name, target = 'TOFFOLI_ancilla', TOFFOLI
    # fixed_qubits, post_select = [3], {3: 0}
    # TODO: deal with correction
    # gates = clifford_t_gates(4)
    # gates = min_clifford_t_gates_and(4, 3, minimalist=False)

    # Toffoli from arXiv:1212.5069, with 4 or 3 qubits
    # targ_name, fixed_qubits = 'TOFFOLI_STAR', [3]
    # target = controlled(4, 0, 1,
    #                     dag(S))*sp.kronecker_product(TOFFOLI, sp.eye(2))
    # gates = h_cnot_t(4)
    # targ_name = 'TOFFOLI_STAR'
    # target = controlled(3, 0, 1, dag(S)) * TOFFOLI
    # gates = h_cnot_t(3)

    # CCCZ
    # targ_name, target = 'CCCZ', CCCZ
    # gates = clifford_t_gates(4)
    # gates = clifford_t_gates_reduced(4)

    # GHZ
    # nb_qubits = 3
    # targ_name, target, fixed_qubits = 'GHZ', ghz(nb_qubits), range(nb_qubits)
    # gates = pauli_h_cnot(nb_qubits)

    # Teleport (no ancilla, so not true teleport)
    # targ_name, target,  = 'TELEPORT', sp.eye(2)
    # fixed_qubits, post_select = [1], {0: 0}
    # gates = clifford_t_gates(2)

    phases, dvars = z3_mat_solve(target, gates, 3, fixed_qubits=fixed_qubits,
                                 post_select=post_select,
                                 sat=True, export=targ_name)

    # Ne pas oublier de choisir à la main le bon jeu de portes.
    # sol_filename = "CZ_gates18_deep3.out"
    # phases, dvars = load_sat_sol(sol_filename, gates)
    # fixed_qubits, post_select = read_metadata(sol_filename)

    if phases is not None and dvars is not None:
        phi, res = parse_result(gates, dvars, phases, target=target,
                                fixed_qubits=fixed_qubits,
                                post_select=post_select)
