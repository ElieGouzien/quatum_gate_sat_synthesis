#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent to automatically derive scheme for solving Q1 of arXiv:2407.17966.

@author: Élie Gouzien
"""
from collections import namedtuple
from itertools import permutations
import time
import datetime
import subprocess

import z3

from classical_rev_circuits import quirk_url


# %% Fonctions auxiliaires
Operation = namedtuple('Operation', 'ctrl1 ctrl2 targ')


def gen_gates(nb_qubits):
    """Crée la liste des portes, sous forme d'une liste."""
    gates = []
    for i, j, k in permutations(range(nb_qubits), 3):
        if not k < i < j:
            continue
        gates.append(Operation(i, j, k))
    return gates


def parse_result(nb_qubits, dvars) -> list[Operation]:
    """Parse a solution into a list of operations.

    dvars are given into the "left to right chronological order".
    (dvars[0] describes the first gate of the circuit)
    No verification here.
    """
    gates = gen_gates(nb_qubits)
    if not len(dvars[0]) == len(gates):
        raise ValueError("dvars and gate set non consistents")

    # Retrieve gates
    if not all(sum(dvar) == 1 for dvar in dvars):
        raise ValueError("Only exactly one gate by time-slice allowed.")
    return [gates[dvar.index(True)] for dvar in dvars]


def simulate(nb_qbits, gates: list[Operation]) -> list[bool]:
    """Give the 0 and 1 resulting from the rules."""
    vect = [i == 0 for i in range(nb_qbits)]
    for g in gates:
        assert g.targ < g.ctrl1 < g.ctrl2
        assert vect[g.targ] and (not vect[g.ctrl1]) and (not vect[g.ctrl2])
        vect[g.targ] = False
        vect[g.ctrl1] = vect[g.ctrl2] = True
    return vect


# def parse_result(nb_qubits, dvars):
#     """Parse a solution into a readable format.

#     dvars are given into the "left to right chronological order".
#     (dvars[0] describes the first gate of the circuit)
#     No verification here.
#     """
#     gates = gen_gates(nb_qubits)
#     if not len(dvars[0]) == len(gates):
#         raise ValueError("dvars and gate set non consistents")

#     # Retrieve gates
#     if not all(sum(dvar) == 1 for dvar in dvars):
#         raise ValueError("Only exactly one gate by time-slice allowed.")
#     res_gates = [gates[dvar.index(True)] for dvar in dvars]
#     res_names = ["TOFFOLI_"+'_'.join(map(str, g)) for g in res_gates]

#     # Print result
#     print("circuit = --", '--'.join(res_names), "--", sep='')
#     return res_names


def make_complete_circuit(nb_qubits, gates_list, res_vec):
    """Build the complete circuit from the given gates."""
    if sum(res_vec) != nb_qubits - 2:
        raise ValueError("Works only when exactly 2 values are 0.")
    qubit1 = res_vec.index(False)
    qubit2 = res_vec.index(False, qubit1+1)
    gates = sum(([f"TOFFOLI_{g.ctrl1}_{g.ctrl2}_{g.targ}", f"NOT_{g.targ}"]
                 for g in gates_list), start=[])
    return (gates + [f"TOFFOLI_{qubit1}_{qubit2}_{nb_qubits}"]
            + list(reversed(gates)))


# %% Résolution avec variables implicites (tableau de valeurs z3)
def _set_operation(solver, dvar, gate: Operation, vars_before, vars_after):
    """Impose the condition for the operation."""
    for i, var in enumerate(vars_after):
        if i == gate.targ:
            solver.add(z3.simplify(
                z3.Implies(dvar, var == z3.BoolVal(False))))
        elif i in (gate.ctrl1, gate.ctrl2):
            solver.add(z3.simplify(
                z3.Implies(dvar, var == z3.BoolVal(True))))
        else:
            solver.add(z3.simplify(
                z3.Implies(dvar, var == vars_before[i])))


def solve_q1_implicit(nb_qubits, deep, min_ones):
    """Solve Q1 of arXiv:2407.17966, implicit intermediate vars."""
    print("Construction du problème…")
    solver = z3.Solver()
    z3.set_option(verbose=2)
    gates = gen_gates(nb_qubits)
    z3_dvars = [z3.BoolVector(f'xd={depth}', len(gates))
                for depth in range(deep)]

    # Un choix à la fois
    for moment_dvars in z3_dvars:
        solver.add(z3.AtMost(*moment_dvars, 1))
        solver.add(z3.Or(*moment_dvars))

    z3_vars = [z3.BoolVector(f"vars_{d}", nb_qubits) for d in range(deep+1)]
    # Conditions initiales
    for i, var in enumerate(z3_vars[0]):
        solver.add(var == z3.BoolVal(i == 0))
    for d in range(deep):
        # Works because any gate touches all qubits and we force at least one.
        for gate_id, gate in enumerate(gates):
            # If ctrl1 is 1, forbid to take the gate.
            solver.add(z3.Implies(z3_vars[d][gate.ctrl1] == z3.BoolVal(True),
                                  z3_dvars[d][gate_id] == z3.BoolVal(False)))
            # If ctrl2 is 1, forbid to take the gate.
            solver.add(z3.Implies(z3_vars[d][gate.ctrl2] == z3.BoolVal(True),
                                  z3_dvars[d][gate_id] == z3.BoolVal(False)))
            # If target is 0, forbid to take the gate.
            solver.add(z3.Implies(z3_vars[d][gate.targ] == z3.BoolVal(False),
                                  z3_dvars[d][gate_id] == z3.BoolVal(False)))
            # If gate, update values
            _set_operation(solver, z3_dvars[d][gate_id], gate,
                           z3_vars[d], z3_vars[d+1])

    # Condition de fin.
    solver.add(z3.AtLeast(*z3_vars[deep], min_ones))

    print("Résolution du problème…")
    start_time = time.time()
    status = solver.check()
    print("\nTemps de résolution :", datetime.timedelta(
        seconds=time.time() - start_time))
    print("Solver stats :", solver.statistics())
    print("Status :", status)

    # Parse result
    if status == z3.sat:
        model = solver.model()
        print(model)
        dvars = [[bool(model[v]) for v in var] for var in z3_dvars]
    else:
        dvars = None
    return dvars


# %% Resolution avec variables explicites (neasted z3.If())
def _update_var(dvar, gate: Operation, vars_before, vars_after,
                simplify=False):
    """Impose the condition for the operation.

    Simplify let z3 simplify the variables. In practice it make it slower.
    """
    for i, var in enumerate(vars_after):
        if i == gate.targ:
            vars_after[i] = z3.If(dvar, z3.BoolVal(False), var)
        elif i in (gate.ctrl1, gate.ctrl2):
            vars_after[i] = z3.If(dvar, z3.BoolVal(True), var)
        else:
            vars_after[i] = z3.If(dvar, vars_before[i], var)
        if simplify:
            vars_after[i] = z3.simplify(vars_after[i])


def solve_q1_explicit(nb_qubits, deep, min_ones):
    """Solve Q1 of arXiv:2407.17966, explicit intermediate vars.

    Seems to be faster than the implicit version (at least with z3 as solver).
    """
    print("Construction du problème…")
    solver = z3.Solver()
    z3.set_option(verbose=2)
    gates = gen_gates(nb_qubits)
    z3_dvars = [z3.BoolVector(f'xd={depth}', len(gates))
                for depth in range(deep)]

    # Un choix à la fois
    for moment_dvars in z3_dvars:
        solver.add(z3.AtMost(*moment_dvars, 1))
        solver.add(z3.Or(*moment_dvars))

    bool_vars = [z3.BoolVal(i == 0) for i in range(nb_qubits)]
    for d in range(deep):
        # Works because any gate touches all qubits and we force at least one.
        new_vars = [z3.BoolVal(False)] * nb_qubits
        for gate_id, gate in enumerate(gates):
            # If ctrl1 is 1, forbid to take the gate.
            solver.add(z3.Implies(bool_vars[gate.ctrl1] == z3.BoolVal(True),
                                  z3_dvars[d][gate_id] == z3.BoolVal(False)))
            # If ctrl2 is 1, forbid to take the gate.
            solver.add(z3.Implies(bool_vars[gate.ctrl2] == z3.BoolVal(True),
                                  z3_dvars[d][gate_id] == z3.BoolVal(False)))
            # If target is 0, forbid to take the gate.
            solver.add(z3.Implies(bool_vars[gate.targ] == z3.BoolVal(False),
                                  z3_dvars[d][gate_id] == z3.BoolVal(False)))
            # If gate, update values
            _update_var(z3_dvars[d][gate_id], gate, bool_vars, new_vars)
        bool_vars = new_vars

    # Condition de fin.
    solver.add(z3.AtLeast(*bool_vars, min_ones))

    print("Résolution du problème…")
    start_time = time.time()
    status = solver.check()
    print("\nTemps de résolution :", datetime.timedelta(
        seconds=time.time() - start_time))
    print("Solver stats :", solver.statistics())
    print("Status :", status)

    # Parse result
    if status == z3.sat:
        model = solver.model()
        print(model)
        dvars = [[bool(model[v]) for v in var] for var in z3_dvars]
    else:
        dvars = None
    return dvars


# %% Partie exécutable
if __name__ == '__main__':
    nb_qubits, deep, min_ones = 16, 11, None
    min_ones = nb_qubits - 2
    # dvars = solve_q1_implicit(nb_qubits, deep, min_ones)
    dvars = solve_q1_explicit(nb_qubits, deep, min_ones)

    if dvars is not None:
        res_gates = parse_result(nb_qubits, dvars)
        final_state = simulate(nb_qubits, res_gates)
        print("Final state:", final_state)
        if sum(final_state) != min_ones:
            print("Résultat meilleur que demandé !")
        if sum(final_state) == nb_qubits - 2:
            complete_circuit = make_complete_circuit(nb_qubits, res_gates,
                                                     final_state)
            url = quirk_url(nb_qubits+1, complete_circuit)
            url = url[:-1] + ',"init":[1]}'
        else:
            url = quirk_url(nb_qubits, [f"TOFFOLI_{g.ctrl1}_{g.ctrl2}_{g.targ}"
                                        for g in res_gates])
        print(url)
        subprocess.run(["firefox", url])
