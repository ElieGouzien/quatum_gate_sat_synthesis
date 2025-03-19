#!/usr/bin/env python3
# coding: utf-8
"""
Solve the gate design problem, with gurobi.

@author: Élie Gouzien
"""
import sys
import os
from cmath import pi, exp

import numpy as np
import sympy as sp
from gurobipy import Model, GRB, quicksum

from gates import parse_result
from gates import (H, Z, S, T, rot_z, dag, TOFFOLI, ghz, controlled, promote,
                   fix_inputs_zero)
from gates import GATES_EXAMPLE, clifford_t_gates, gidney_problem_set
from gates import min_clifford_t_gates, min_clifford_t_gates_and, pauli_h_cnot


def _add_np_var(model, shape, bound, **kwargs):
    """Create a tupledict var and translate it into a numpy object array."""
    if 'lb' in kwargs:
        lb = kwargs['lb']
        del kwargs['lb']
    else:
        lb = -bound
    tupledict = model.addVars(*shape, lb=lb, ub=bound, **kwargs)
    return np.asarray(tupledict.values()).reshape(shape)


def gurobi_mat_solve(target, gates, deep, nb_phases=16, fixed_qubits=(), *,
                     verb=True, optim=True):
    """Build and solve the problem, with gurobi.

    It uses matrix expressions an general constraints to propagate the value
    of the matrix.

    If nb_phases is None, use continuous variables with nonconvex quadratic
    constraint.
    """
    # TODO: essayer avec une expression quadratiques
    out = sys.stdout if verb else open(os.devnull, 'w', encoding='utf-8')

    print("\tConversion des portes en numérique…", file=out)
    gates_shape = next(iter(gates.values())).shape
    assert all(gate.shape == gates_shape for gate in gates.values())
    complex_gates = [np.asarray(gate).astype(complex)
                     for gate in gates.values()]
    target = fix_inputs_zero(np.asarray(target).astype(complex), fixed_qubits)
    shape = target.shape

    print("Création du modèle et des variables…", file=out)
    model = Model()
    if nb_phases is None:
        grb_re_phase = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS,
                                    name='phase_re')
        grb_im_phase = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS,
                                    name='phase_im')
    else:
        grb_phases = model.addVars(nb_phases, name='phase', vtype=GRB.BINARY)
    # dvars are in the chronological order, not as in abstract_prob.py
    grb_dvars = [model.addVars(len(gates), name=f'x{depth}', vtype=GRB.BINARY)
                 for depth in range(deep)]

    # Intermediate matrices
    # HINT: Unitary matrices coefficients modulus <= 1 (U^\dag @ U = \identity)
    grb_re_mats = [_add_np_var(model, shape, 1, vtype=GRB.CONTINUOUS,
                               name=f"mat_re_{i}") for i in range(1, deep)]
    grb_im_mats = [_add_np_var(model, shape, 1, vtype=GRB.CONTINUOUS,
                               name=f"mat_im_{i}") for i in range(1, deep)]

    # Result matrices
    if optim:
        grb_res_mat_re = _add_np_var(model, shape, 1, vtype=GRB.CONTINUOUS,
                                     name="mat_res_re")
        grb_res_mat_im = _add_np_var(model, shape, 1, vtype=GRB.CONTINUOUS,
                                     name="mat_res_im")
        grb_res_mat_abs_re = _add_np_var(model, shape, 1, lb=0,
                                         vtype=GRB.CONTINUOUS,
                                         name="mat_res_abs_re")
        grb_res_mat_abs_im = _add_np_var(model, shape, 1, lb=0,
                                         vtype=GRB.CONTINUOUS,
                                         name="mat_res_abs_im")

    # Exclusion conditions : one phase, one gate by time slot
    if nb_phases is None:
        model.addConstr(grb_re_phase*grb_re_phase
                        + grb_im_phase*grb_im_phase == 1)
    if nb_phases is not None:
        model.addConstr(grb_phases.sum() == 1)
    # model.addConstrs(dvars.sum() == 1 for dvars in grb_dvars)
    for dvars in grb_dvars:
        model.addConstr(quicksum(v for v in dvars.values()) == 1)

    # Contraintes de propagation de l'information.
    print("Imposition des contraintes de succession", file=out)
    # 1ere porte : trivial
    print(f"\tCouche 1/{deep}", file=out)
    grb_re_mats.insert(0, sum(fix_inputs_zero(mat, fixed_qubits).real*var
                              for var, mat in zip(grb_dvars[0].values(),
                                                  complex_gates)))
    grb_im_mats.insert(0, sum(fix_inputs_zero(mat, fixed_qubits).imag*var
                              for var, mat in zip(grb_dvars[0].values(),
                                                  complex_gates)))

    # portes suivantes
    for i in range(1, deep):
        print(f"\tCouche {i+1}/{deep}", file=out)
        for var, mat in zip(grb_dvars[i].values(), complex_gates):
            re_res = mat.real@grb_re_mats[i-1] - mat.imag@grb_im_mats[i-1]
            im_res = mat.imag@grb_re_mats[i-1] + mat.real@grb_im_mats[i-1]
            for c in np.ndindex(shape):
                model.addGenConstrIndicator(var, True,
                                            grb_re_mats[i][c] == re_res[c])
                model.addGenConstrIndicator(var, True,
                                            grb_im_mats[i][c] == im_res[c])

    # build phased target
    if nb_phases is None:
        phase_re, phase_im = grb_re_phase, grb_im_phase
    else:
        phi = exp(2j*pi/nb_phases)
        phase_re = sum((phi**k).real * ph
                       for k, ph in enumerate(grb_phases.values()))
        phase_im = sum((phi**k).imag * ph
                       for k, ph in enumerate(grb_phases.values()))
    phased_target_re = target.real*phase_re - target.imag*phase_im
    phased_target_im = target.imag*phase_re + target.real*phase_im

    # Force result matrix value
    res_mat_re = grb_re_mats[-1] - phased_target_re
    res_mat_im = grb_im_mats[-1] - phased_target_im
    if optim:
        for c in np.ndindex(shape):
            model.addConstr(res_mat_re[c] == grb_res_mat_re[c])
            model.addConstr(res_mat_im[c] == grb_res_mat_im[c])
            model.addGenConstrAbs(grb_res_mat_abs_re[c], grb_res_mat_re[c])
            model.addGenConstrAbs(grb_res_mat_abs_im[c], grb_res_mat_im[c])

        model.setObjective(grb_res_mat_abs_re.sum() + grb_res_mat_abs_im.sum(),
                           GRB.MINIMIZE)
    else:
        for c in np.ndindex(shape):
            model.addConstr(res_mat_re[c] == 0)
            model.addConstr(res_mat_im[c] == 0)

    # Solve model
    if nb_phases is None:
        model.Params.NonConvex = 2
    model.optimize()

    # Parse result
    if model.status in (GRB.OPTIMAL, GRB.INTERRUPTED):
        phases = ([p.x > 0.5 for p in grb_phases.values()]
                  if nb_phases is not None else None)
        dvars = [[v.x > 0.5 for v in var.values()] for var in grb_dvars]
        dvars.reverse()  # back to left to right product order
    else:
        phases, dvars = None, None
    return phases, dvars, model


if __name__ == '__main__':
    # H
    # targ_name, target, fixed_qubits = 'H', H, []
    # gates = GATES_EXAMPLE

    # Single qubit rotation
    # Example taken from arXiv:quant-ph/0306018
    # Result to be compared with arXiv:1403.2975
    # targ_name, target, fixed_qubits = 'R_theta', rot_z(sp.pi/128), []
    # gates = {'H': H, 'T': T, 'T_DAG': dag(T)}

    # CZ
    targ_name, target, fixed_qubits = 'CZ', controlled(2, 0, 1, Z), []
    gates = clifford_t_gates(2)
    del gates['CZ_0_1']

    # S, as asked in https://twitter.com/CraigGidney/status/1379593866563743748
    # targ_name, target, fixed_qubits = 'S_0', promote(2, 0, S), [1]
    # gates = gidney_problem_set(2)

    # AND
    # targ_name, target, fixed_qubits = 'AND', TOFFOLI, [2]
    # gates = clifford_t_gates(3)
    # gates = min_clifford_t_gates_and(3)

    # Toffoli
    # targ_name, target, fixed_qubits = 'TOFFOLI', TOFFOLI, []
    # gates = clifford_t_gates(3)
    # gates = min_clifford_t_gates(3)

    # GHZ
    # nb_qubits = 3
    # targ_name, target, fixed_qubits = 'GHZ', ghz(nb_qubits), range(nb_qubits)
    # gates = pauli_h_cnot(nb_qubits)

    # Résolution
    phases, dvars, model = gurobi_mat_solve(target, gates, 3, optim=False,
                                            fixed_qubits=fixed_qubits)
    if dvars is not None:
        phi, res = parse_result(gates, dvars, phases, target=target,
                                fixed_qubits=fixed_qubits)


# %% Debugging functions
def show_np_grb_mat_val(array_real, array_imag, ndigits=10):
    """Montre la valeur résultat du model pour la matrice ; pour débug."""
    shape = array_real.shape
    assert array_imag.shape == shape
    round_res = np.vectorize(lambda x: round(x.x, ndigits))
    return round_res(array_real) + 1j*round_res(array_imag)
