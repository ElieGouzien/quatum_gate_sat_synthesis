#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code design classical reversible circuits.

@author: Élie Gouzien
"""

from math import comb
from itertools import permutations, product
import time
import datetime
from functools import reduce
from operator import and_
import warnings
import gzip
import subprocess
import os.path

import z3

from sat_synthesis import export_dimacs, load_sat_sol


# %% Solver with explicit inputs
def _set_not(solver, dvar, targ, vars_before, vars_after):
    """Impose the condition for a NOT."""
    for i, (var1, var2) in enumerate(zip(vars_before, vars_after)):
        if i == targ:
            solver.add(z3.simplify(
                z3.Implies(dvar, var2 == z3.Not(var1))))
        else:
            solver.add(z3.simplify(
                z3.Implies(dvar, var2 == var1)))


def _set_cnot(solver, dvar, ctrl, targ, vars_before, vars_after):
    """Impose the condition for a CNOT."""
    for i, (var1, var2) in enumerate(zip(vars_before, vars_after)):
        if i == targ:
            solver.add(z3.simplify(
                z3.Implies(dvar, var2 == z3.Xor(var1, vars_before[ctrl]))))
        else:
            solver.add(z3.simplify(
                z3.Implies(dvar, var2 == var1)))


def _set_toffoli(solver, dvar, ctrl1, ctrl2, targ, vars_before, vars_after):
    """Impose the condition for a Toffoli."""
    for i, (var1, var2) in enumerate(zip(vars_before, vars_after)):
        if i == targ:
            solver.add(z3.simplify(
                z3.Implies(dvar, var2 == z3.Xor(var1,
                                                z3.And(vars_before[ctrl1],
                                                       vars_before[ctrl2])))))
        else:
            solver.add(z3.simplify(
                z3.Implies(dvar, var2 == var1)))


class Nothing:
    """Class for objects that do nothing."""

    @staticmethod
    def nothing(*args, **kw):
        """Identity fonction."""
        pass

    def __getattr__(self, _):
        """Any attribute is the nothing function."""
        return self.nothing


def solve_classical_circuit(nb_qubits, deep, max_toff=None, max_cnot=None,
                            export=False, dry_run=False):
    """Fonction pour concevoir un circuit classique réversible.

    Pour l'instant les contraintes définissant la fonction sont définies en
    modifiant cette fonction. C'est crade.

    Version avec autant de conditions que d'entrées possibles.
    """
    # TODO: faire une entrée propre de la spécification du circuit.
    #       et en profiter pour dégager "dry_run"
    # Création du model
    if nb_qubits < 3:
        warnings.warn("Il faut au moins 3 qubits pour faire des Toffolis !")
    nb_gates = (nb_qubits  # NOTs
                + (nb_qubits**2-nb_qubits)   # CNOTs
                + nb_qubits*comb(nb_qubits-1, 2))  # Toffolis
    if not dry_run:
        print("Construction du problème…")
    if dry_run:
        solver = Nothing()
    elif export:
        solver = z3.Goal()
    else:
        solver = z3.Solver()
    z3.set_option(verbose=2)
    z3_dvars = [z3.BoolVector(f'x{depth}', nb_gates)
                for depth in range(deep)]

    # Un choix de porte par tranche
    for moment_dvars in z3_dvars:
        solver.add(z3.AtMost(*moment_dvars, 1))
        solver.add(z3.Or(*moment_dvars))
    if max_toff is not None:
        toff_index = nb_qubits + (nb_qubits**2-nb_qubits)
        solver.add(z3.AtMost(
            *sum((moment_dvars[toff_index:] for moment_dvars in z3_dvars),
                 start=[]),
            max_toff))
        # Exclusion des Toffoli ayant pour origine le bit classique
        # TODO: en faire une option, et une sous-fonction.
        # toff_index = []
        # gate_id = 0
        # for i in range(nb_qubits):
        #     gate_id += 1
        # for i, j in permutations(range(nb_qubits), 2):
        #     gate_id += 1
        # for i, j, k in permutations(range(nb_qubits), 3):
        #     if not i < j:
        #         continue
        #     if 1 not in (i, j):  # On ne compte pas contrôle classique
        #         toff_index.append(gate_id)
        #     gate_id += 1
        # solver.add(z3.AtMost(*[moment_dvars[i] for moment_dvars in z3_dvars
        #                        for i in toff_index],
        #                      max_toff))
    if max_cnot is not None:
        cnot_index = nb_qubits
        toff_index = nb_qubits + (nb_qubits**2-nb_qubits)
        solver.add(z3.AtMost(
            *sum((moment_dvars[cnot_index:toff_index]
                  for moment_dvars in z3_dvars), start=[]), max_cnot))

    # Ici le "pour tout" est fait en énumérant explicitement toutes les
    # possibilités. Voir solve_classical_circuit_forall pour une manière plus
    # propre.
    ancillas = [nb_qubits-1]
    for init_vars_id, init_vars in enumerate(product((0, 1), repeat=nb_qubits)):
        # Mettre ici les exclusions de cas qu'on ne veut pas traiter
        if init_vars[-1] != 0:  # Ancilla
            continue
        # if init_vars[3] != (init_vars[0] & ((init_vars[1] & init_vars[2])
        #                                     ^ init_vars[1] ^ init_vars[2])):
        #     continue
        # if init_vars[3] != ((init_vars[0] & init_vars[1])
        #                     ^ (init_vars[0] & init_vars[2])
        #                     ^ (init_vars[1] & init_vars[2])):
        #     continue
        z3_int_vals = [z3.BoolVector(f"val_{init_vars_id}_{depth}", nb_qubits)
                       for depth in range(deep+1)]
        # Imposition du début
        for i in range(nb_qubits):
            solver.add(z3_int_vals[0][i] == z3.BoolVal(init_vars[i]))
        # Impositions des contraintes de succession
        # TODO: charger les portes depuis la fonction gen_gates() (ou en faire
        #       une nouvelle qui énumère les portes et leur type.)
        for depth in range(deep):
            gate_id = 0
            for i in range(nb_qubits):
                _set_not(solver, z3_dvars[depth][gate_id], i,
                         z3_int_vals[depth],
                         z3_int_vals[depth+1])
                gate_id += 1
            for i, j in permutations(range(nb_qubits), 2):
                # Exemple de forçage de porte
                # if depth == deep - 1 and (i, j) == (1, 2):
                #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(True))
                # Exemple d'interdiction de portes (cas classique)
                # if j == 1:  # wire 1 classical
                #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(False))
                _set_cnot(solver, z3_dvars[depth][gate_id], i, j,
                          z3_int_vals[depth],
                          z3_int_vals[depth+1])
                gate_id += 1
            for i, j, k in permutations(range(nb_qubits), 3):
                if not i < j:
                    continue
                # Exemple de forçage de porte
                # if depth == 1 and (i, j, k) == (1, 2, 3):
                #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(True))
                # Exemple d'interdiction de portes classiques
                # if k == 1:  # wire 1 classical
                #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(False))
                _set_toffoli(solver, z3_dvars[depth][gate_id], i, j, k,
                             z3_int_vals[depth], z3_int_vals[depth+1])
                gate_id += 1
        # Imposition de la fonction à accomplir
        # TODO : rendre ça programmable extérieurement ; ici c'est le calcul de
        # la retenue
        # if nb_qubits != 3:
        #     raise NotImplementedError("Pour l'instant la fonction est codée en dur.")
        # Addition semi-classique
        # g, c, x = init_vars  # pylint: disable=C0103
        # solver.add(z3_int_vals[deep][2] == z3.BoolVal(c & x))
        # Addition semi-classique avec qubit auxiliaire
        # ctrl, c, x, cc = init_vars  # pylint: disable=C0103
        # solver.add(z3_int_vals[deep][0] == z3.BoolVal(ctrl))
        # solver.add(z3_int_vals[deep][1] == z3.BoolVal(c & ctrl))
        # solver.add(z3_int_vals[deep][2] == z3.BoolVal(x ^ (c & ctrl) ^ ctrl))
        # solver.add(z3_int_vals[deep][3] == z3.BoolVal(0))
        # Addition quantique
        # c, y, x = init_vars  # pylint: disable=C0103
        # solver.add(z3_int_vals[deep][2] == z3.BoolVal((c & x) ^ (c & y) ^ (x & y)))
        # Addition semi-classique : UMA controlé
        # c, p, x, cc, ctrl = init_vars  # pylint: disable=C0103
        # solver.add(z3_int_vals[deep][0] == z3.BoolVal(c))
        # solver.add(z3_int_vals[deep][1] == z3.BoolVal(p))
        # solver.add(z3_int_vals[deep][2] == z3.BoolVal(x ^ (ctrl & (c ^ p))))
        # solver.add(z3_int_vals[deep][3] == z3.BoolVal(False))
        # solver.add(z3_int_vals[deep][4] == z3.BoolVal(ctrl))
        # Gros &&&&&
        # solver.add(z3_int_vals[deep][0] == z3.BoolVal(
        #     reduce(and_, init_vars[1:]) ^ init_vars[0]))
        # for var, val in zip(z3_int_vals[deep][1:], init_vars[1:]):
        #     solver.add(var == z3.BoolVal(val))
        # Gros &&& avec un auxiliaire sale
        if len(ancillas) == 1:
            gate_name = "Toffoli-clean-ancilla"
        else:
            gate_name = "Toffoli-dirty-ancilla"
        solver.add(z3_int_vals[deep][0] == z3.BoolVal(
            reduce(and_, init_vars[1:-1]) ^ init_vars[0]))
        for var, val in zip(z3_int_vals[deep][1:], init_vars[1:]):
            solver.add(var == z3.BoolVal(val))

    if dry_run:
        return None, gate_name

    if export:
        export_dimacs(solver,
                      filename_root(gate_name, nb_qubits, deep,
                                    max_toff, max_cnot) + '.cnf',
                      init_vars=sum(z3_dvars, []),
                      verb=True,
                      gates_dict={g: None for g in gen_gates(nb_qubits)})
        return None, gate_name

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
    return dvars, gate_name


def auto_solve_classical_circuit(nb_qubits, deep, max_toff=None, max_cnot=None,
                                 export=False):
    """Automatically run `solve_classical_circuit` and kissat."""
    if export:
        print("Dry run to get the gate name…")
        _, gate_name = solve_classical_circuit(
            nb_qubits, deep, max_toff=max_toff, max_cnot=max_cnot,
            export=export, dry_run=True)
        filename = filename_root(gate_name, nb_qubits, deep,
                                 max_toff, max_cnot)
    if not export or not os.path.isfile(filename+'.cnf.gz'):
        dvars, gate_name = solve_classical_circuit(
            nb_qubits, deep, max_toff=max_toff, max_cnot=max_cnot,
            export=export)
    else:
        print("DIMACS file already exist, skip problem formulation…")
    if export:
        if not os.path.isfile(filename+'.out.gz'):
            with gzip.open(filename+'.out.gz', 'wt') as outfile:
                print("Calling Kissat...")
                with subprocess.Popen(
                        ["kissat", filename+'.cnf.gz'],
                        stdout=subprocess.PIPE, text=True) as proc:
                    for line in proc.stdout:
                        print(line, end='')
                        print(line, file=outfile, end='')
            print("\n"*2)
        else:
            print("Solution file already exist, skip resolution…")
        print("Reading solution...")
        _, dvars = load_sat_sol(filename+'.out.gz',
                                gen_gates(nb_qubits), deep, 0)
    return dvars


# %% Solver with forall
def _not_update_var(dvar, targ, vars_before, vars_after):
    """Update vars_after to take into account the NOT gate."""
    for i, (var1, var2) in enumerate(zip(vars_before, vars_after)):
        if i == targ:
            vars_after[i] = z3.If(dvar, z3.Not(var1), var2)
        else:
            vars_after[i] = z3.If(dvar, var1, var2)


def _cnot_update_var(dvar, ctrl, targ, vars_before, vars_after):
    """Update vars_after to take into account the CNOT gate."""
    for i, (var1, var2) in enumerate(zip(vars_before, vars_after)):
        if i == targ:
            vars_after[i] = z3.If(dvar, z3.Xor(var1, vars_before[ctrl]), var2)
        else:
            vars_after[i] = z3.If(dvar, var1, var2)


def _toffoli_update_var(dvar, ctrl1, ctrl2, targ, vars_before, vars_after):
    """Impose the condition for a Toffoli."""
    for i, (var1, var2) in enumerate(zip(vars_before, vars_after)):
        if i == targ:
            vars_after[i] = z3.If(dvar, z3.Xor(var1, z3.And(vars_before[ctrl1],
                                                            vars_before[ctrl2])
                                               ), var2)
        else:
            vars_after[i] = z3.If(dvar, var1, var2)


def solve_classical_circuit_forall(nb_qubits, deep,
                                   max_toff=None, max_cnot=None):
    """Fonction pour concevoir un circuit classique réversible.

    Pour l'instant les contraintes définissant la fonction sont définies en
    modifiant cette fonction. C'est crade.

    Version avec des ForAll. En pratique semble beaucoup plus lent que l'autre.
    """
    # TODO: factoriser ce qui se peut entre les fonctions.
    # TODO: faire une entrée propre de la spécification du circuit.
    # Création du model
    if nb_qubits < 3:
        warnings.warn("Il faut au moins 3 qubits pour faire des Toffolis !")
    nb_gates = (nb_qubits  # NOTs
                + (nb_qubits**2-nb_qubits)   # CNOTs
                + nb_qubits*comb(nb_qubits-1, 2))  # Toffolis
    print("Construction du problème…")
    solver = z3.Solver()
    z3.set_option(verbose=2)
    z3_dvars = [z3.BoolVector(f'x{depth}', nb_gates)
                for depth in range(deep)]

    # Un choix de porte par tranche
    for moment_dvars in z3_dvars:
        solver.add(z3.AtMost(*moment_dvars, 1))
        solver.add(z3.Or(*moment_dvars))
    if max_toff is not None:
        toff_index = nb_qubits + (nb_qubits**2-nb_qubits)
        solver.add(z3.AtMost(
            *sum((moment_dvars[toff_index:] for moment_dvars in z3_dvars),
                 start=[]), max_toff))
        # Exclusion des Toffoli ayant pour origine le bit classique
        # TODO: en faire une option, et une sous-fonction.
        # toff_index = []
        # gate_id = 0
        # for i in range(nb_qubits):
        #     gate_id += 1
        # for i, j in permutations(range(nb_qubits), 2):
        #     gate_id += 1
        # for i, j, k in permutations(range(nb_qubits), 3):
        #     if not i < j:
        #         continue
        #     if 1 not in (i, j):  # On ne compte pas contrôle classique
        #         toff_index.append(gate_id)
        #     gate_id += 1
        # solver.add(z3.AtMost(*[moment_dvars[i] for moment_dvars in z3_dvars
        #                        for i in toff_index],
        #                      max_toff))
    if max_cnot is not None:
        cnot_index = nb_qubits
        toff_index = nb_qubits + (nb_qubits**2-nb_qubits)
        solver.add(z3.AtMost(
            *sum((moment_dvars[cnot_index:toff_index]
                  for moment_dvars in z3_dvars), start=[]), max_cnot))

    # Valeurs initiales (sur lesquelles le forall s'applique)
    # TODO: donner la possibilité de fixer des valeurs pour auxiliaires.
    z3_init_vals = z3.BoolVector("init_val", nb_qubits)
    # Impositions des contraintes de succession
    # TODO: charger les portes depuis la fonction gen_gates() (ou en faire
    #       une nouvelle qui énumère les portes et leur type.)

    bool_vals = z3_init_vals
    for depth in range(deep):
        bool_vals_new = [z3.BoolVal(False)]*nb_qubits
        gate_id = 0
        for i in range(nb_qubits):
            _not_update_var(z3_dvars[depth][gate_id], i,
                            bool_vals, bool_vals_new)
            gate_id += 1
        for i, j in permutations(range(nb_qubits), 2):
            # Exemple de forçage de porte
            # if depth == deep - 1 and (i, j) == (1, 2):
            #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(True))
            # Exemple d'interdiction de portes (cas classique)
            # if j == 1:  # wire 1 classical
            #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(False))
            _cnot_update_var(z3_dvars[depth][gate_id], i, j,
                             bool_vals, bool_vals_new)
            gate_id += 1
        for i, j, k in permutations(range(nb_qubits), 3):
            if not i < j:
                continue
            # Exemple de forçage de porte
            # if depth == 1 and (i, j, k) == (1, 2, 3):
            #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(True))
            # Exemple d'interdiction de portes classiques
            # if k == 1:  # wire 1 classical
            #     solver.add(z3_dvars[depth][gate_id] == z3.BoolVal(False))
            _toffoli_update_var(z3_dvars[depth][gate_id], i, j, k,
                                bool_vals, bool_vals_new)
            gate_id += 1
        bool_vals = bool_vals_new
        del bool_vals_new
    # Imposition de la fonction à accomplir
    # TODO : rendre ça programmable extérieurement
    # HINT: la fonction solve_classical_circuit contient plein d'exemples dont
    # on peut s'inspirer.
    # Gros &&&&&
    # solver.add(z3.ForAll(z3_init_vals, bool_vals[0] == reduce(
    #     and_, z3_init_vals[1:]) ^ z3_init_vals[0]))
    # for var, init_var in zip(bool_vals[1:], z3_init_vals[1:]):
    #     solver.add(z3.ForAll(z3_init_vals, var == init_var))
    # Gros &&& avec un auxiliaire sale
    solver.add(z3.ForAll(z3_init_vals, bool_vals[0] == reduce(
        and_, z3_init_vals[1:-1]) ^ z3_init_vals[0]))
    for var, init_var in zip(bool_vals[1:], z3_init_vals[1:]):
        solver.add(z3.ForAll(z3_init_vals, var == init_var))

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


# %% Gates building + exports
def gen_gates(nb_qubits):
    """Crée la liste des portes, sous forme d'une liste."""
    nb_gates = (nb_qubits  # NOTs
                + (nb_qubits**2-nb_qubits)   # CNOTs
                + nb_qubits*comb(nb_qubits-1, 2))  # Toffolis
    gates = []
    for i in range(nb_qubits):
        gates.append(f"NOT_{i}")
    for i, j in permutations(range(nb_qubits), 2):
        gates.append(f"CNOT_{i}_{j}")
    for i, j, k in permutations(range(nb_qubits), 3):
        if not i < j:
            continue
        gates.append(f"TOFFOLI_{i}_{j}_{k}")
    assert len(gates) == nb_gates
    return gates


def filename_root(gate_name, nb_qubits, deep, max_toff, max_cnot):
    """Generate the filename root for the problem."""
    return (f'classique_gate={gate_name}_{nb_qubits=}_{deep=}_{max_toff=}_'
            f'{max_cnot=}')


def parse_result(nb_qubits, dvars):
    """Parse a solution into a readable format.

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
    res_names = [gates[dvar.index(True)] for dvar in dvars]

    # Print result
    print("circuit = --", '--'.join(res_names), "--", sep='')
    return res_names


def _to_qirk_list(nb_qubits, gate):
    """Give the list for quirk URL corresponding to the gate."""
    res = ['1'] * nb_qubits
    match gate.split('_'):
        case ['NOT', i]:
            res[int(i)] = '"X"'
        case ['CNOT', i, j]:
            res[int(i)] = '"•"'
            res[int(j)] = '"X"'
        case ['TOFFOLI', i, j, k]:
            res[int(i)] = res[int(j)] = '"•"'
            res[int(k)] = '"X"'
        case _:
            raise ValueError(f"{gate} is not an allowed gate")
    return '[' + ','.join(res) + ']'


def quirk_url(nb_qubits, gates, *, display=False, open_quirk=False):
    """Build the quirk URL corresponding to the circuit."""
    quirk = "https://algassert.com/quirk"
    start = '#circuit={"cols":['
    end = ']}'
    url = quirk + start + ','.join(
        _to_qirk_list(nb_qubits, gate) for gate in gates) + end
    if display:
        print(url)
    if open_quirk:
        subprocess.run(["firefox", url], check=False)
    return url


# %% Main part
if __name__ == '__main__':
    nb_qubits, deep, max_toff, max_cnot = 5, 3, 3, None
    dvars = auto_solve_classical_circuit(nb_qubits, deep,
                                         max_toff=max_toff, max_cnot=max_cnot,
                                         export=True)
    # dvars = solve_classical_circuit_forall(nb_qubits, deep,
    #                                        max_toff=max_toff,
    #                                        max_cnot=max_cnot)

    if dvars is not None:
        gates = parse_result(nb_qubits, dvars)
        quirk_url(nb_qubits, gates, display=True, open_quirk=True)
