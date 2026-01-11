Quantum gate SAT synthesis
==========================

This is the repository associated with the paper [arXiv:2503.15452](https://arxiv.org/abs/2503.15452).
The method is introduced in the paper.
Updated for parallel gates on classical reversible case following [arXiv:2602.05425](https://arxiv.org/abs/2602.05425).

Files are:
* sat_synthesis.py: main file: implementation of the translation of the exact gate synthesis problem to SAT
* algebraic_extension.py: defines classes to handle matrices of extended integers
* gates.py: define gates and gate sets
* classical_rev_circuits.py: synthesis of classical reversible circuits through SAT
* mip_synthesis.py: mixed-integer linear programming version of the method
* conditionnaly_clean.py: not really part of the project; playing around the [arXiv:2407.17966](https://arxiv.org/abs/2407.17966) with a SAT solver (to conjecture that they reached optimality)

This code is provided "as it is", i.e. not clean, with comments in French, and some choices that rely on commenting/uncommenting.
If you would like a cleaner code and proper packaging, please open an issue or better, a merge request.

The Mixed-integer linear programming approach requires a Gurobi licence.
I don't have such licence anymore, so it has not been tested since a long time, is likely to fail because of changes of interfaces in the other files, and I can't help on it.
A port to licence-free solver might be considered if there is interest.
