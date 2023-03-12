# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------

import timeit
import numpy as np
import pandas as pd
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

# --------------------------------------------------------------------
# ************************  Functions *********************************
# --------------------------------------------------------------------

def func(problem, mapper):
    start = timeit.default_timer()
    np.set_printoptions(precision=4, suppress=True)
    # Driver
    fermionic_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)
    # Info
    num_qubits = qubit_op.num_qubits
    num_fer_op_terms = len(fermionic_op.items())
    num_pauli_op_lines = len(qubit_op.to_pauli_op())
    depth = len(qubit_op.to_pauli_op())
    num_spin_orbitals = problem.num_spin_orbitals
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles
    register_length = fermionic_op.register_length
    num_gates = num_pauli_op_lines * num_qubits

    # runtime
    stop = timeit.default_timer()
    runtime = stop - start
    df = pd.DataFrame({
        '#qubits': num_qubits,
        '#Fermionic operator terms': num_fer_op_terms,
        '#Pauli operator lines': num_pauli_op_lines,
        '#Depth': depth,
        '#Spin orbitals': num_spin_orbitals,
        '#Spatial orbitals': num_spatial_orbitals,
        '#particles': num_particles,
        'Register Length': register_length,
        '#Ham_gates': num_gates,
        'Runtime(min)': runtime / 60
    })
    return df

# --------------------------------------------------------------------
# **************************  Dictionary ****************************
# --------------------------------------------------------------------
#   Molecule:
molecule = 'C; ' \
           'O 1 1.19619; ' \
           'O 1 1.19638 2 179.97438; ' \
           'Zn 3 1.79541 1 120.07056 2 180.00000'
#   Essential arguments:

basis_set = 'sto3g'
spin = 0
charge = 2
mapper = [BravyiKitaevMapper(), JordanWignerMapper(), ParityMapper()]
mapper=mapper[2]

#   dataframe:
name_excel = 'info_znco2.xlsx'


#   Active Space Transformation:
# Number of active electrons ---> 2hl
active_electrons_2hl = 2
active_orbitals_2hl = 2
# Number of active electrons ---> 4hl
active_electrons_4hl = 4
active_orbitals_4hl = 4
# Number of active electrons ---> 6hl
active_electrons_6hl = 6
active_orbitals_6hl = 6
# Number of active electrons ---> 8hl
active_electrons_8hl = 8
active_orbitals_8hl = 8
# Number of active electrons ---> 10hl
active_electrons_10hl = 10
active_orbitals_10hl = 10

#   List of all active orbitals ---> A list of integers
active_orbitals_list_2hl = [24, 25]
active_orbitals_list_4hl = [23, 24, 25, 26]
active_orbitals_list_6hl = [22, 23, 24, 25, 26, 27]
active_orbitals_list_8hl = [21, 22, 23, 24, 25, 26, 27, 28]
active_orbitals_list_10hl = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# --------------------------------------------------------------------
# **********************  Constructing Molecule *********************
# --------------------------------------------------------------------
driver = PySCFDriver(molecule.format(), unit=DistanceUnit.ANGSTROM, charge=charge, spin=spin, basis=basis_set)
full_problem = driver.run()
# ---------------- Reduction To HOMO LUMO Problem ----------------
transformer_2hl = ActiveSpaceTransformer(active_electrons_2hl, active_orbitals_2hl,
                                         active_orbitals=active_orbitals_list_2hl)
transformer_4hl = ActiveSpaceTransformer(active_electrons_4hl, active_orbitals_4hl,
                                         active_orbitals=active_orbitals_list_4hl)
transformer_6hl = ActiveSpaceTransformer(active_electrons_6hl, active_orbitals_6hl,
                                         active_orbitals=active_orbitals_list_6hl)
transformer_8hl = ActiveSpaceTransformer(active_electrons_8hl, active_orbitals_8hl,
                                         active_orbitals=active_orbitals_list_8hl)
transformer_10hl = ActiveSpaceTransformer(active_electrons_10hl, active_orbitals_10hl,
                                         active_orbitals=active_orbitals_list_10hl)

problem_2hl = transformer_2hl.transform(full_problem)
problem_4hl = transformer_4hl.transform(full_problem)
problem_6hl = transformer_6hl.transform(full_problem)
problem_8hl = transformer_8hl.transform(full_problem)
problem_10hl = transformer_10hl.transform(full_problem)

# --------------------------------------------------------------------
# ************************  Hamiltonian *****************************
# --------------------------------------------------------------------
# ------- Qubit Encoding Full Problem -----------
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

# --------------------------------------------------------------------
# *****************************  Job *********************************
# --------------------------------------------------------------------
info_2hl = func(problem_2hl, mapper)
info_4hl = func(problem_4hl, mapper)
info_6hl = func(problem_6hl, mapper)
info_8hl = func(problem_8hl, mapper)

info_10hl = func(problem_10hl, mapper)
info_full = func(full_problem, mapper)


with pd.ExcelWriter(name_excel) as writer:
    info_2hl.to_excel(writer, sheet_name='info_2hl', index=False)
    info_4hl.to_excel(writer, sheet_name='info_4hl', index=False)
    info_6hl.to_excel(writer, sheet_name='info_6hl', index=False)
    info_8hl.to_excel(writer, sheet_name='info_8hl', index=False)
    info_10hl.to_excel(writer, sheet_name='info_10hl', index=False)
    info_full.to_excel(writer, sheet_name='info_full', index=False)



