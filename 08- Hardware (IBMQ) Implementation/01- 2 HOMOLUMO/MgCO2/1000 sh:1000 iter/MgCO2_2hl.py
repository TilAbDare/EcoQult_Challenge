# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------

import timeit

# IBMQ.providers()
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyscf import gto
from qiskit import IBMQ
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.runtime import VQEClient
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit

# --------------------------------------------------------------------
# ***************************  IBMQ Acces ****************************
# --------------------------------------------------------------------
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials on disk.
# QiskitRuntimeService.save_account(channel='ibm_quantum', token=<IBM Quantum API key>)


# IBMQ.save_account("02796e1792cecd11233e343cba83b0e9637e48ac78fba13a82c791cb905500adc64f70ae5f4a5da11b5fe8525c19e5fc2d890c5ecd7daccd0d374557beec5ce4", overwrite=True)
IBMQ.load_account()  # Load account from disk
provider = IBMQ.get_provider(hub='deloitte-event23', group='level-1-access', project='ecocult')
backend = provider.get_backend("ibmq_qasm_simulator")

# --------------------------------------------------------------------
# ***************************  Inputs ********************************
# --------------------------------------------------------------------
#   dataframe and ploting:
name_excel = 'MgCO2_2hl_1000sh500iter.xlsx'
name_plot = 'CO2-Mg(2+)'
name_as = '1HOMO-1LUMO'
format = 'png'
dpi = 1200

#   Active Space Transformation:
# Number of active electrons ---> An integer
a_el_ion = 2
a_el_target = 2
a_el_ligand = 2

# Number of active orbitals ---> An integer
a_or_ion = 2
a_or_target = 2
a_or_ligand = 2

#   List of all active orbitals ---> A list of integers
a_mos_ion = [4, 5]
a_mos_target = [10, 11]
a_mos_ligand = [15, 16]

#   Molecules and their geometries in z-matrices format:
ion = 'Mg'
target = 'C;' \
         'O 1 1.19700;' \
         'O 1 1.19700 2 179.97438'
ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 1.97923 1 120.07450 2 180.00000'
#   The point to be varied, make it blank ---> {}
ligand_loop = 'C; ' \
              'O 1 1.19630; ' \
              'O 1 1.19628 2 179.97438; ' \
              'Mg 3 {} 1 120.07450 2 180.00000'

#   Essential arguments for Computation section:
basis_set = 'sto3g'
bonding_distances = np.arange(0.50, 5.00, 0.15)
mapper = BravyiKitaevMapper()
ansatz = EfficientSU2(num_qubits=4, reps=2, entanglement="linear", insert_barriers=True)
ansatz.decompose().draw("mpl", style="iqx")
print(ansatz.decompose().draw("mpl", style="iqx"))
shots = 1000
opt_iter = 1000
optimizer = SLSQP(maxiter=opt_iter)
initial_point = np.random.random(ansatz.num_parameters)
vqe_runtime = VQEClient(ansatz=ansatz,
                        optimizer=optimizer,
                        initial_point=initial_point,
                        provider=provider,
                        backend=backend,
                        shots=shots,
                        )


# --------------------------------------------------------------------
# ************************* Functions ********************************
# --------------------------------------------------------------------
#   Qiskit library - molecule builder:
def q_driver(molecule, charge, spin, basis, geo_format):
    q_Laboratory = PySCFDriver(molecule.format(geo_format),
                               unit=DistanceUnit.ANGSTROM,
                               charge=charge,
                               spin=spin,
                               basis=basis,
                               )

    return q_Laboratory.run()


#   PySCF library - molecule builder:
def c_driver(molecule, charge, spin, basis, geo_format):
    mol = gto.Mole()
    c_laboratory = mol.build(atom=molecule.format(geo_format),
                             charge=charge,
                             spin=spin,
                             basis=basis)
    return c_laboratory


#   Complexity Reduction ---> Active Space Transformer
def as_reduction(problem, ac_elec, ac_orbitals, address):
    transformer = ActiveSpaceTransformer(ac_elec, ac_orbitals, active_orbitals=address)
    return transformer.transform(problem)


#   Solver ---> Output = total energy
def solver(approach, problem, converter):
    gse_solver = GroundStateEigensolver(converter, approach)
    return gse_solver.solve(problem).total_energies[0]


#   Solver ---> Output = Raw data
def solver_raw(approach, problem, converter):
    gse_solver = GroundStateEigensolver(converter, approach)
    return gse_solver.solve(problem)


#   Unit Converter ---> Hartree to kJ/mol
def ha_to_kj(hartree):
    kj = hartree * 2625.5
    return kj



