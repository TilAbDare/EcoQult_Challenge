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
ansatz = EfficientSU2(num_qubits=4, reps=1, entanglement="linear", insert_barriers=True)
# ansatz.decompose().draw("mpl", style="iqx")
shots = 500
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


# --------------------------------------------------------------------
# ************************  Laboratory *******************************
# --------------------------------------------------------------------
#   Qiskit problems:
ion_q_prob = q_driver(ion, 2, 0, basis_set, None)
target_q_prob = q_driver(target, 0, 0, basis_set, None)
ligand_q_prob = q_driver(ligand, 2, 0, basis_set, None)

#   pyscf Problems:
ion_c_prob = c_driver(ion, 2, 0, basis_set, None)
target_c_prob = c_driver(target, 0, 0, basis_set, None)
ligand_c_prob = c_driver(ligand, 2, 0, basis_set, None)
# --------------------------------------------------------------------
# **********************  Complexity Reduction ***********************
# --------------------------------------------------------------------
#   Active Space Reduction
ion_q_prob = as_reduction(ion_q_prob, a_el_ion, a_or_ion, a_mos_ion)
target_q_prob = as_reduction(target_q_prob, a_el_target, a_or_target, a_mos_target)
ligand_q_prob = as_reduction(ligand_q_prob, a_el_ligand, a_or_ligand, a_mos_ligand)

# --------------------------------------------------------------------
# **************************  Qubti Encoding *************************
# --------------------------------------------------------------------
#   Qubit Encoding
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

# --------------------------------------------------------------------
# ***** Ground State Energy Calculation in the Optimal Geometry ******
# --------------------------------------------------------------------
#   HW implementation ---> VQE-Client:
# For dissociation energy:
ion_energy_vqe_hw = solver(vqe_runtime, ion_q_prob, converter)
target_energy_vqe_hw = solver(vqe_runtime, target_q_prob, converter)

# For history analysis:
start_single = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)

ligand_energy_vqe_hw = solver_raw(vqe_runtime, ligand_q_prob, converter)

stop_single = timeit.default_timer()
runtime_single = stop_single - start_single

#   History
runtime_result = ligand_energy_vqe_hw.raw_result
history = runtime_result.optimizer_history
loss_energy = history["energy"]
loss_param = history["params"]
loss_plus_gse = loss_energy + ligand_energy_vqe_hw.total_energies[0]
loss_plus_nr = loss_energy + ligand_energy_vqe_hw.nuclear_repulsion_energy

start_full = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)
# --------------------------------------------------------------------
# ******* Potential Energy Surface in Varying  the Geometry **********
# --------------------------------------------------------------------
#   Qiskit:
vqe_gse_hw = []

for i, d in enumerate(bonding_distances):
    #   Laboratory Creation by Qiskit:
    ligand_q_loop = q_driver(ligand_loop, 2, 0, basis_set, d)
    ligand_q_loop = as_reduction(ligand_q_loop, a_el_ligand, a_or_ligand, a_mos_ligand)
    #   VQE - IBMQ:
    ligand_energy_vqe_hw = solver(vqe_runtime, ligand_q_loop, converter)
    vqe_gse_hw += [ligand_energy_vqe_hw]

stop_full = timeit.default_timer()
runtime_full = stop_full - start_full
# --------------------------------------------------------------------
# ******* Bonding Dissociation Energy in Varying the Geometry ********
# --------------------------------------------------------------------
# Dissociation Energy ---> Hartree
ligand_diss_vqe_hw = []

for a in range(len(vqe_gse_hw)):
    ligand_diss_vqe_hw += [vqe_gse_hw[a] - (ion_energy_vqe_hw + target_energy_vqe_hw)]

# Dissociation Energy ---> kj/mol

ligand_diss_vqe_kj_hw = []

for a in range(len(ligand_diss_vqe_hw)):
    ligand_diss_vqe_kj_hw += [ha_to_kj(ligand_diss_vqe_hw[a])]

# --------------------------------------------------------------------
# ********************* Saving The Data locally **********************
# --------------------------------------------------------------------
df_gse_ha = pd.DataFrame(list(zip(bonding_distances,
                                  vqe_gse_hw)),
                         columns=['Distance(A)', 'VQE-IBMQ'])

df_diss_ha = pd.DataFrame(list(zip(bonding_distances,
                                   ligand_diss_vqe_hw)),
                          columns=['Distance(A)', 'VQE-IBMQ'])

df_diss_kj = pd.DataFrame(list(zip(bonding_distances,
                                   ligand_diss_vqe_kj_hw)),
                          columns=['Distance(A)', 'VQE-IBMQ'])

df_runtime_full = pd.DataFrame({'runtime_full_problem (min)': [runtime_full / 60]})
df_runtime_single = pd.DataFrame({'runtime_single_problem (min)': [runtime_single / 60]})

df_opt_history = pd.DataFrame(list(zip(loss_energy,
                                       loss_param,
                                       loss_plus_gse,
                                       loss_plus_nr)),
                              columns=['his_loss_energy', 'his_loss_parameters', 'loss+GSE', 'loss+NRP'])

#   Saving Locally;
with pd.ExcelWriter(name_excel) as writer:
    df_gse_ha.to_excel(writer, sheet_name='PES (ha)', index=False)
    df_diss_ha.to_excel(writer, sheet_name='Dissociation (ha)', index=False)
    df_diss_kj.to_excel(writer, sheet_name='Dissociation (kJ)', index=False)
    df_runtime_full.to_excel(writer, sheet_name='Runtime_full', index=False)
    df_runtime_single.to_excel(writer, sheet_name='Runtime_single', index=False)
    df_opt_history.to_excel(writer, sheet_name='Opt_History', index=False)

# --------------------------------------------------------------------
# ****************************** Plot ********************************
# --------------------------------------------------------------------
# Dissociation Energy in (Hartree)
plt.plot(bonding_distances, ligand_diss_vqe_hw, 'r*', label='VQE-IBMQ')
plt.grid(True, linestyle='-.', linewidth=0.1, which='major')
plt.title('Dissociation Energy')
plt.title(name_as, loc='right')
plt.title(name_plot, loc='left')
plt.xlabel('Distance ($\AA$)')
plt.ylabel('\u0394E (Ha)')
plt.legend()
plt.savefig('de(ha).png', format=format, dpi=dpi)
plt.show()

# Dissociation Energy in (kj/mole)
plt.plot(bonding_distances, ligand_diss_vqe_kj_hw, 'r*', label='VQE-IBMQ')
plt.grid(True, linestyle='-.', linewidth=0.1, which='major')
plt.title(name_as, loc='right')
plt.title(name_plot, loc='left')
plt.title('Dissociation Energy', )
plt.xlabel('Distance ($\AA$)')
plt.ylabel('\u0394E (kJ/mol)')
plt.legend()
plt.savefig('de(kj).png', format=format, dpi=dpi)
#plt.show()

# Potential Energy Surface in (Hartree)
plt.plot(bonding_distances, vqe_gse_hw, 'r*', label='VQE-IBMQ')
plt.grid(True, linestyle='-.', linewidth=0.1, which='major')
plt.title(name_as, loc='right')
plt.title('Potential Energy Surface')
plt.title(name_plot, loc='left')
plt.xlabel('Distance ($\AA$)')
plt.ylabel('Energy (Ha)')
plt.legend()
plt.savefig('pes(ha).png', format=format, dpi=dpi)
#plt.show()

# History
# plot loss and reference value
plt.plot(loss_plus_gse, label="Runtime VQE")
# plt.axhline(y=target_energy + 0.2, color="tab:red", ls=":", label="Target + 200mH")
# plt.axhline(y=target_energy, color="tab:red", ls="--", label="Target")
plt.legend(loc="best")
plt.xlabel("Iteration")
plt.ylabel("Energy [H]")
plt.title("VQE energy")
plt.savefig('History', format=format, dpi=dpi)
#plt.show()
