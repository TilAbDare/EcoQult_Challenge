# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------
import timeit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyscf import gto
from pyscf import scf, cc
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit

# --------------------------------------------------------------------
# ***************************  Inputs ********************************
# --------------------------------------------------------------------
#   dataframe and ploting:
name_excel = 'MgCO2_4hl.xlsx'
name_plot = 'CO2-Mg(2+)'
name_as = '2HOMO-2LUMO'
format = 'png'
dpi = 1200

#   Active Space Transformation:
# Number of active electrons ---> An integer
a_el_ion = 4
a_el_target = 4
a_el_ligand = 4

# Number of active orbitals ---> An integer
a_or_ion = 4
a_or_target = 4
a_or_ligand = 4

#   List of all active orbitals ---> A list of integers
a_mos_ion = [3, 4, 5, 6]
a_mos_target = [9, 10, 11, 12]
a_mos_ligand = [14, 15, 16, 17]

#   Molecules and their geometries in z-matrices format:
ion = 'Mg'
target = 'C;' \
         'O 1 1.19700;' \
         'O 1 1.19700 2 179.97438'
ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 1.97923 1 180.000 2 180.00000'
#   The point to be varied, make it blank ---> {}
ligand_loop = 'C; ' \
              'O 1 1.19630; ' \
              'O 1 1.19628 2 179.97438; ' \
              'Mg 3 {} 1 180.0000 2 180.00000'


#   Essential arguments:
basis_set = 'sto3g'
bonding_distances = np.arange(0.5, 6, 0.15)
mapper = BravyiKitaevMapper()
ansatz = UCCSD()
initial_state = HartreeFock()
optimizer = SLSQP()
estimator = Estimator()
numpy_solver = NumPyMinimumEigensolver()
vqe_factory = VQEUCCFactory(estimator, ansatz, optimizer, initial_point=None, initial_state=None)


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


#   Solver ---> GroundStateEigensolver:
def solver(approach, problem, converter):
    gse_solver = GroundStateEigensolver(converter, approach)
    return gse_solver.solve(problem).total_energies[0]


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
#   Qiskit ---> VQE-UCCSD:
ion_energy_vqe = solver(vqe_factory, ion_q_prob, converter)
target_energy_vqe = solver(vqe_factory, target_q_prob, converter)

start_single_vqe = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)

ligand_energy_vqe = solver(vqe_factory, ligand_q_prob, converter)

stop_single_vqe = timeit.default_timer()
runtime_single_vqe = stop_single_vqe - start_single_vqe


#   Classical Solver ---> NumpyMinimumEigensolver:
ion_energy_numpy = solver(numpy_solver, ion_q_prob, converter)
target_energy_numpy = solver(numpy_solver, target_q_prob, converter)

start_single_numpy = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)

ligand_energy_numpy = solver(numpy_solver, ligand_q_prob, converter)

stop_single_numpy = timeit.default_timer()
runtime_single_numpy = stop_single_numpy - start_single_numpy
#   Classical Method by PySCF library ---> RHF
ion_rhf = scf.RHF(ion_c_prob).run()
ion_energy_rhf = ion_rhf.e_tot

target_rhf = scf.RHF(target_c_prob).run()
target_energy_rhf = target_rhf.e_tot

start_single_rhf = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)

ligand_rhf = scf.RHF(ligand_c_prob).run()
ligand_energy_rhf = ligand_rhf.e_tot

stop_single_rhf = timeit.default_timer()
runtime_single_rhf = stop_single_rhf - start_single_rhf
#   Classical Method by pyscf library ---> CCSD
ion_ccsd = cc.CCSD(ion_rhf).run()
ion_energy_ccsd = ion_ccsd.e_tot

target_ccsd = cc.CCSD(target_rhf).run()
target_energy_ccsd = target_ccsd.e_tot

start_single_ccsd = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)

ligand_ccsd = cc.CCSD(ligand_rhf).run()
ligand_energy_ccsd = ligand_ccsd.e_tot

stop_single_ccsd = timeit.default_timer()
runtime_single_ccsd = stop_single_ccsd - start_single_ccsd
#   Classical Method by pyscf library ---> CCSD(T)
ion_energy_ccsdt = (ion_ccsd.ccsd_t() + ion_energy_ccsd)
target_energy_ccsdt = (target_ccsd.ccsd_t() + ion_energy_ccsd)
ligand_energy_ccsdt = (ligand_ccsd.ccsd_t() + ion_energy_ccsd)

# --------------------------------------------------------------------
# ******* Potential Energy Surface in Varying  the Geometry **********
# --------------------------------------------------------------------

#   Qiskit:
vqe_gse = []
numpy_gse = []
rhf_gse = []
ccsd_gse = []
ccsdt_gse = []

start_full = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)
for i, d in enumerate(bonding_distances):
    #   Laboratory Creation by Qiskit:
    ligand_q_loop = q_driver(ligand_loop, 2, 0, basis_set, d)
    ligand_q_loop = as_reduction(ligand_q_loop, a_el_ligand, a_or_ligand, a_mos_ligand)

    #   Quantum Algorithm Approach:
    # VQE
    ligand_energy_vqe = solver(vqe_factory, ligand_q_loop, converter)
    vqe_gse += [ligand_energy_vqe]

stop_full = timeit.default_timer()
runtime_full = stop_full - start_full

for i, d in enumerate(bonding_distances):
    #   Laboratory Creation by Qiskit:
    ligand_q_loop = q_driver(ligand_loop, 2, 0, basis_set, d)
    ligand_q_loop = as_reduction(ligand_q_loop, a_el_ligand, a_or_ligand, a_mos_ligand)

    #   Classical Approach:
    ligand_energy_numpy = solver(numpy_solver, ligand_q_loop, converter)
    numpy_gse += [ligand_energy_numpy]

    #   Laboratory Creation by pyscf Library:
    ligand_c_loop = c_driver(ligand_loop, 2, 0, basis_set, d)

    #   Classical Method ---> RHF
    ligand_rhf = scf.RHF(ligand_c_loop).run()
    rhf_gse += [ligand_rhf.e_tot]

    #   Classical Method ---> CCSD
    ligand_ccsd = cc.CCSD(ligand_rhf).run()
    ccsd_gse += [ligand_ccsd.e_tot]

    #   Classical Method ---> CCSD(T)
    ccsdt_gse += [ligand_ccsd.ccsd_t() + ligand_ccsd.e_tot]

# --------------------------------------------------------------------
# ******* Bonding Dissociation Energy in Varying the Geometry ********
# --------------------------------------------------------------------
# Unit ---> Hartree
ligand_diss_vqe = []
ligand_diss_numpy = []
ligand_diss_rhf = []
ligand_diss_ccsd = []
ligand_diss_ccsdt = []

for a, c, d, e, f in zip(range(len(vqe_gse)),
                         range(len(numpy_gse)),
                         range(len(rhf_gse)),
                         range(len(ccsd_gse)),
                         range(len(ccsdt_gse))):
    ligand_diss_vqe += [vqe_gse[a] - (ion_energy_vqe + target_energy_vqe)]
    ligand_diss_numpy += [numpy_gse[c] - (ion_energy_numpy + target_energy_numpy)]
    ligand_diss_rhf += [rhf_gse[d] - (ion_energy_rhf + target_energy_rhf)]
    ligand_diss_ccsd += [ccsd_gse[e] - (ion_energy_ccsd + target_energy_ccsd)]
    ligand_diss_ccsdt += [ccsdt_gse[f] - (ion_energy_ccsdt + target_energy_ccsdt)]

# Unit ---> kJ/mol

ligand_diss_vqe_kj = []
ligand_diss_adapt_vqe_kj = []
ligand_diss_numpy_kj = []
ligand_diss_rhf_kj = []
ligand_diss_ccsd_kj = []
ligand_diss_ccsdt_kj = []

for a, c, d, e, f in zip(range(len(ligand_diss_vqe)),
                         range(len(ligand_diss_numpy)),
                         range(len(ligand_diss_rhf)),
                         range(len(ligand_diss_ccsd)),
                         range(len(ligand_diss_ccsdt))):
    ligand_diss_vqe_kj += [ha_to_kj(ligand_diss_vqe[a])]
    ligand_diss_numpy_kj += [ha_to_kj(ligand_diss_numpy[c])]
    ligand_diss_rhf_kj += [ha_to_kj(ligand_diss_rhf[d])]
    ligand_diss_ccsd_kj += [ha_to_kj(ligand_diss_ccsd[e])]
    ligand_diss_ccsdt_kj += [ha_to_kj(ligand_diss_ccsdt[f])]

# --------------------------------------------------------------------
# ********************* Saving The Data locally **********************
# --------------------------------------------------------------------
df_gse_ha = pd.DataFrame(list(zip(bonding_distances,
                                  vqe_gse,
                                  numpy_gse,
                                  rhf_gse,
                                  ccsd_gse,
                                  ccsdt_gse)),
                         columns=['Distance(A)', 'VQE-UCCSD', 'Numpy', 'RHF', 'CCSD', 'CCSD(T)'])

df_diss_ha = pd.DataFrame(list(zip(bonding_distances,
                                   ligand_diss_vqe,
                                   ligand_diss_numpy,
                                   ligand_diss_rhf,
                                   ligand_diss_ccsd,
                                   ligand_diss_ccsdt)),
                          columns=['Distance(A)', 'VQE-UCCSD', 'Numpy', 'RHF', 'CCSD', 'CCSD(T)'])

df_diss_kj = pd.DataFrame(list(zip(bonding_distances,
                                   ligand_diss_vqe_kj,
                                   ligand_diss_numpy_kj,
                                   ligand_diss_rhf_kj,
                                   ligand_diss_ccsd_kj,
                                   ligand_diss_ccsdt_kj)),
                          columns=['Distance(A)', 'VQE-UCCSD', 'Numpy', 'RHF', 'CCSD', 'CCSD(T)'])


df_runtime_single_vqe = pd.DataFrame({'runtime_single_vqe (min)': [runtime_single_vqe / 60]})
df_runtime_single_numpy = pd.DataFrame({'runtime_single_numpy (min)': [runtime_single_numpy / 60]})
df_runtime_single_rhf = pd.DataFrame({'runtime_single_rhf (min)': [runtime_single_rhf / 60]})
df_runtime_single_ccsd = pd.DataFrame({'runtime_single_ccsd (min)': [runtime_single_ccsd / 60]})
df_runtime_full_vqe = pd.DataFrame({'runtime_full_vqe (min)': [runtime_full / 60]})

#   Saving Locally;
with pd.ExcelWriter(name_excel) as writer:
    df_gse_ha.to_excel(writer, sheet_name='PES (ha)', index=False)
    df_diss_ha.to_excel(writer, sheet_name='Dissociation (ha)', index=False)
    df_diss_kj.to_excel(writer, sheet_name='Dissociation (kJ)', index=False)
    df_runtime_single_vqe.to_excel(writer, sheet_name='runtime_single_vqe', index=False)
    df_runtime_single_numpy.to_excel(writer, sheet_name='runtime_single_numpy', index=False)
    df_runtime_single_rhf.to_excel(writer, sheet_name='runtime_single_rhf', index=False)
    df_runtime_single_ccsd.to_excel(writer, sheet_name='runtime_single_ccsd', index=False)
    df_runtime_full_vqe.to_excel(writer, sheet_name='runtime_full_vqe', index=False)

# --------------------------------------------------------------------
# ****************************** Plot ********************************
# --------------------------------------------------------------------
# Dissociation Energy in (Hartree)
plt.plot(bonding_distances, ligand_diss_vqe, 'bx', label='VQE-UCCSD')
#plt.plot(bonding_distances, ligand_diss_numpy, 'g--', label='Numpy Eigensolver')
plt.plot(bonding_distances, ligand_diss_rhf, 'p-', label='RHF')
plt.plot(bonding_distances, ligand_diss_ccsd, 'r.', label='CCSD')
plt.plot(bonding_distances, ligand_diss_ccsdt, 'm^', label='CCSD(T)')
plt.grid(True, linestyle='-.', linewidth=0.1, which='major')
plt.title('Dissociation Energy')
plt.title(name_as, loc='right')
plt.title(name_plot, loc='left')
plt.xlabel('Distance ($\AA$)')
plt.ylabel('\u0394E (Ha)')
plt.legend()
plt.savefig('de(ha).png', format=format, dpi=dpi)
#plt.show()

# Dissociation Energy in (kj/mole)
plt.plot(bonding_distances, ligand_diss_vqe_kj, 'bx', label='VQE-UCCSD')
#plt.plot(bonding_distances, ligand_diss_numpy_kj, 'g--', label='Eigensolver_exact')
plt.plot(bonding_distances, ligand_diss_rhf_kj, 'p-', label='RHF')
plt.plot(bonding_distances, ligand_diss_ccsd_kj, 'r.', label='CCSD')
plt.plot(bonding_distances, ligand_diss_ccsdt_kj, 'm^', label='CCSD(T)')
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
plt.plot(bonding_distances, vqe_gse, 'bx', label='VQE-UCCSD')
#plt.plot(bonding_distances, numpy_gse, 'g--', label='Eigensolver_exact')
plt.plot(bonding_distances, rhf_gse, 'p-', label='RHF')
plt.plot(bonding_distances, ccsd_gse, 'r.', label='CCSD')
plt.plot(bonding_distances, ccsdt_gse, 'm^', label='CCSD(T)')
plt.grid(True, linestyle='-.', linewidth=0.1, which='major')
plt.title(name_as, loc='right')
plt.title('Potential Energy Surface')
plt.title(name_plot, loc='left')
plt.xlabel('Distance ($\AA$)')
plt.ylabel('Energy (Ha)')
plt.legend()
plt.savefig('pes(ha).png', format=format, dpi=dpi)
#plt.show()
