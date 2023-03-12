# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------

import timeit
import numpy as np
import pandas as pd
from numpy import reshape
from pyscf import gto, scf, mcscf, cc
from matplotlib import pyplot as plt
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms import VQEUCCFactory
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from pyscf import gto, dft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


# --------------------------------------------------------------------
# ***************************  Inputs ********************************
# --------------------------------------------------------------------
#   dataframe and ploting:
name_excel = 'ZnCO2_2hl.xlsx'
name_plot = 'CO2-Zn(2+)'
name_as = '1HOMO-1LUMO'
format = 'png'
dpi = 1200


#   Molecules and their geometries in z-matrices format:


ligand = 'C; ' \
        'O 1 1.19619; ' \
        'O 1 1.19638 2 179.97438; ' \
        'Zn 3 1.79541 1 120.07056 2 180.00000'
# Which bonding in you want to vary, make it blank!

ligand = 'C; ' \
        'O 1 1.19619; ' \
        'O 1 1.19638 2 179.97438; ' \
        'Zn 3 {} 1 120.07056 2 {}'

points = 20
dist = np.linspace(1.39541, 2.19541, points)
angle = np.linspace(130, 230, points)

#   Arguments:
mapper = BravyiKitaevMapper()
ansatz = UCCSD()
optimizer = SLSQP()
estimator = Estimator()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)
numpy_solver = NumPyMinimumEigensolver()
vqe_factory = VQEUCCFactory(estimator, ansatz, optimizer)


# --------------------------------------------------------------------
# ***************  Active Space Transformation Defining **************
# --------------------------------------------------------------------
as_transformer = ActiveSpaceTransformer(2, 2, active_orbitals=[24, 25])

# --------------------------------------------------------------------
# ************************  Hamiltonian *****************************
# --------------------------------------------------------------------
# ------------ Qubit Encoding ------------
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

# --------------------------------------------------------------------
# ***************** Energy Calculation Over bonding ******************
# --------------------------------------------------------------------

def func_vqe(dist, angle):
    uccsd_energy = []
    for d in range(len(dist)):
        for a in range(len(angle)):
            driver = PySCFDriver(ligand.format(dist[d], angle[a]),
                                 unit=DistanceUnit.ANGSTROM,
                                 charge=2,
                                 spin=0,
                                 basis='sto3g')
            problem = driver.run()
            problem = as_transformer.transform(problem)
            VQE = GroundStateEigensolver(converter, vqe_factory)
            uccsd_result = VQE.solve(problem).total_energies[0]
            uccsd_energy += [uccsd_result]
    return uccsd_energy


def func_ccsd(dist, angle):
    ccsd = []
    for d in range(len(dist)):
        for a in range(len(angle)):
            mol = gto.Mole()
            c_laboratory = mol.build(atom=ligand.format(dist[d], angle[a]),
                                     charge=2,
                                     spin=0,
                                     basis='sto3g')
            ligand_rhf = scf.RHF(c_laboratory).run()
            ligand_ccsd = cc.CCSD(ligand_rhf).run()
            ccsd += [ligand_ccsd.e_tot]

    return ccsd


def func_rhf(dist, angle):
    rhf = []
    for d in range(len(dist)):
        for a in range(len(angle)):
            mol = gto.Mole()
            c_laboratory = mol.build(atom=ligand.format(dist[d], angle[a]),
                                     charge=2,
                                     spin=0,
                                     basis='sto3g')
            ligand_rhf = scf.RHF(c_laboratory).run()
            rhf += [ligand_rhf.e_tot]

    return rhf

def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]


start_full = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)

energy_uccsd = func_vqe(dist, angle)

stop_full = timeit.default_timer()
runtime_full = stop_full - start_full

energy_ccsd = func_ccsd(dist, angle)
energy_rhf = func_rhf(dist, angle)


df_PES_3D = pd.DataFrame(list(zip(dist,
                                   angle,
                                   energy_uccsd,
                                   energy_ccsd,
                                   energy_rhf,
                                   )),
                          columns=['Distance(A)','Angle', 'VQE', 'CCSD', 'RHF'])

df_runtime = pd.DataFrame({'runtime_single_problem (min)': [runtime_full / 60]})

#   Saving Locally;
with pd.ExcelWriter(name_excel) as writer:
    df_PES_3D.to_excel(writer, sheet_name='PES 3D (ha)', index=False)
    df_runtime.to_excel(writer, sheet_name='Runtime_single', index=False)

X = np.array(dist)
Y = np.array(angle)
list = np.array(convert_1d_to_2d(energy_uccsd, points))
Z = list
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1)
plt.title('PES 3D - VQE UCCSD')
plt.title(name_plot, loc='left')
ax.set_xlabel('Distance ($\AA$)')
ax.set_ylabel('Angle (degree)')
ax.set_zlabel('Energy (Ha)')
plt.savefig('PES 3D VQE.png', format=format, dpi=dpi)
plt.show()

X = np.array(dist)
Y = np.array(angle)
list = np.array(convert_1d_to_2d(energy_ccsd, points))
Z = list
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1)
plt.title('PES 3D - CCSD')
plt.title(name_plot, loc='left')
ax.set_xlabel('Distance ($\AA$)')
ax.set_ylabel('Angle (degree)')
ax.set_zlabel('Energy (Ha)')
plt.savefig('PES 3D CCSD.png', format=format, dpi=dpi)
plt.show()


X = np.array(dist)
Y = np.array(angle)
Z = np.array(convert_1d_to_2d(energy_rhf, points))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1)
plt.title('PES 3D - CCSD')
plt.title(name_plot, loc='left')
ax.set_xlabel('Distance ($\AA$)')
ax.set_ylabel('Angle (degree)')
ax.set_zlabel('Energy (Ha)')
plt.savefig('PES 3D RHF.png', format=format, dpi=dpi)
plt.show()












