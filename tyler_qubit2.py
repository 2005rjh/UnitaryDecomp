import numpy as np
from UnitaryChain import *


I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1.],[1.,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1.,0],[0,-1.]])
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)
##	Some example of two-body gates
CntrlZ = np.diag([1.,1.,1.,-1.])
CntrlX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)

np.set_printoptions(precision=4, linewidth=10000, suppress=True)

##	Target
#UC = two_qubits_unitary(np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]))		# iSWAP (conversion)
#UC = two_qubits_unitary(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]))		# conversion Y
#UC = two_qubits_unitary(np.array([[0,0,0,1j],[0,1,0,0],[0,0,1,0],[1j,0,0,0]]))		# gain
#UC = two_qubits_unitary(np.kron(PauliX,I2)*1j)		# Rabi 1
#UC = two_qubits_unitary(np.kron(I2,PauliY)*1j)		# Rabi 2
#UC = two_qubits_unitary(np.kron(PauliX,PauliX)*1j)		# conv + gain
UC = two_qubits_unitary(CntrlZ)
UC.subdivide_at_step(0, 3)
print(UC.Vs)
print(UC.str())
UC.Vs = [np.identity(4), np.kron(Hadamard, Hadamard), np.array([[1/2**(1/2), 0, 0, 1j/2**(1/2)], [0, 1/2**(1/2), 1j/2**(2), 0], [0, 1j/2**(1/2), 1/2**(2), 0], [1j/2**(1/2), 0, 0, 1/2**(1/2)]], dtype=complex), np.kron(Hadamard, Hadamard), CntrlZ]
print(UC.weight_of_U(UC.Vs[1]))
print(UC.Vs)
print(UC.str())

##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print( Gaussian_Hermitian(2, RNG=RNG) )

## Try to update Vs[i] (steps i-1 and i)
"""
UC.backup_Vs()
for i in [1]:
	for itr in range(30):
		old_w = UC.weight_total()
		smallU = random_small_Unitary(UC.d, RNG=RNG, sigma=0.05)
		UC.Vs[i] = smallU @ UC.Vs[i]		# make sures to mulitply from the left
		new_w = UC.weight_total()
		if new_w > old_w:
#			print("{} -> {}  (reject)".format( old_w, new_w ))
			UC.restore_from_backup_Vs()
		else:
#			print("{} -> {}  (accept)".format( old_w, new_w ))
			UC.backup_Vs()
print(UC.Vs)
# print(UC.str())
"""