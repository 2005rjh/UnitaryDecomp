import numpy as np
from UnitaryChain import *
from stringtools import *
from solutionary import *

I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1.],[1.,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1.,0],[0,-1.]])
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)
##	Some example of two-body gates
CntrlZ = np.diag([1.,1.,1.,-1.])
CntrlX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)

np.set_printoptions(precision=4, linewidth=10000, suppress=True)

UC = two_qubits_unitary(CntrlZ)

dictionary = solIndex()
dictionary.load("tyler_sols.obj")

for x in range(dictionary.length()):
	print("-"*5, "SOLUTION {}".format(x), "-"*5, "\n"+dictionary.access(x).str(verbose=3))

dictionary.save("tyler_sols.obj")
"""
UC = two_qubits_unitary(CntrlZ)
# print(UC.str(), UC.Vs)
UC2 = [np.kron(Hadamard, Hadamard), np.array([[1/2**(1/2), 0, 0, 1j/2**(1/2)], [0, 1/2**(1/2), 1j/2**(2), 0], [0, 1j/2**(1/2), 1/2**(2), 0], [1j/2**(1/2), 0, 0, 1/2**(1/2)]], dtype=complex), np.kron(Hadamard, Hadamard)]

UC2b = [np.array([[1/2, -1/2, -1/2, 1/2], [1/2, 1/2, -1/2, -1/2], [1/2, -1/2, 1/2, -1/2], [1/2, 1/2, 1/2, 1/2]], dtype=complex), np.array([[1/2**(1/2), 0, 0, 1j/2**(1/2)], [0, 1/2**(1/2), 1j/2**(1/2), 0], [0, 1j/2**(1/2), 1/2**(1/2), 0], [1j/2**(1/2), 0, 0, 1/2**(1/2)]], dtype=complex), np.array([[1/2, 1/2, 1/2, 1/2], [-1/2, 1/2, -1/2, 1/2], [-1/2, -1/2, 1/2, 1/2], [1/2, -1/2, -1/2, 1/2]], dtype=complex)]
UC.load_from_Ulist(UC2b)
print(UC.str(verbose=3))
dictionary.add(UC2b)
dictionary.save("tyler_sols.obj")
"""