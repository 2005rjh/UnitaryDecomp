import numpy as np
from UnitaryChain import *


I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1],[1,0]], dtype=float)
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1,0],[0,-1]], dtype=float)
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)

UC = UnitaryChain(Hadamard)
for i in range(UC.N):
	print("Step", i, ":\n", UC.U(i))


