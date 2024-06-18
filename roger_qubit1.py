import numpy as np
from UnitaryChain import *


I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1.],[1.,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1.,0],[0,-1.]])
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)

np.set_printoptions(precision=4, linewidth=10000, suppress=True)

##	Target the Hadamard gate
UC = qubit_unitary(Hadamard)
#UC.subdivide_at_step(0, 3)		## split step 0 into 3 pieces
# print(UC.str())

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
	for itr in range(3000):
		old_w = UC.weight_total()
		smallU = random_small_Unitary(2, RNG=RNG, sigma=0.05)
		UC.Vs[i] = smallU @ UC.Vs[i]		# make sures to mulitply from the left
		new_w = UC.weight_total()
		if new_w > old_w:
#			print("{} -> {}  (reject)".format( old_w, new_w ))
			UC.restore_from_backup_Vs()
		else:
#			print("{} -> {}  (accept)".format( old_w, new_w ))
			UC.backup_Vs()
print(UC.str())
"""
UC = qubit_unitary(Hadamard)
UC.subdivide_at_step(0, 2)
print(UC.str())

if 0:		# old code, deprecated and won't really work anymore
	UC.backup_Vs()
	for itr in range(3000):
		for i in range(1, 3):
			old_w = UC.weight_total()
			smallU = random_small_Unitary(2, RNG=RNG, sigma=0.05)
			UC.Vs[i] = smallU @ UC.Vs[i]		# make sures to mulitply from the left
			new_w = UC.weight_total()
			if new_w > old_w:
	#			print("{} -> {}  (reject)".format( old_w, new_w ))
				UC.restore_from_backup_Vs()
			else:
	#			print("{} -> {}  (accept)".format( old_w, new_w ))
				UC.backup_Vs()
if 1:		# new working code
	UCbk = UC.copy()
	for itr in range(3000):
		for i in range(1, 3):
			old_w = UCbk.weight_total()
			smallU = random_small_Unitary(2, RNG=RNG, sigma=0.05)
			UC.apply_U_to_V_at_step(i, smallU)
			new_w = UC.weight_total()
			if new_w > old_w:
		#		print("{} -> {}  (reject)".format( old_w, new_w ))
				UC = UCbk.copy()
			else:
		#		print("{} -> {}  (accept)".format( old_w, new_w ))
				UCbk = UC.copy()



print("done")
print(UC.str())