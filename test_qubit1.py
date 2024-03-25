import numpy as np
from UnitaryChain import *

def weightEval(gate):
	UC = qubit_unitary(gate)
	UC.subdivide_at_step(0, 3)		## split step 0 into 3 pieces
	UC.subdivide_at_step(1, 2)		## then, split step 1 into 2 pieces
	for i in range(UC.N):
		print("Step {}:  (weight = {})\n{}".format( i, UC.weight_at_step(i), zero_real_if_close(UC.logU(i)) ))
	print("Final U:\n", zero_real_if_close(UC.Ufinal()))
	print("U to target:  (weight = {})\n{}".format( UC.weight_to_target(), zero_real_if_close(UC.U_to_target()) ))
	print("Total weight:", UC.weight_total())

I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1.],[1.,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1.,0],[0,-1.]])
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)

np.set_printoptions(precision=4, linewidth=10000, suppress=True)
weightEval(Hadamard);
weightEval(PauliZ);

##	Target the Hadamard gate


##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print(Gaussian_Hermitian(2, RNG=RNG) )
print(random_small_Unitary(2, RNG=RNG, sigma=0.1) )
#	print( Gaussian_Hermitian(2, RNG=RNG) )

## Try to update Vs[i] (steps i-1 and i)
UC.backup_Vs()
for i in [1]:
	for itr in range(100):
		old_w = UC.weight_total()
		smallU = random_small_Unitary(2, RNG=RNG, sigma=0.1)
		UC.Vs[i] = smallU @ UC.Vs[i]		# make sures to mulitply from the left
		new_w = UC.weight_total()
		if new_w > old_w:
			print("{} -> {}  (reject)".format( old_w, new_w ))
			UC.restore_from_backup_Vs()
		else:
			print("{} -> {}  (accept)".format( old_w, new_w ))
			UC.backup_Vs()
