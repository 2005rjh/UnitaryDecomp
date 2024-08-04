import numpy as np
import math
from UnitaryChain import *


I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1.],[1.,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1.,0],[0,-1.]])
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)

np.set_printoptions(precision=4, linewidth=10000, suppress=True)

##	Target the Hadamard gate
UC = qubit_UChain(Hadamard)
##UC.subdivide_at_step(0, 3)		## split step 0 into 3 pieces
##UC.subdivide_at_step(1, 2)		## then, split step 1 into 2 pieces
for i in range(UC.N):
	print("Step {}:  (weight = {})\n{}".format( i, UC.weight_at_step(i), zero_real_if_close(UC.logU(i)) ))
print("Final U:\n", zero_real_if_close(UC.Ufinal()))
print("U to target:  (weight = {})\n{}".format( UC.weight_to_target(), zero_real_if_close(UC.U_to_target()) ))
print("Total weight:", UC.weight_total())

##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print( Gaussian_Hermitian(2, RNG=RNG) )

## Try to update Vs[i] (steps i-1 and i)

'''for i in [1,2,3,4]:
	move_on = False
	while (not move_on):
		current_step_weight = UC.weight_at_step(i)
		prev_step_weight = UC.weight_at_step(i-1) ## Will have to check if step is 0 or something idk
		weight_delta = current_step_weight - prev_step_weight ## Check to see if the delta is positive or negative (generally positive == bad)

		if weight_delta > 0: ## positive
			"""
			Plan for this conditional:
				1. Check the weight -> target of previous step
				2. If increased, redo the current step with a unitary gate with a smaller sigma
					a. Redo that step with UC.Vs[i] = new_unitary @ UC.Vs[i-1]
				3. Repeat while loop
			"""
			prev_unitary = UC.U(i)
			prev_unitary.weight_to_target()
		if weight_delta < 0: ## negative
			"""
			Plan for this conditional:
				1. I really have no plan, if it's negative for now I guess just let it go on
			
			"""
			move_on = True'''

print("Attempt to lower weight.")
print("")
"""
	Plans post Meeting:

	Start to make the weight more managable/smaller now without modifying sigma for now

	Next, start to either make a whole new algorithm to find a low weight (might do this)

"""
UC.backup_Vs()
for itr in range(300):
	old_weight = UC.weight_total()
	sigma = 0.2
	for i in range(UC.N, 0, -1):
		cur_w = UC.weight_total()
		test_matrix = random_small_Unitary(2, RNG=RNG, sigma=sigma) @ UC.Vs[i]

		UC.Vs[i] = test_matrix

	## Test to see if weight got any lower
	new_weight = UC.weight_total()
	UC.check_consistency()
	if new_weight > old_weight:
		UC.restore_from_backup_Vs()
	else:
		UC.backup_Vs()
print(UC.str())

"""
Attempt to preform a subdivision on the optimized solution
"""
UC.subdivide_at_step(0, 2)
print(UC.str())

UC.backup_Vs()
for itr in range(300):
	old_weight = UC.weight_total()
	sigma = 0.0008
	for i in range(UC.N, 0, -1):
		cur_w = UC.weight_total()
		test_matrix = random_small_Unitary(2, RNG=RNG, sigma=sigma) @ UC.Vs[i]

		UC.Vs[i] = test_matrix

	## Test to see if weight got any lower
	new_weight = UC.weight_total()
	UC.check_consistency()
	if new_weight > old_weight:
		UC.restore_from_backup_Vs()
	else:
		UC.backup_Vs()
print(UC.str())
