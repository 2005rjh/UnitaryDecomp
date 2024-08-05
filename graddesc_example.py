import numpy as np
from UnitaryChain import *


##	Set up options for displaying numpy arrays
np.set_printoptions(precision=4, linewidth=10000, suppress=True)


##	Initialize random number generator
##	The RNG object is used whenever any step involves randomization
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)


##	set up CZ gate as the target
CntrlZ = np.diag([1.,1.,1.,-1.])
UC = two_qubits_unitary(CntrlZ);


##	Either load in a pre-defined sequence of steps or random unitaries
if 0:
	print("== Load CtrlZ with 3-step construction.")
	UC.set_coef(Rabi1 = 0.2)
	UC.load_from_Ulist([ np.kron( [[1,-1],[1,1]], [[1,-1],[1,1]] )/2. ,
		np.array([ [1,0,0,1j], [0,1,1j,0], [0,1j,1,0], [1j,0,0,1] ])/np.sqrt(2) ,
		np.kron( [[1,1],[-1,1]], [[1,1],[-1,1]] )/2. , ])

else:
	print("== Start with random unitaries.")
	UC.subdivide_at_step(0, 3)	# subdivide into 3 steps
	for p in range(1, UC.N):		# randomize points 1 through N-1
		# Note, one can use random_small_Unitary(UC.d, RNG=RNG, sigma=0.1) instead of random_Unitary if you want to start with something close to the original UChain
		smallU = random_Unitary(UC.d, RNG=RNG)
		UC.apply_U_to_V_at_point(p, smallU)
	##	Randomize point N in a way that keeps weight2_to_target zero
	##	sigma controls the 'size' of the randomization.
	UC.apply_random_small_phases_to_Vfinal(RNG=RNG, sigma=10)

##	Display current state of UC
##	Higher verbosity means more crap
print(UC.str(verbose = 1))

##	Do gradient descent
if 1:
	grad_desc_step_size = 0.02
	print("== Perform gradient descent with step size @ {}".format(grad_desc_step_size))
	new_w = UC.weight_total()
	for itr in range(3000):
		##	Call the gradient function in a way that enforces weight2_to_target to be zero
		gradH = UC.compute_grad_weight2(enforce_U2t_0weight=True)
		old_w = new_w
		for stp in range(1, UC.N+1):
			UC.apply_expiH_to_V_at_point(stp, -gradH[stp] * grad_desc_step_size)
			new_w = UC.weight_total()
		if np.mod(itr, 200) == 0 or itr < 11: print("iter {}:  \t{} \t(w1 = {})\tchk err = {}".format( itr, new_w, UC.weight1_total(), UC.check_consistency()['err'] ))
		if new_w > old_w: print("Uh oh...")
		if new_w + 1e-8 > old_w: break
	print()

print("After gradient descent:")
print(UC.str(verbose = 1))
print("  consistency check err = {}\n  weight2 to target = {}\n".format( UC.check_consistency()['err'] , UC.weight2_to_target() ))

##	Unitarize
UC.unitarize_point('all')
##	Forces weight2_to_target to be zero again (to counteract accumulated floating point errors)
UC.force_weight2t_to_zero()
print("After unitarize:")
print("  consistency check err = {}\n  weight2 to target = {}\n".format( UC.check_consistency()['err'] , UC.weight2_to_target() ))

