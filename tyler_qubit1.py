import numpy as np
from UnitaryChain import *
from stringtools import *

def rand_optimize(UC, sub, sigma):
    print("subdivisions: {}\tsigma: {}".format(sub, sigma))
    UC.subdivide_at_step(0, sub) # subdividing steps beyond 0 reduces weight by bug
    print(UC.Vs)
    print(UC.str())
    UCbk = UC.copy()
    for itr in range(3000):
    	# print("------STEP {}------".format(i))
    	for i in range(1, UC.N+1):
    		old_w = UCbk.weight_total()
    		smallU = random_small_Unitary(2, RNG=RNG, sigma=sigma)
    		UC.apply_U_to_V_at_step(i, smallU)		# make sures to mulitply from the left
    		new_w = UC.weight_total()
    		if new_w > old_w:
    			# print("{} -> {}  (reject)".format( old_w, new_w ))
    			UC = UCbk.copy()
    		else:
    			# print("{} -> {}  (accept)".format( old_w, new_w ))
    			UCbk = UC.copy()
    	#print("FINAL WEIGHT {}".format(UC.weight_at_step(i-1)))
    	#print("WEIGHT LIST: {}".format(UC.weight_list()))
    
    try:
        UC.check_consistency()
    except:
        print("FAILED CONSISTENCY CHECK")
    print(UC.str())
    print(UC.Vs)
    return UC

def grad_optimize(UC, sub, step, pen):
	UC.set_coef(penalty=pen)
	UC.subdivide_at_step(0, sub)
	print("UC coef: ", UC.coef)
	grad_desc_step_size = step
	new_w = UC.weight_total()
	print("start:   \t{}".format( new_w ))
	for itr in range(5000):
		gradH = UC.compute_grad_weight2()
		old_w = new_w
		for stp in range(1, UC.N+1):
			UC.apply_expiH_to_V_at_step(stp, -gradH[stp] * grad_desc_step_size)
			new_w = UC.weight_total()
		if np.mod(itr, 50) == 0: print("iter {}:  \t{}".format( itr, new_w ))
		if new_w > old_w: print("Uh oh...")
		if new_w + 1e-8 > old_w: break

	print("="*20, "done", "="*20)
	print("UC coef: ", UC.coef, "\n")
	print(UC.str())

I2 = np.eye(2, dtype=float)
PauliX = np.array([[0,1.],[1.,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1.,0],[0,-1.]])
Hadamard = np.array([[1,1],[1,-1]], dtype=float) / np.sqrt(2)

np.set_printoptions(precision=4, linewidth=10000, suppress=True)

##	Target the Hadamard gate

##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print( Gaussian_Hermitian(2, RNG=RNG) )

# Optimize by subdivisions
"""
mini = qubit_unitary(Hadamard)
minix = 0
for x in range(1, 5):
    UC = qubit_unitary(Hadamard)
    new = optimize(UC, x, 0.013)
    if mini.weight_total() > new.weight_total():
        mini = new
        minix = x
print("\n\nThe smallest configuration is:\n", mini.str(), "\nat {} subdivisions".format(minix))
"""

# Optimize by sigma
"""
mini = qubit_unitary(Hadamard)
miniy = 0
for y in range(1, 15):
    UC = qubit_unitary(Hadamard)
    new = optimize(UC, minix, y/1000)
    if mini.weight_total() > new.weight_total():
        mini = new
        miniy = y
print("\n\nThe smallest configuration is:\n", mini.str(), "\nat sigma={}".format(miniy/1000))
"""
"""
# Optimize subdivisions by distance
mini = optimize(qubit_unitary(Hadamard), 1, 0.013)
minix = 1
for x in range(2, 6):
    UC = qubit_unitary(Hadamard)
    new = optimize(UC, x, 0.013)
    if mini.weight_to_target() > new.weight_to_target():
        mini = new
        minix = x
print("\n\nThe best approximation is:\n", mini.str(), "\nat {} subdivisions".format(minix))
"""
rand_optimize(qubit_unitary(Hadamard), 3, 0.05)
grad_optimize(qubit_unitary(Hadamard), 3, 0.01, 5.0)
# UC.subdivide_at_step(0, 2)		## split step 0 into 3 pieces
# UC.subdivide_at_step(1, 4)		## then, split step 1 into 2 pieces