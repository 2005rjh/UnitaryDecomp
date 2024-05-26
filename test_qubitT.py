import numpy as np
from UnitaryChain import *

def optimize(UC, sub, sigma):
    print("subdivisions: {}\tsigma: {}".format(sub, sigma))
    UC.subdivide_at_step(0, sub) # subdividing steps beyond 0 reduces weight by bug
    print(UC.str())
    UC.backup_Vs()
    for itr in range(3000):
    	# print("------STEP {}------".format(i))
    	for i in range(1, UC.N+1):
    		old_w = UC.weight_total()
    		smallU = random_small_Unitary(2, RNG=RNG, sigma=sigma)
    		UC.Vs[i] = smallU @ UC.Vs[i]		# make sures to mulitply from the left
    		new_w = UC.weight_total()
    		if new_w > old_w:
    			# print("{} -> {}  (reject)".format( old_w, new_w ))
    			UC.restore_from_backup_Vs()
    		else:
    			# print("{} -> {}  (accept)".format( old_w, new_w ))
    			UC.backup_Vs()
    	#print("FINAL WEIGHT {}".format(UC.weight_at_step(i-1)))
    	#print("WEIGHT LIST: {}".format(UC.weight_list()))
    
    try:
        UC.check_consistency()
    except:
        print("FAILED CONSISTENCY CHECK")
    print(UC.str())
    return UC

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
mini = qubit_unitary(Hadamard)
minix = 0
for x in range(1, 5):
    UC = qubit_unitary(Hadamard)
    new = optimize(UC, x, 0.05)
    if mini.weight_total() > new.weight_total():
        mini = new
        minix = x
print("\n\nThe smallest configuration is:\n", mini.str(), "\nat {} subdivisions".format(minix))

mini = qubit_unitary(Hadamard)
miniy = 0
for y in range(1, 15):
    UC = qubit_unitary(Hadamard)
    new = optimize(UC, minix, y/1000)
    if mini.weight_total() > new.weight_total():
        mini = new
        miniy = y
print("\n\nThe smallest configuration is:\n", mini.str(), "\nat sigma={}".format(miniy/1000))
 
# UC.subdivide_at_step(0, 2)		## split step 0 into 3 pieces
# UC.subdivide_at_step(1, 4)		## then, split step 1 into 2 pieces