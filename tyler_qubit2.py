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
B = np.sqrt(2-np.sqrt(2))/2 * np.array([[1+np.sqrt(2), 0, 0, 1j], [0, 1, 1j*(1+np.sqrt(2)), 0], [0, 1j*(1+np.sqrt(2)), 1, 0], [1j, 0, 0, 1+np.sqrt(2)]])

np.set_printoptions(precision=4, linewidth=10000, suppress=True)

##	Target
#UC = two_qubits_unitary(np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]))		# iSWAP (conversion)
#UC = two_qubits_unitary(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]))		# conversion Y
#UC = two_qubits_unitary(np.array([[0,0,0,1j],[0,1,0,0],[0,0,1,0],[1j,0,0,0]]))		# gain
#UC = two_qubits_unitary(np.kron(PauliX,I2)*1j)		# Rabi 1
#UC = two_qubits_unitary(np.kron(I2,PauliY)*1j)		# Rabi 2
#UC = two_qubits_unitary(np.kron(PauliX,PauliX)*1j)		# conv + gain
##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print( Gaussian_Hermitian(2, RNG=RNG) )

## Try to update Vs[i] (steps i-1 and i)

def rand_optimize(x, UC):
	UCbk = UC.copy()
	for itr in range(x):
	    for i in range(1, UC.N):
	        old_w = UCbk.weight_total()
	        smallU = random_small_Unitary(4, RNG=RNG, sigma=5.0)
	        UC.check_consistency()
	        UC.apply_U_to_V_at_point(i, smallU)
	        UC.check_consistency()
	        new_w = UC.weight_total()
	        # if new_w > old_w:
	        #		print("{} -> {}  (reject)".format( old_w, new_w ))
	        #    UC = UCbk.copy()
	        # else:
	        #		print("{} -> {}  (accept)".format( old_w, new_w ))
	        #    UCbk = UC.copy()
	print(UC.str())
	return UC


def grad_optimize(UC):
	grad_desc_step_size = 0.005
	new_w = UC.weight_total()
	print("start:   \t{}".format( new_w ))
	for itr in range(5000):
		gradH = UC.compute_grad_weight2()
		old_w = new_w
		for stp in range(1, UC.N+1):
			UC.apply_expiH_to_V_at_point(stp, -gradH[stp] * grad_desc_step_size)
			new_w = UC.weight_total()
		if np.mod(itr, 50) == 0: print("iter {}:  \t{}".format( itr, new_w ))
		if new_w > old_w:
			print("Uh oh...")
		if new_w + 1e-8 > old_w: break
	
	print("="*20, "done", "="*20)
	print("UC coef: ", UC.coef, "\n")
	print(UC.str())
	return UC, grad_desc_step_size

def print_sol():
	print("UC coef: ", UC.coef)
	print("Step size: {} --- Randomizations: {} --- Subdivisions: {} parts of {}".format(grad_desc_step_size, rands, prim_sub, sub_sub))
	print("Weight 1: ", UC.weight1_total(), "\tWeight 2: ", UC.weight_total())
	print(UC.str(verbose=3))

def subdivide_optimize(prim_sub, sub_sub, UC):
	for x in range(prim_sub):
		UC.subdivide_at_step(x, sub_sub)
		print("-"*5, "subdivided step {} by {}".format(x, sub_sub), "-"*5)
		UC, grad_desc_step_size = grad_optimize(UC)
	return UC, grad_desc_step_size

UC = two_qubits_unitary(CntrlZ)
dictionary = solutionary()
dictionary.load("tyler_sols.obj")
UC = dictionary.access(3).copy()
UC.set_coef(penalty=6.0)
print(UC.coef)
print(UC.str(verbose=3))

prim_sub = 3
sub_sub = 2
rands = 1

# UC.subdivide_at_step(0, prim_sub)
"""
for x in range(2):
	UC = rand_optimize(1, UC)
	UC, grad_desc_step_size = grad_optimize(UC)
"""

UC, grad_desc_step_size = subdivide_optimize(prim_sub, sub_sub, UC)
UC, grad_desc_step_size = grad_optimize(UC)

print_sol()

query = input("\n\nSave solution? (y/n): ")
if query == "y":
	dictionary.add(UC)
	print("Saved as solution {}".format(dictionary.length()-1))
	dictionary.save("tyler_sols.obj")
