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
#UC = two_qubits_unitary(CntrlX)
#UC = two_qubits_unitary(CntrlZ); UC.update_V_at_point(1, np.eye(4))
#UC.subdivide_at_step(0, 3)
#print(UC.str())


##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print( Gaussian_Hermitian(2, RNG=RNG) )




if 1:
	##	set up CZ
	print("== Load CtrlZ with 3-step construction.")
	UC = two_qubits_unitary(CntrlZ);
	UC.set_coef(Rabi1 = 0.2)
	UC.load_from_Ulist([ np.kron( [[1,-1],[1,1]], [[1,-1],[1,1]] )/2. ,
		np.array([ [1,0,0,1j], [0,1,1j,0], [0,1j,1,0], [1j,0,0,1] ])/np.sqrt(2) ,
		np.kron( [[1,1],[-1,1]], [[1,1],[-1,1]] ) / 2. , ])
	print(UC.str(verbose = 2))

	if 1:
		UC.apply_random_small_phase_to_Vfinal(RNG=RNG, sigma=0.01)
		#print(UC.str(verbose = 3))
		grad_desc_step_size = 0.01
		new_w = UC.weight_total()
		UC.subdivide_every_step(2)
		for itr in range(1200):
			gradH = UC.compute_grad_weight2(enforce_U2t_0weight=True)
			old_w = new_w
			for stp in range(1, UC.N+1):
				UC.apply_expiH_to_V_at_point(stp, -gradH[stp] * grad_desc_step_size)
				new_w = UC.weight_total()
			if np.mod(itr, 50) == 0: print("iter {}:  \t{} \t(w1 = {})\tchk err = {}".format( itr, new_w, UC.weight1_total(), UC.check_consistency()['err'] ))
			if np.mod(itr, 81) == 0: UC.check_consistency()
			if new_w > old_w: print("Uh oh...")
			if new_w + 1e-8 > old_w: break

		print(UC.str(verbose = 3))
		print(UC.str(verbose = 1))

if 0:
	## Try to update Vs[i] (steps i-1 and i)
	UCbk = UC.copy()
	for i in [1]:
		for itr in range(100):
			old_w = UCbk.weight_total()
			smallH = Gaussian_Hermitian(UC.d, RNG=RNG, sigma=0.05)
			UC.apply_expiH_to_V_at_point(i, smallH)
			new_w = UC.weight_total()
			if new_w > old_w:
				UC = UCbk.copy()		# reject
			else:
				UCbk = UC.copy()		# accept
	print(UC.str())

