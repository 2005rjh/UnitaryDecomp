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
#UC = two_qubits_unitary(CntrlZ); UC.update_V_at_step(1, np.eye(4))
#UC.subdivide_at_step(0, 3)
#print(UC.str())

if 0:
	print(UC.U_decomp(0))
	print(UC.U_decomp(-1))

##	Initialize random number generator
if np.version.version >= '1.17.0':
	RNG = np.random.default_rng(55)
else:
	RNG = np.random.RandomState(55)
#for i in range(10):
#	print( Gaussian_Hermitian(2, RNG=RNG) )


def check_grad_weight2_at_step(par):
	rand_seed, tol_factor = par
	##	Initialize random number generator
	if np.version.version >= '1.17.0':
		RNG = np.random.default_rng(rand_seed)
	else:
		RNG = np.random.RandomState(rand_seed)

	UC0 = two_qubits_unitary(np.eye(4));
	UC0.set_coef(penalty=5.)
	UC0.subdivide_at_step(0, 3)
	UC0.update_V_at_step(1, random_small_Unitary(4, RNG=RNG, sigma=0.5))
	UC0.update_V_at_step(2, random_small_Unitary(4, RNG=RNG, sigma=0.5))
	UC0.update_V_at_step(3, random_small_Unitary(4, RNG=RNG, sigma=0.5))
	print(UC0.str())
	print("U_decomp(0)", UC0.U_decomp(0))
	UC0.check_consistency()

	w0ref = UC0.weight2_at_step(0)
	w1ref = UC0.weight2_at_step(1)
	gradHL_s0, gradHR_s0 = UC0.compute_grad_weight2_at_step(0)
	gradHL_s1, gradHR_s1 = UC0.compute_grad_weight2_at_step(1)
	print(gradHL_s0, "= gradHL")
	print(gradHR_s1, "= gradHR")
	H = Gaussian_Hermitian(4, RNG=RNG)
	#print(H, "= H")
	HgradHL = np.sum( gradHL_s0 * H ).real
	HgradHR = np.sum( gradHR_s1 * H ).real
	print("HgradHL", HgradHL, "\tHgradHR", HgradHR)
	for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
		UC = UC0.copy()
		UC.apply_expiH_to_V_at_step(1, H * eps)
		#UC.check_consistency()
		#print(UC.U(0) @ UC0.U(0).conj().T)
		w0 = UC.weight2_at_step(0)
		w1 = UC.weight2_at_step(1)
		dw0 = w0 - w0ref
		dw1 = w1 - w1ref
		print("eps = {} \t w0ref = {}, w0 = {}, dw0 = {} \t dw1 = {}".format( eps, w0ref, w0, dw0, dw1 ))


if 0:		# test derivative
#	check_grad_weight2_to_target((65, 20.))
#	check_grad_weight2_to_target((40, 40.))
#	check_grad_weight2_to_target((90, 6.))
	check_grad_weight2_at_step((85, 20.))


if 0:
	## Try to update Vs[i] (steps i-1 and i)
	UCbk = UC.copy()
	for i in [1]:
		for itr in range(100):
			old_w = UCbk.weight_total()
			smallH = Gaussian_Hermitian(UC.d, RNG=RNG, sigma=0.05)
			UC.apply_expiH_to_V_at_step(i, smallH)
			new_w = UC.weight_total()
			if new_w > old_w:
				UC = UCbk.copy()		# reject
			else:
				UCbk = UC.copy()		# accept
	print(UC.str())

