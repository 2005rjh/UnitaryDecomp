import numpy as np
from UnitaryChain import *


##	two qubit gates
CntrlX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)
CntrlZ = np.diag([1.,1.,1.,-1.])


def check_grad_weight2_to_target(par):
	"""Check the function compute_grad_weight2_to_target()."""
	rand_seed, tol_factor = par
	##	Initialize random number generator
	if np.version.version >= '1.17.0':
		RNG = np.random.default_rng(rand_seed)
	else:
		RNG = np.random.RandomState(rand_seed)

	UC0 = two_qubits_unitary(CntrlZ);
	UC0.update_V_at_step(1, random_small_Unitary(4, RNG=RNG, sigma=0.5))
	print(UC0.str())
	UC0.check_consistency()

	W0 = UC0.weight2_to_target()
	gradW = UC0.compute_grad_weight2_to_target(UC0.U_to_target())
	H = Gaussian_Hermitian(4, RNG=RNG)
	HgradW = np.sum(gradW * H).real
	print("H . gradW =", HgradW)
	for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
		UC = UC0.copy()
		UC.apply_expiH_to_V_at_step(1, H * eps)
		W = UC.weight2_to_target()
		dW = W - W0
		second_order_comp = dW/eps-HgradW
		#print("w0 = {} , w = {} , dw = {}".format( W0, W, dW ))
		print("eps = {}  \t {} \t {}".format( eps, dW/eps, second_order_comp ))
		assert second_order_comp < eps * tol_factor
	print('\n')


def test_grad_weight2():
	yield check_grad_weight2_to_target, (65, 20.)
	yield check_grad_weight2_to_target, (40, 40.)
	yield check_grad_weight2_to_target, (90, 6.)


################################################################################
if __name__ == "__main__":
	print("==================== Test matrix identities ====================")
	np.set_printoptions(linewidth=2000, precision=4, threshold=10000, suppress=False)

	if 1:		# test derivative
		for t,c in test_grad_weight2(): t(c)

