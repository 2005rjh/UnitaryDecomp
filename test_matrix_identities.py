import numpy as np
from UnitaryChain import *


Mx3list = [ np.arange(9).reshape(3,3), np.array([[2,0,-1],[1,1,-2],[0,-1,3]]), np.arange(-5,4).reshape(3,3).transpose(), np.array([[0,0,-1],[1,1.5,0],[0,0.5,0.5]]) ]

def check_adjoint_op(par):
	M1i, M2i, tol = par 
	print("check_adjoint_op({}, {})".format(M1i,M2i))
	M1 = Mx3list[M1i]
	M2 = Mx3list[M2i]
	v2 = M2.reshape(-1)
	ad1 = adjoint_op_MxRep(M1)
	assert ad1.shape == (9,9)
	commute = M1 @ M2 - M2 @ M1
	diff = ad1 @ v2 - commute.reshape(-1)
	assert np.max(np.abs(diff)) <= tol


def test_adjoint_op():
	for i in range(len(Mx3list)):
		for j in range(len(Mx3list)):
			yield check_adjoint_op, (i,j,1e-16)


if __name__ == "__main__":
	print("==================== Test matrix identities ====================")
	np.set_printoptions(linewidth=2000, precision=4, threshold=10000, suppress=False)

	for t,c in test_adjoint_op():
		t(c)

