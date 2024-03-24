import numpy as np
import scipy as sp


class UnitaryChain(object):

##	Us[i] stores U[i-1] U[i-2] ... U[1] U[0]

	def __init__(self, Utarget):
		self.N = 1		## number of steps
		assert isinstance(Utarget, np.ndarray)
		self.d = len(Utarget)		## size of unitary matrix
		assert Utarget.shape == (self.d, self.d)
		self.Utarget = Utarget

		self.dtype = complex		## work everything out with complex numbers
		self.Us = [ np.eye(self.d, dtype=self.dtype), self.Utarget.copy() ]
		self.check_consistency()


	def check_consistency(self, tol=1e-14):
		d = self.d
		N = self.N
		dtype = self.dtype
		Utarget = self.Utarget
		Us = self.Us
		assert type(d) == int and d > 0
		assert isinstance(Us, list)
		assert len(Us) == N+1
		unitarity = np.zeros(N+1)
		IMx = np.eye(d)
		for i in range(N+1):
			U = Us[i]
			assert id(U) != id(Utarget)		## make sures that references aren't duplicated
			assert isinstance(U, np.ndarray) and U.shape == (d,d)
			unitarity[i] = np.max(np.abs(U.conj().T @ U - IMx))		## determine how close each matrix is to unitary
		#print(unitarity)


	def U(self, n):
		"""Returns the unitary matrix U at step n, where 0 <= n < N."""
		return self.Us[n+1] @ self.Us[n].conj().T


	def subdivide_at_step(self, step, num_div):
		"""Evenly subdivide the unitary at step (step) into num_div pieces.
The resulting UnitaryChain has (num_div-1) extra steps."""
		assert num_div > 0
		Us = self.Us
		Ustart = Us[step]
		Ustep = Us[step+1] @ Ustart.conj().T
	##	Diagonalize matrix
		v,w = np.linalg.eig(Ustep)
	##	At this stage: Ustep = w @ diag(v) @ inv(w)
		arg_v = np.angle(v)
		Us_insert = []
		invw_Ustart = np.linalg.inv(w) @ Ustart
		for i in range(1, num_div):
			D = np.diag(np.exp(1j * arg_v * i / num_div))
			Us_insert.append(w @ D @ invw_Ustart)
	##	Add extra matrices
		self.Us = Us[:step+1] + Us_insert + Us[step+1:]
		self.N += num_div - 1


