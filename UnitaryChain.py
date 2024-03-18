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
		"""Returns the U at step n, where 0 <= n < N."""
		return self.Us[n+1] @ self.Us[n].conj().T

