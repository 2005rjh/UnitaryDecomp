import numpy as np
import scipy as sp

## Note: we always use tabs for indentation


##################################################
## General array processing
def zero_if_close(a, tol=1e-14):
   """Take an np.ndarray and set small elements to zero."""
   if np.iscomplexobj(a):
      cr = np.array(np.abs(a.real) < tol, int)
      ci = np.array(np.abs(a.imag) < tol, int)
      ar = np.choose(cr, [a.real, np.zeros(a.shape)])
      ai = np.choose(ci, [a.imag, np.zeros(a.shape)])
      return ar + 1j * ai
   else:
      c = np.array(np.abs(a) < tol, int)
      return np.choose(c, [a, np.zeros_like(a)])


def zero_real_if_close(a, tol=1e-14):
   """Take an np.ndarray and set small elements to zero.  Make it real if all its imaginary components vanishes."""
   zic_a = zero_if_close(a, tol=tol)
   if np.count_nonzero(zic_a.imag) > 0: return zic_a
   return zic_a.real


def Gaussian_Hermitian(n, RNG=None, sigma=1.0):
	"""Return an n*n Hermitian matrix where each element is pull from a Gaussian distribution."""
	M = RNG.normal(scale=sigma, size=(n,n))
	Mupper = np.triu(M, k=1) / np.sqrt(2)
	Mlower = np.tril(M, k=-1) / np.sqrt(2)
	return np.diag(np.diag(M)) + Mupper.transpose() + Mupper + 1j * Mlower - 1j * Mlower.transpose()
	


##################################################
class UnitaryChain(object):

##	Us[i] stores U[i-1] U[i-2] ... U[1] U[0]

	def __init__(self, Utarget):
		self.N = 1		## number of steps
		assert isinstance(Utarget, np.ndarray)
		self.d = len(Utarget)		## size of unitary matrix
		assert Utarget.shape == (self.d, self.d)
		self.Utarget = Utarget.copy()
		self.Utarget.flags.writeable = False

		self.dtype = complex		## work everything out with complex numbers
		self.Us = [ np.eye(self.d, dtype=self.dtype), self.Utarget.copy() ]
		self.check_consistency()


	def check_consistency(self, tol=1e-14):
		d = self.d
		N = self.N
		dtype = self.dtype
		Utarget = self.Utarget
		Us = self.Us
		output = { 'Utarget unitarity': None, 'Ustep unitarity': np.zeros(N+1) }
		assert type(d) == int and d > 0
		assert type(N) == int and N > 0
		assert isinstance(Us, list)
		assert len(Us) == N+1
		IdMx = np.eye(d)
		output['Utarget unitarity'] = np.max(np.abs(Utarget.conj().T @ Utarget - IdMx))
		for i in range(N+1):
			U = Us[i]
			assert id(U) != id(Utarget)		## make sures that references aren't duplicated
			assert isinstance(U, np.ndarray) and U.shape == (d,d)
			if i == 0:
				output['Ustep unitarity'][0] = np.max(np.abs(U - IdMx))
			else:
				output['Ustep unitarity'][i] = np.max(np.abs(U.conj().T @ U - IdMx))		## determine how close each matrix is to unitary
		#print(unitarity)
		output['tol'] = max( np.max(output['Ustep unitarity']), output['Utarget unitarity'] )
		if type(tol) == float and output['tol'] > tol:
			raise ArithmeticError("UnitaryChain.check_consistency:  {} > tol ({})".format( output['tol'], tol ))
		return output


	def Ufinal(self):
		return self.Us[self.N]


	def U(self, s):
		"""Returns the unitary matrix U at step s, where 0 <= s < N."""
		return self.Us[s+1] @ self.Us[s].conj().T


	def logU(self, s):
		"""Returns the log of the unitary matrix U at step s, where 0 <= s < N."""
		Ustep = self.U(s)
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
		v,w = np.linalg.eig(Ustep)
		arg_v = np.angle(v)
		log_Ustep = w @ np.diag(arg_v) @ np.linalg.inv(w)
		return log_Ustep


	def Ufinal_to_Utarget(self):
		return self.Utarget @ self.Us[self.N].conj().T


	def subdivide_at_step(self, step, num_div):
		"""Evenly subdivide the unitary at step (step) into num_div pieces.
The resulting UnitaryChain has (num_div-1) extra steps."""
		assert num_div > 0
		Us = self.Us
		Ustart = Us[step]
		Ustep = Us[step+1] @ Ustart.conj().T
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
		v,w = np.linalg.eig(Ustep)
		arg_v = np.angle(v)
		Us_insert = []
		invw_Ustart = np.linalg.inv(w) @ Ustart
		for i in range(1, num_div):
		##	The i^th term is  Ustep^(i/n) @ Ustart  =  w @ diag(v^(i/n)) @ inv(w) @ Ustart
			D = np.diag(np.exp(1j * arg_v * i / num_div))
			Us_insert.append(w @ D @ invw_Ustart)
	##	Add extra matrices
		self.Us = Us[:step+1] + Us_insert + Us[step+1:]
		self.N += num_div - 1
		self.check_consistency()


##################################################
class qubit_unitary(UnitaryChain):

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 2


##################################################
