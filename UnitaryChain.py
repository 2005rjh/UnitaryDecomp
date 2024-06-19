import numpy as np
import scipy as sp
if np.version.version < '1.17.0': import scipy.linalg

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


def Frob_norm(M):
	#return np.linalg.norm(M, ord='fro')
	return np.sum(np.abs(M)**2)


def log_unitary(U):
	"""Returns (1/i) log(U), a Hermitian matrix.  Assumes U is a unitary matrix."""
	n = len(U)
	assert U.shape == (n,n)
	##	Diagonalize matrix:  Ustep = Z @ T @ Z^dag
	T,Z = sp.linalg.schur(U, output='complex')
	arg_v = np.angle(np.diag(T))
	log_U = Z @ np.diag(arg_v) @ np.conj(np.transpose(Z))
	return log_U

def log_unitary_old(U):
	"""Returns (1/i) log(U), a Hermitian matrix."""
	n = len(U)
	assert U.shape == (n,n)
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
	v,w = np.linalg.eig(U)		# TODO, use scipy.linalg.schur
	arg_v = np.angle(v)
	log_U = w @ np.diag(arg_v) @ np.linalg.inv(w)
	return log_U


def Gaussian_Hermitian(n, RNG=None, sigma=1.0):
	"""Return an n*n Hermitian matrix where each element is pull from a Gaussian distribution."""
	M = RNG.normal(scale=sigma, size=(n,n))
	Mupper = np.triu(M, k=1) / np.sqrt(2)
	Mlower = np.tril(M, k=-1) / np.sqrt(2)
	return np.diag(np.diag(M)) + Mupper.transpose() + Mupper + 1j * Mlower - 1j * Mlower.transpose()


def random_small_Unitary(n, RNG=None, sigma=1.0):
	return sp.linalg.expm(1j * Gaussian_Hermitian(n, RNG=RNG, sigma=sigma))


##################################################
def adjoint_op_MxRep(X):
	"""Returns adjoint[X] in the matrix reprensentation (dim = n^2).  X is a n*n matrix."""
	n = len(X)
	assert X.shape == (n,n)
	Id = np.identity(n, dtype=X.dtype)
	return np.kron(X, Id) - np.kron(Id, X.transpose())


def Mx_exprel(X):
	"""Compute [exp(X)-1] X^{-1} for a matrix X."""
	n = len(X)
	assert X.shape == (n,n)
	u,v = np.linalg.eig(X)		# X = v @ diag(u) @ inv(v)
	## scipy.special.exprel only takes in real numbers
	exprel_u = np.sinc(0.5j * u / np.pi) * np.exp(u / 2)
	return v @ np.diag(exprel_u) @ np.linalg.inv(v)

def Mx_nexprel(X):
	"""Compute [1-exp(-X)] X^{-1} for a matrix X."""
	n = len(X)
	assert X.shape == (n,n)
	u,v = np.linalg.eig(X)		# X = v @ diag(u) @ inv(v)
	## scipy.special.exprel only takes in real numbers
	exprel_u = np.sinc(0.5j * u / np.pi) * np.exp(-u / 2)
	return v @ np.diag(exprel_u) @ np.linalg.inv(v)

#TODO Mx_exprel_inv, Mx_nexprel_inv



################################################################################
################################################################################
class UnitaryChain(object):

##	Vs[i] stores U[i-1] U[i-2] ... U[1] U[0]
##	Ufinal = Vs[N] = U[N-1] ... U[1] U[0]

	def __init__(self, Utarget):
		self.N = 1		## number of steps
		assert isinstance(Utarget, np.ndarray)
		self.d = len(Utarget)		## size of unitary matrix
		assert Utarget.shape == (self.d, self.d)
		self.Utarget = Utarget.copy()
		self.Utarget.flags.writeable = False

		self.dtype = complex		## work everything out with complex numbers
		self.Vs = [ np.eye(self.d, dtype=self.dtype), self.Utarget.copy() ]
		self.cache = { 'U_decomp':{}, 'weights2':{} }
		self.check_consistency()		#TODO optional


	def copy(self):
		"""Return a deep copy of the current object."""
		c = type(self)(self.Utarget)
		self._deepcopy_to_c(c)
		c.check_consistency()		#TODO optional
		return c

	def _deepcopy_to_c(self, c):
		"""Helper for copy().  Can be overloaded for derived classes."""
		c.N = N = self.N
		c.dtype = self.dtype
		c.Vs = [ self.Vs[i].copy() for i in range(N+1) ]
		c.cache['U_decomp'] = self.cache['U_decomp'].copy()
		#c.cache['weights2'] = self.cache['weights2'].copy()
##	somehow, recompuoting the weights is faster than copying the 'weights2' dictionary
		return c


	def check_consistency(self, tol=1e-13):
		d = self.d
		N = self.N
		dtype = self.dtype
		Utarget = self.Utarget
		Vs = self.Vs
		cache = self.cache
		output = { 'Utarget unitarity': None, 'Vs unitarity': np.zeros(N+1), 'U_decomp err': -np.ones(N), }
		assert type(d) == int and d > 0
		assert type(N) == int and N > 0
		assert isinstance(Vs, list)
		assert len(Vs) == N + 1
	##	check cache structure
		assert type(cache) == dict
		assert type(cache['U_decomp']) == dict
		assert type(cache['weights2']) == dict
	##	check matrix values
		def compareMx(M1, M2):
			return np.max(np.abs( M1 - M2 ))
		IdMx = np.eye(d)
		output['Utarget unitarity'] = compareMx( Utarget.conj().T @ Utarget , IdMx )
		for i in range(N+1):
			V = Vs[i]
			assert id(V) != id(Utarget)		## make sures that references aren't duplicated
			assert isinstance(V, np.ndarray) and V.shape == (d,d)
			if i == 0:
				output['Vs unitarity'][0] = compareMx( V , IdMx )
			else:
				output['Vs unitarity'][i] = compareMx( V.conj().T @ V , IdMx )		## determine how close each matrix is to unitary
				Ustep = V @ Vs[i-1].conj().T
				try:		## test that Ustep = Z @ e^{i v) @ Z^dag
					Uv, UZ = cache['U_decomp'][i-1]
					## todo, check UZ is unitary, Uv real
					output['U_decomp err'][i-1] = compareMx( UZ @ np.diag(np.exp(1j * Uv)) @ UZ.conj().T , Ustep )
				except KeyError:
					pass
		#print(unitarity)
	##
		output['err'] = max( output['Utarget unitarity'], np.max(output['Vs unitarity']), np.max(output['U_decomp err']), 0 )
		if type(tol) == float and output['err'] > tol:
			raise ArithmeticError("UnitaryChain.check_consistency:  {} > tol ({})".format( output['err'], tol ))
		return output

	##################################################
	##	Retrieval

	def Ufinal(self):
		return self.Vs[self.N]


	def U(self, s):
		"""Returns the unitary matrix U at step s, where 0 <= s < N."""
		return self.Vs[s+1] @ self.Vs[s].conj().T


	def logU(self, s):
		"""Returns the log of the unitary matrix U at step s, where 0 <= s < N."""
		v, Z = self.U_decomp(s)
		return Z @ np.diag(v) @ Z.conj().T
	##	Old code:
		#return log_unitary(self.U(s))


	def U_to_target(self, V=None):
		"""Return the unitary needed to reach Utarget from V.  If V is None, then use Ufinal."""
		if V is None: V = self.Vs[self.N]
		return self.Utarget @ V.conj().T


	##	weights
	def weight_at_step(self, s):
		if s in self.cache['weights2']: return self.cache['weights2'][s]	# look up in cache
		w2 = self.compute_weight2_at_step(s)
		self.cache['weights2'][s] = w2	# cache value
		return w2

	def compute_weight2_at_step(self, s):
		raise NotImplementedError		# to be overloaded

	def weight_to_target(self):
		raise NotImplementedError		# to be overloaded

	def weight_list(self):
		w = [ self.weight_at_step(s) for s in range(self.N) ] + [ self.weight_to_target() ]
		return np.array(w);

	def weight_total(self):
		w = [ self.weight_at_step(s) for s in range(self.N) ] + [ self.weight_to_target() ]
		return np.sum(w)


	##	human-readable-ish output
	def str(self):
		s = ""
		for i in range(self.N):
			s += "Step {}:  (weight = {})\n".format( i, self.weight_at_step(i) ) + str(zero_real_if_close(self.logU(i))) + "\n"
		s += "Final U:\n" + str(zero_real_if_close(self.Ufinal())) + "\n"
		s += "U to target:  (weight = {})\n".format( self.weight_to_target() ) + str(zero_real_if_close(self.U_to_target())) + "\n"
		s += "Total weight: {}\n".format( self.weight_total() )
		return s


	##################################################
	##

	def update_V_at_step(self, s, newV):
		"""Update Vs[s] to newV.
s is an integer between 1 <= s <= N.  This will alter steps s-1 and s."""
		assert isinstance(newV, np.ndarray) and newV.shape == (self.d, self.d)
		self.Vs[s] = newV.astype(self.dtype)
		self.invalidate_cache_at_step(s - 1); self.invalidate_cache_at_step(s)

	def apply_U_to_V_at_step(self, s, U):
		"""Update Vs[s] -> U Vs[s], where U is a d*d unitary matrix (assuming U is unitary).
s is an integer between 1 <= s <= N.  This will alter steps s-1 and s."""
		self.Vs[s] = U @ self.Vs[s]
		self.invalidate_cache_at_step(s - 1); self.invalidate_cache_at_step(s)

	def apply_H_to_V_at_step(self, s, H):
		"""Update Vs[s] -> exp[iH] Vs[s], where H is a d*d Hermitian matrix.
s is an integer between 1 <= s <= N.  This will alter steps s-1 and s."""
		A = 0.5j * ( H + np.conj(np.transpose(H)) )
		self.Vs[s] = sp.linalg.expm(A) @ self.Vs[s]
		self.invalidate_cache_at_step(s - 1); self.invalidate_cache_at_step(s)


	def invalidate_cache_at_step(self, s):
		#TODO documentatation
		self.cache['weights2'].pop(s, None)
		self.cache['U_decomp'].pop(s, None)


	def subdivide_at_step(self, step, num_div):
		"""Evenly subdivide the unitary at step (step) into num_div pieces.
The resulting UnitaryChain has (num_div-1) extra steps."""
		assert num_div > 0
		Vs = self.Vs
		Vstart = Vs[step]
		Ustep = Vs[step+1] @ Vstart.conj().T
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
		v,w = np.linalg.eig(Ustep)		# TODO, use U_decomp()
		arg_v = np.angle(v)
		Vs_insert = []
		invw_Vstart = np.linalg.inv(w) @ Vstart
		for i in range(1, num_div):
		##	The i^th term is  Ustep^(i/n) @ Vstart  =  w @ diag(v^(i/n)) @ inv(w) @ Vstart
			D = np.diag(np.exp(1j * arg_v * i / num_div))
			Vs_insert.append(w @ D @ invw_Vstart)
	##	Add extra matrices
		self.Vs = Vs[:step+1] + Vs_insert + Vs[step+1:]
		self.N += num_div - 1
		#TODO, update cache
		self.check_consistency()


	def backup_Vs(self):
		self.backupVs = [ self.Vs[i].copy() for i in range(self.N+1) ]
		print("backup_Vs deprecated")

	def restore_from_backup_Vs(self):
		self.Vs = [ self.backupVs[i].copy() for i in range(self.N+1) ]
		print("restore_from_backup_Vs deprecated")


	##################################################
	##	Tools to compute weights and derivatives

	def U_decomp(self, s):
		"""Computes the spectral decompisition of the unitary matrix U at step s.
If 0 <= s < N, use U at that step.  If s == -1, use U_to_target instead.

Returns pair v, W.
exp[i v] are the eigenvalues of U, W are the eigenvectors, such that U = Z @ np.diag(np.exp(1j * v)) @ np.transpose(np.conj(Z)) ."""
		if s in self.cache['U_decomp']: return self.cache['U_decomp'][s]
		if s == -1: U = self.U_to_target(V=None)
		else: U = self.U(s)
		d = self.d
		T, Z = sp.linalg.schur(U, output='complex')
	##	Assumes U is unitary, so T is diagonal.
		v = np.angle(np.diag(T))
		#log_U = Z @ np.diag(v) @ Z.conj().T
		#print(np.max(np.abs( Z @ np.diag(np.exp(1j * v)) @ np.conj(np.transpose(Z)) - U )))
		self.cache['U_decomp'][s] = (v, Z)
		return v, Z


	def d_logU_before(self, s):
		"""Determines the 1st order change of (-i)logU (at step s) from altering Vs[s]."""
		raise NotImplementedError


	def d_logU_after(self, s):
		"""Determines the 1st order change of (-i)logU (at step s) from altering Vs[s+1]."""
		raise NotImplementedError


	##################################################



##################################################
##	TODO: UnitaryChain_MxCompWeight


##################################################
class qubit_unitary(UnitaryChain):
	"""Specialize to 1 single qubit.

coefficients:
	Rabi: the weight given to an X/Y (off diagonal) drives
	k: the weight given to I,X (diagonal) drives
"""

	##	class variables
	I2 = np.eye(2, dtype=float)
	PX = np.array([[0,1.],[1.,0]])
	PY = np.array([[0,-1j],[1j,0]])
	PZ = np.array([[1.,0],[0,-1.]])
	PauliList = [I2,PX,PY,PZ]

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 2
		self.coef = {'Rabi':1., 'k':15.}
		self.check_consistency()


	def compute_weight2_at_step(self, s):
		logU = self.logU(s)
		weight = ( np.abs(logU[0,0]**2) + np.abs(logU[1,1]**2) ) * self.coef['k']
		weight += np.abs(logU[0,1]**2) * self.coef['Rabi']
		return weight


	def weight_to_target(self, V=None):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate."""
		Utt = self.U_to_target(V=V)
		weight = Frob_norm(np.triu(Utt, k=1) + np.tril(Utt, k=-1))	# off diagonal term
		weight += Frob_norm(np.abs(np.diag(Utt)) - 1)		# measures how far the diagonal terms are to pure phases
		return self.coef['k'] * weight



##################################################
class two_qubits_unitary(UnitaryChain):
	"""Specialize to 2 qubits.

coefficients:
	Rabi1: the weight given to an single qubit X/Y drives (assigns Rabi1 to half Rabi period)
	Rabi2: the weight given to pair drives (assigns2 Rabi2 to half Rabi period of conversion or gain)
	penalty: the weight given to other drives
"""

	##	class variables
	I2 = np.eye(2, dtype=float)
	PX = np.array([[0,1.],[1.,0]])
	PY = np.array([[0,-1j],[1j,0]])
	PZ = np.array([[1.,0],[0,-1.]])
	PauliList = [I2,PX,PY,PZ]

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 4
		two_qubits_unitary.set_up_Pauli_operators()
		self.coef = {'Rabi1':0.1, 'Rabi2':1, 'penalty':10.}
	##	Set up weights
		self.MxComp_list = two_qubits_unitary.P2list		# divide by 4?
		R1, R2, pe = self.coef['Rabi1'], self.coef['Rabi2'], self.coef['penalty']
		self.MxComp_weights = np.array([ pe, R1, R1, pe, R1, 2*R2, 2*R2, pe, R1, 2*R2, 2*R2, pe, pe, pe, pe, pe ])
		self.check_consistency()

	@classmethod
	def set_up_Pauli_operators(cls):
		if hasattr(two_qubits_unitary, 'P2list'): return
		##	Pauli list:  11, 1X, 1Y, 1Z, X1, XX, XY, ..., ZY, ZZ
		two_qubits_unitary.P2list = np.array([ np.kron(P1,P2) for P1 in two_qubits_unitary.PauliList for P2 in two_qubits_unitary.PauliList ])
		two_qubits_unitary.P2list.flags.writeable = False
		assert two_qubits_unitary.P2list.shape == (16,4,4)

	def weight_to_target(self, V=None):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate on either qubit."""
		Utt = self.U_to_target(V=V)
		weight = Frob_norm(np.triu(Utt, k=1) + np.tril(Utt, k=-1))	# off diagonal term
		weight += Frob_norm(np.abs(np.diag(Utt)) - 1)		# measures how far the diagonal terms are to pure phases
		weight += np.abs( Utt[0,0] * Utt[1,1] - Utt[0,1] * Utt[1,0] )**2		# check how close the diagonal is to a Kronecker product
		return self.coef['penalty'] * weight

	def compute_weight2_at_step(self, s):
		logU = self.logU(s)		# returns a Hermitian matrix
		logUT = logU.transpose()
		##	Pauli components: Pcomp[i] = tr(P2[i] . logU) / 2pi, or logU = (pi/2) sum_i Pcomp[i] P2[i]
		MxComps = np.array([ np.sum(P * logUT) for P in two_qubits_unitary.P2list ]).real / (2 * np.pi)
		return np.sum(MxComps**2 * self.MxComp_weights)        



##################################################