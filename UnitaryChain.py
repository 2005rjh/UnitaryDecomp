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
	"""Returns (1/i) log(U), a Hermitian matrix."""
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
	v,w = np.linalg.eig(U)
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


## use scipy.special.exprel


##################################################
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
		self.weight_func = [ UnitaryChain.zero_weight_U ] * self.N
		self.check_consistency()


	def check_consistency(self, tol=1e-14):
		d = self.d
		N = self.N
		dtype = self.dtype
		Utarget = self.Utarget
		Vs = self.Vs
		weight_func = self.weight_func
		output = { 'Utarget unitarity': None, 'Vs unitarity': np.zeros(N+1) }
		assert type(d) == int and d > 0
		assert type(N) == int and N > 0
		assert isinstance(Vs, list)
		assert len(Vs) == N+1
		IdMx = np.eye(d)
		output['Utarget unitarity'] = np.max(np.abs(Utarget.conj().T @ Utarget - IdMx))
		for i in range(N+1):
			V = Vs[i]
			assert id(V) != id(Utarget)		## make sures that references aren't duplicated
			assert isinstance(V, np.ndarray) and V.shape == (d,d)
			if i == 0:
				output['Vs unitarity'][0] = np.max(np.abs(V - IdMx))
			else:
				output['Vs unitarity'][i] = np.max(np.abs(V.conj().T @ V - IdMx))		## determine how close each matrix is to unitary
		#print(unitarity)
		assert len(weight_func) == N
	##
		output['tol'] = max( np.max(output['Vs unitarity']), output['Utarget unitarity'] )
		if type(tol) == float and output['tol'] > tol:
			raise ArithmeticError("UnitaryChain.check_consistency:  {} > tol ({})".format( output['tol'], tol ))
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
		return log_unitary(self.U(s))


	def U_to_target(self, V=None):
		"""Return the unitary needed to reach Utarget from V.  If V is None, then use Ufinal."""
		if V is None: V = self.Vs[self.N]
		return self.Utarget @ V.conj().T


	def weight_at_step(self, s):
		return self.weight_func[s](self.U(s))


	def weight_total(self):
		w = [ self.weight_at_step(s) for s in range(self.N) ] + [ self.weight_to_target() ]
		return np.sum(w)

	def weight_list(self):
		w = [self.weight_at_step(s) for s in range(self.N)] + [ self.weight_to_target() ]
		return np.array(w);

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

	def subdivide_at_step(self, step, num_div):
		"""Evenly subdivide the unitary at step (step) into num_div pieces.
The resulting UnitaryChain has (num_div-1) extra steps."""
		assert num_div > 0
		Vs = self.Vs
		Vstart = Vs[step]
		Ustep = Vs[step+1] @ Vstart.conj().T
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
		v,w = np.linalg.eig(Ustep)
		arg_v = np.angle(v)
		Vs_insert = []
		invw_Vstart = np.linalg.inv(w) @ Vstart
		for i in range(1, num_div):
		##	The i^th term is  Ustep^(i/n) @ Vstart  =  w @ diag(v^(i/n)) @ inv(w) @ Vstart
			D = np.diag(np.exp(1j * arg_v * i / num_div))
			Vs_insert.append(w @ D @ invw_Vstart)
	##	Add extra matrices
		self.Vs = Vs[:step+1] + Vs_insert + Vs[step+1:]
		self.weight_func = self.weight_func[:step] + [ self.weight_func[step] ] * num_div + self.weight_func[step+1:]
		self.N += num_div - 1
		self.check_consistency()


	def backup_Vs(self):
		self.backupVs = [ self.Vs[i].copy() for i in range(self.N+1) ]


	def restore_from_backup_Vs(self):
		self.Vs = [ self.backupVs[i].copy() for i in range(self.N+1) ]


##	function as placeholder
	@classmethod
	def zero_weight_U(cls, U):
		return 0.

##	to be overloaded
	def weight_to_target(self):
		return 0.



##################################################
class qubit_unitary(UnitaryChain):
	"""Specialize to 1 single qubit.

coefficients:
	Rabi: the weight given to an X/Y (off diagonal) drives
	k: the weight given to I,X (diagonal) drives
"""

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 2
		self.coef = {'Rabi':1., 'k':15.}
		self.weight_func = [ self.weight_of_U ] * self.N
		self.check_consistency()


	def weight_to_target(self, V=None):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate."""
		Utt = self.U_to_target(V=V)
		weight = Frob_norm(np.triu(Utt, k=1) + np.tril(Utt, k=-1))	# off diagonal term
		weight += Frob_norm(np.abs(np.diag(Utt)) - 1)		# measures how far the diagonal terms are to pure phases
		return self.coef['k'] * weight


	## default weighting function for qubit gates
	def weight_of_U(self, U):
		logU = log_unitary(U)
		weight = ( np.abs(logU[0,0]**2) + np.abs(logU[1,1]**2) ) * self.coef['k']
		weight += np.abs(logU[0,1]**2) * self.coef['Rabi']
		return weight



##################################################
class two_qubits_unitary(UnitaryChain):
	"""Specialize to 2 qubits.

coefficients:
	Rabi1: the weight given to an single qubit X/Y drives (assigns Rabi1 to half Rabi period)
	Rabi2: the weight given to pair drives (assigns2 Rabi2 to half Rabi period of conversion or gain)
	penalty: the weight given to other drives
"""
	Id2 = np.eye(2, dtype=float)
	PX = np.array([[0,1.],[1.,0]])
	PY = np.array([[0,-1j],[1j,0]])
	PZ = np.array([[1.,0],[0,-1.]])
	PauliList = [Id2,PX,PY,PZ]

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 4
		self.coef = {'Rabi1':0.1, 'Rabi2':1, 'penalty':10.}
		self.weight_func = [ self.weight_of_U ] * self.N
		if not hasattr(two_qubits_unitary, 'P2list'):
			#two_qubits_unitary.P2list = [ [ np.kron(P1,P2) for P2 in two_qubits_unitary.PauliList ] for P1 in two_qubits_unitary.PauliList ]
			##	list:  11, 1X, 1Y, 1Z, X1, XX, XY, ..., ZY, ZZ
			two_qubits_unitary.P2list = [ np.kron(P1,P2) for P1 in two_qubits_unitary.PauliList for P2 in two_qubits_unitary.PauliList ]
			assert np.array(two_qubits_unitary.P2list).shape == (16,4,4)
		self.check_consistency()

	def weight_to_target(self, V=None):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate on either qubit."""
		Utt = self.U_to_target(V=V)
		weight = Frob_norm(np.triu(Utt, k=1) + np.tril(Utt, k=-1))	# off diagonal term
		weight += Frob_norm(np.abs(np.diag(Utt)) - 1)		# measures how far the diagonal terms are to pure phases
		weight += np.abs( Utt[0,0] * Utt[1,1] - Utt[0,1] * Utt[1,0] )**2		# check how close the diagonal is to a Kronecker product
		return self.coef['penalty'] * weight

	## default weighting function for qubit gates
	def weight_of_U(self, U):
		logU = log_unitary(U)		# returns a Hermitian matrix
		logUT = logU.transpose()
		##	Pauli components: Pcomp[i] = tr(P2[i] . logU) / 2pi, or logU = (pi/2) sum_i Pcomp[i] P2[i]
		Pcomps = np.array([ np.sum(P * logUT) for P in two_qubits_unitary.P2list ]).real / (2 * np.pi)
		R1, R2, p = self.coef['Rabi1'], self.coef['Rabi2'], self.coef['penalty']
		wcomps = np.array([ p, R1, R1, p, R1, 2*R2, 2*R2, p, R1, 2*R2, 2*R2, p, p, p, p, p ])
		return np.sum(Pcomps**2 * wcomps)
		


##################################################
