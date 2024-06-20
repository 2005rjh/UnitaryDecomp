import numpy as np
import scipy as sp
import scipy.special
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
		self.reset_cache()
	##	specify UnitaryChain class because this __init__ is called by subclass' initializers, which at this point have not finished yet and so this object may not be subclass-consistent.
		UnitaryChain.check_consistency(self)		#optional


	def copy(self):
		"""Return a deep copy of the current object."""
		c = type(self)(self.Utarget)
		self._deepcopy_to_c(c)
		c.check_consistency()		#optional
		return c

	def _deepcopy_to_c(self, c):
		"""Helper for copy().  Can be overloaded for derived classes."""
		c.N = N = self.N
		c.dtype = self.dtype
		c.Vs = [ self.Vs[i].copy() for i in range(N+1) ]
		c.cache['U_decomp'] = self.cache['U_decomp'].copy()
		##	somehow, recompuoting the weights is faster than copying the 'weights2' dictionary
		#c.cache['weights2'] = self.cache['weights2'].copy()
		return c


	def check_consistency(self, tol=1e-13):
#TODO, add cmptxt
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
		def compareMx(M1, M2, cmptxt=None):
			maxdiff = np.max(np.abs( M1 - M2 ))
			if maxdiff > tol: print('UnitaryChain.check_consistency(tol = {}) failed{}'.format( tol, " for '"+str(cmptxt)+"'" if cmptxt is not None else '' ))
			return maxdiff
		IdMx = np.eye(d)
		output['Utarget unitarity'] = compareMx( Utarget.conj().T @ Utarget , IdMx )
		for i in range(N+1):
			V = Vs[i]
			assert id(V) != id(Utarget)		## make sures that references aren't duplicated
			assert isinstance(V, np.ndarray) and V.shape == (d,d)
			if i == 0:
				output['Vs unitarity'][0] = compareMx( V , IdMx )
				continue
			##	i > 0 from now on
			output['Vs unitarity'][i] = compareMx( V.conj().T @ V , IdMx )		## determine how close each matrix is to unitary
			Ustep = V @ Vs[i-1].conj().T
			try:		## test that Ustep = Z @ e^{i v) @ Z^dag
				Uv, UZ = cache['U_decomp'][i-1]
				assert isinstance(Uv, np.ndarray) and Uv.shape == (d,)
				assert Uv.dtype == float
				assert isinstance(UZ, np.ndarray) and UZ.shape == (d,d)
				output['U_decomp err'][i-1] = compareMx( UZ.conj().T @ UZ , IdMx )
				output['U_decomp err'][i-1] += compareMx( UZ @ np.diag(np.exp(1j * Uv)) @ UZ.conj().T , Ustep )
			except KeyError:
				pass
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
	def weight2_at_step(self, s):
		if s in self.cache['weights2']: return self.cache['weights2'][s]	# look up in cache
		w2 = self.compute_weight2_at_step(s)
		self.cache['weights2'][s] = w2	# cache value
		return w2

	def weight2_to_target(self):
		#TODO add to cache
		return self.compute_weight2_to_target(self.U_to_target())

	def compute_weight2_at_step(self, s):
		raise NotImplementedError		# must be overloaded

	def compute_weight2_to_target(self, U2t):
		raise NotImplementedError		# must be overloaded

	def weight2_list(self):
		w = [ self.weight2_at_step(s) for s in range(self.N) ] + [ self.weight2_to_target() ]
		return np.array(w);

	def weight_total(self):		# returns weight2_total
		w = [ self.weight2_at_step(s) for s in range(self.N) ] + [ self.weight2_to_target() ]
		return np.sum(w)

	def weight1_total(self):
		w = [ self.weight2_at_step(s) for s in range(self.N) ] + [ self.weight2_to_target() ]
		return np.sum(np.sqrt(w))


	##	human-readable-ish output
	def str(self):
		s = ""
		for i in range(self.N):
			s += "Step {}:  (weight2 = {})\n".format( i, self.weight2_at_step(i) ) + str(zero_real_if_close(self.logU(i))) + "\n"
		s += "Final U:\n" + str(zero_real_if_close(self.Ufinal())) + "\n"
		s += "U to target:  (weight2 = {})\n".format( self.weight2_to_target() ) + str(zero_real_if_close(self.U_to_target())) + "\n"
		s += "Total weight1: {}\n".format( self.weight1_total() )
		s += "Total weight2: {}\n".format( self.weight_total() )
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

	def apply_expiH_to_V_at_step(self, s, H):
		"""Update Vs[s] -> exp[iH] Vs[s], where H is a d*d Hermitian matrix.
s is an integer between 1 <= s <= N.  This will alter steps s-1 and s."""
		A = 0.5j * ( H + np.conj(np.transpose(H)) )
		self.Vs[s] = sp.linalg.expm(A) @ self.Vs[s]
		self.invalidate_cache_at_step(s - 1); self.invalidate_cache_at_step(s)


	def reset_cache(self):
		self.cache = { 'U_decomp':{}, 'weights2':{} }

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
		self.reset_cache()
		self.check_consistency()


	def del_Vs(self, s):
		"""Delete Vs[s] from the list, where 0 < s <= N.
If s < N, then this combines steps (s-1) with s into one step.  If s == N, then this removes the final step and makes Vs[N-1] the new Ufinal."""
		N = self.N
		if N == 1: raise RuntimeError("Can't have less than one step.")
		assert 0 < s and s <= N
		del self.Vs[s]
		self.N = N - 1
		self.reset_cache()
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
		"""Computes the spectral decomposition of the unitary matrix U at step s.
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

	##	End of UnitaryChain class
	##################################################



################################################################################
class UnitaryChain_MxCompWeight(UnitaryChain):
	"""An abstract subclass of UnitaryChain when the weight2's of each step are all quadratic functions of logU.

Assumes (a) the weight2 of each step is the same function,
(b) weight2_to_target enforces U_to_target to be diagonal,
and (c) ... [something about U2t_DiagTests]

Required attributes:
	MxComp_list:  a list of matrices
	ConjMxComp_list:  a list of matrices conjugate to MxComp_list (under Frobenius norm)
	MxComp_weights2:
	U2t_DiagTests:
Also the class will have to overload _deepcopy_to_c() to make sure these attributes are copied.

Formula:
	MxComps_i = 2 / pi * Tr[ ConjMxComp_list[i] . logU ]
	weight2(Ustep) = sum_i MxComp_weights2_i MxComps_i^2
	weight2(U2target) = ODpenalty + ...
"""

	def check_consistency(self, tol=1e-13):
		output = super().check_consistency(tol=tol)
		d = self.d
		MxComp_list = self.MxComp_list
		ConjMxComp_list = self.ConjMxComp_list
		MxComp_weights2 = self.MxComp_weights2
		U2t_DiagTests = self.U2t_DiagTests
		nMx = d**2
		def compareMx(M1, M2, cmptxt=None):
			maxdiff = np.max(np.abs( M1 - M2 ))
			if maxdiff > tol: print('UnitaryChain_MxCompWeight.check_consistency(tol = {}) failed{}'.format( tol, " for '"+str(cmptxt)+"'" if cmptxt is not None else '' ))
			return maxdiff
	##	check MxComp_weights2
		assert isinstance(MxComp_weights2, np.ndarray) and MxComp_weights2.shape == (nMx,)
		assert MxComp_weights2.dtype == float
	##	check MxComp_list and ConjMxComp_list
		assert isinstance(MxComp_list, np.ndarray) and MxComp_list.shape == (nMx, d, d)
		assert isinstance(ConjMxComp_list, np.ndarray) and MxComp_list.shape == (nMx, d, d)
		output['ConjMxCom Herm'] = compareMx( ConjMxComp_list.transpose(0,2,1).conj() , ConjMxComp_list, 'ConjMxCom Herm' )
		output['MxComp compat'] = compareMx( np.dot( MxComp_list.reshape(nMx, d**2) , ConjMxComp_list.reshape(nMx, d**2).conj().T ) , np.eye(nMx), 'MxComp compat' )
	##	check U2t_DiagTests
		assert isinstance(U2t_DiagTests, list)
		for chk in U2t_DiagTests:
			assert isinstance(chk, tuple) and len(chk) == 4
			#TODO check everything is within [0,d)
	##
		#TODO, add MxComp_compat
		output['err'] = max( output['ConjMxCom Herm'], output['MxComp compat'], 0, output['err'] )
		if type(tol) == float and output['err'] > tol:
			raise ArithmeticError("UnitaryChain_MxCompWeight.check_consistency:  {} > tol ({})".format( output['err'], tol ))
		return output


	def compute_weight2_at_step(self, s):
		jlogU = self.logU(s)		# expects a Hermitian matrix
		d = self.d
		jlogUT = jlogU.conj()
		MxComps = np.array([ np.sum(P * jlogUT) for P in self.ConjMxComp_list ]).real / (np.pi/2)
	##	Matrix components: jlogU = (pi/2) sum_i MxComps[i] MxComp_list[i]
		#print("MxComps", MxComps)
		#print("MxComp_weights2", self.MxComp_weights2)
		return np.sum(MxComps**2 * self.MxComp_weights2)


	def compute_weight2_to_target(self, U2t):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate on either qubit."""
		dist2 = 0		# how 'far' U2t is from an acceptible phase gate
		D = np.diag(U2t)		# diagonal part of U2t
		##	we penalize off-diagonal terms and how far the diagonal are pure phases
		## since frobnorm(U) = d, we only need the diagonal elements to compute the distance
		dist2 += 2 * np.sum(1 - np.abs(D)**2)		# measures how far the diagonal terms are to pure phases
		##	equvalent to: dist2 += Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1)) + np.sum(1 - np.abs(D)**2)
		#print("OD weight =", dist2); return dist2
		for chk in self.U2t_DiagTests:
			dist2 += np.abs( D[chk[0]] * D[chk[3]] - D[chk[1]] * D[chk[2]] )**2
		return self.coef['penalty']**2 * dist2


	def compute_grad_weight2_at_step(self, s):
		"""Compute the gradient of compute_weight2_at_step() with respect to H* (applied to Vs[s] or Vs[s+1])
Specifically:
	grHL_{i,j} = d compute_weight2_at_step( exp[i HL] . U[s] ) / d HL_{i,j}*
	grHR_{i,j} = d compute_weight2_at_step( U[s] . exp[-i HR] ) / d HR_{i,j}*
or equivalently
	Delta compute_weight2_at_step( exp[i HL] . U[s] ) ~ Tr[ grHL . Delta(HL) ] + ...
	Delta compute_weight2_at_step( U[s] . exp[-i HR] ) ~ Tr[ grHR . Delta(HR) ] + ...

Caution: gradient code may not work well if v is close to +-pi.
"""
		assert 0 <= s and s < self.N
		d = self.d
		ConjMx = self.ConjMxComp_list
		v, Z = self.U_decomp(s)
		jlogU = Z @ np.diag(v) @ Z.conj().T		# is this needed?
		## Conjugate each matrix in ConjMx, M -> Z^dag M Z
		##	ZConjMx.shape = (nMx, d, d)
		ZConjMx = np.tensordot(np.tensordot(Z.conj(), ConjMx, axes=[[0],[1]]), Z, axes=[2,0]).transpose(1,0,2)
		commMx = np.repeat(v, d).reshape(d, d) - v		# commutation matrix, such that [i diag(v), M] = i commMx * M (Hadamard product)
		phiM = np.exp(-0.5j * commMx) / sp.special.sinc(0.5 * commMx / np.pi)
		phiP = phiM + 1j * commMx		# also same as phiM.T
		grHL = np.zeros((d,d), dtype=complex)
		grHR = np.zeros((d,d), dtype=complex)
		##	d[weight] = (8/pi^2) sum_i MxComp_weights2_i Tr[ ConjM[i] jlogU ] Tr[ ConjM[i] d[jlogU] ]
		## Recall jlogU = log[U] / i
		##	Tr[ X d[jlogU] ] = Tr[ phiP[X] dHL ] - Tr[ phiM[X] dHR ]
		MxComps = (2/np.pi) * np.array([ np.sum(np.diag(M).real * v) for M in ZConjMx ])
		for i in range(len(ConjMx)):
			phiP_ConjMx = Z @ (phiP * ZConjMx[i]) @ Z.conj().T
			phiM_ConjMx = Z @ (phiM * ZConjMx[i]) @ Z.conj().T
			grHL += (4/np.pi) * self.MxComp_weights2[i] * MxComps[i] * phiP_ConjMx
			grHR -= (4/np.pi) * self.MxComp_weights2[i] * MxComps[i] * phiM_ConjMx
		#print(phiP, "= phiP")
		#print(phiM, "= phiM")
		#print("weight2 = ", np.sum(MxComps**2 * self.MxComp_weights2))
		return grHL, grHR


	def compute_grad_weight2_to_target(self, U2t):
		"""Compute the gradient of compute_weight2_to_target() with respect to H* (applied to Vs[N])
Specifically:  d compute_weight2_to_target(U2t . exp[-i H]) / d H_{i,j}*
"""
		d = self.d
		D = np.diag(U2t)
		##	d[weight2] = sum_i grD_i d[D]_{i,i} + c.c.
		grD = np.zeros(d, dtype=complex)
		for chk in self.U2t_DiagTests:
			c0,c1,c2,c3 = chk		# add to grD terms from U2t_DiagTests
			chk_valc = np.conj( D[c0] * D[c3] - D[c1] * D[c2] )
			grD[c0] += D[c3] * chk_valc
			grD[c1] -= D[c2] * chk_valc
			grD[c2] -= D[c1] * chk_valc
			grD[c3] += D[c0] * chk_valc
		grD += -2 * D.conj()		# the portion from grad sum(1 - |D|^2)
		## convert from grD to grH:  d[weight2] = sum_{i,j} grH_{i,j} d[H]_{i,j}
		grH = -1j * U2t.T * grD
		grH = grH + grH.conj().T
		#print(grH, "= grad w2t")
		return grH.conj() * self.coef['penalty']**2



################################################################################
class qubit_unitary(UnitaryChain_MxCompWeight):
	"""Specialize to 1 single qubit.

coefficients:
	Rabi: the weight given to an X/Y (off diagonal) drives
	k: the weight given to I,X (diagonal) drives
"""
#TODO, make this a derived class of UnitaryChain_MxCompWeight

	##	class variables
	I2 = np.eye(2, dtype=float)
	PX = np.array([[0,1.],[1.,0]])
	PY = np.array([[0,-1j],[1j,0]])
	PZ = np.array([[1.,0],[0,-1.]])

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 2
		self.coef = {'Rabi':1., 'penalty':np.sqrt(15)}
		self.set_coef()
	##	Set up weights
		qubit_unitary.set_up_MxComp_lists()
		self.U2t_DiagTests = []		# no constraints on the phases of U_to_target
	##	Done!
		self.check_consistency()


	def set_coef(self, Rabi=None, penalty=None):
		if Rabi is not None:
			assert type(Rabi) == float and 0 <= Rabi
			self.coef['Rabi'] = float(Rabi)
		if penalty is not None:
			assert type(penalty) == float and 0 <= penalty
			self.coef['penalty'] = float(penalty)
		R1 = self.coef['Rabi']**2; pe = self.coef['penalty']**2
		self.MxComp_weights2 = np.array([ pe, R1, R1, pe ])


	def _deepcopy_to_c(self, c):
		super()._deepcopy_to_c(c)
		c.coef = self.coef.copy()
		c.set_coef()


	@classmethod
	def set_up_MxComp_lists(cls):
		if hasattr(qubit_unitary, 'ConjMxComp_list'): return
		qubit_unitary.PauliList = np.array([cls.I2, cls.PX, cls.PY, cls.PZ])
		qubit_unitary.PauliList.flags.writeable = False
		assert qubit_unitary.PauliList.shape == (4,2,2)
		qubit_unitary.MxComp_list = qubit_unitary.PauliList
		qubit_unitary.ConjMxComp_list = qubit_unitary.PauliList / 2
		qubit_unitary.ConjMxComp_list.flags.writeable = False


	def _old_compute_weight2_at_step(self, s):
		logU = self.logU(s)
		weight = ( np.abs(logU[0,0]**2) + np.abs(logU[1,1]**2) ) * self.coef['penalty']**2
		weight += np.abs(logU[0,1]**2) * self.coef['Rabi']**2
		return weight


	def _old_compute_weight2_to_target(self, U2t):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate."""
		weight = Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1))	# off diagonal term
		weight += Frob_norm(np.abs(np.diag(U2t)) - 1)		# measures how far the diagonal terms are to pure phases
		return self.coef['penalty']**2 * weight



##################################################
class two_qubits_unitary(UnitaryChain_MxCompWeight):
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
		self.coef = {'Rabi1':0.1, 'Rabi2':1, 'penalty':10.}
		self.set_coef()
	##	Set up weights
		two_qubits_unitary.set_up_Pauli_operators()
		self.U2t_DiagTests = [ (0,1,2,3), ]		# check how close the diagonal is to a Kronecker product
	##	Done!
		self.check_consistency()


	def set_coef(self, Rabi1=None, Rabi2=None, penalty=None):
		if Rabi1 is not None:
			assert type(Rabi1) == float and 0 <= Rabi1
			self.coef['Rabi1'] = float(Rabi1)
		if Rabi2 is not None:
			assert type(Rabi2) == float and 0 <= Rabi2
			self.coef['Rabi2'] = float(Rabi2)
		if penalty is not None:
			assert type(penalty) == float and 0 <= penalty
			self.coef['penalty'] = float(penalty)
		R1 = self.coef['Rabi1']**2; R2 = self.coef['Rabi2']**2; pe = self.coef['penalty']**2
		self.MxComp_weights2 = np.array([ pe, R1, R1, pe, R1, 2*R2, 2*R2, pe, R1, 2*R2, 2*R2, pe, pe, pe, pe, pe ])


	def _deepcopy_to_c(self, c):
		super()._deepcopy_to_c(c)
		c.coef = self.coef.copy()
		c.set_coef()


	@classmethod
	def set_up_Pauli_operators(cls):
		if hasattr(two_qubits_unitary, 'P2list'): return
		##	Pauli list:  11, 1X, 1Y, 1Z, X1, XX, XY, ..., ZY, ZZ
		two_qubits_unitary.P2list = np.array([ np.kron(P1,P2) for P1 in two_qubits_unitary.PauliList for P2 in two_qubits_unitary.PauliList ])
		two_qubits_unitary.P2list.flags.writeable = False
		assert two_qubits_unitary.P2list.shape == (16,4,4)
		two_qubits_unitary.MxComp_list = two_qubits_unitary.P2list
		two_qubits_unitary.ConjMxComp_list = two_qubits_unitary.P2list / 4
		two_qubits_unitary.ConjMxComp_list.flags.writeable = False


	##	compute_weight2_at_step() and compute_weight2_to_target() derived from UnitaryChain_MxCompWeight

	def _old_compute_weight2_to_target(self, U2t):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate on either qubit."""
		weight = Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1))	# off diagonal term
		Dg = np.diag(U2t)		# diagonal part of U2t
		weight += Frob_norm(np.abs(Dg) - 1)		# measures how far the diagonal terms are to pure phases
		weight += np.abs( Dg[0] * Dg[3] - Dg[1] * Dg[2] )**2		# check how close the diagonal is to a Kronecker product
		return self.coef['penalty'] * weight

	def _old_compute_weight2_at_step(self, s):
	## superseded by UnitaryChain_MxCompWeight.compute_weight2_at_step()
		logU = self.logU(s)		# returns a Hermitian matrix
		logUT = logU.transpose()
		##	Pauli components: Pcomp[i] = tr(P2[i] . logU) / 2pi, or logU = (pi/2) sum_i Pcomp[i] P2[i]
		MxComps = np.array([ np.sum(P * logUT) for P in two_qubits_unitary.P2list ]).real / (2 * np.pi)
		return np.sum(MxComps**2 * self.MxComp_weights2)



##################################################
