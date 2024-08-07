import numpy as np
import scipy as sp
import scipy.special
if np.version.version < '1.17.0':
	import scipy.linalg
	import scipy.sparse.linalg
import stringtools

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


def rank1tensor_approx(a):
	"""Find a rank-1 tensor approximation to input tensor (a).
Returns an nd-array with the same shape as the input.
Note: the returned array may having a different dtype (e.g. float if input is int)."""
	ndim = a.ndim
	sh = a.shape
	## TODO: checks on shape?
	if ndim <= 1: return a.copy()
	if ndim == 2:
		U, s, Vh = sp.linalg.svd(a, full_matrices=False, compute_uv=True, check_finite=True)
		return np.outer(U[:,0], s[0] * Vh[0])
		## TODO: consider using sparse scipy.sparse.linalg.svds
	raise NotImplementedError
	##	TODO: alternating least squares (ALS) or alternating SVD (ASVD) method


##################################################
## Matrix functions
def Frob_norm(M):
	#return np.linalg.norm(M, ord='fro')
	return np.sum(np.abs(M)**2)


def unitarize(M):
	"""Given M, returns the 'closest' unitary matrix to M.  More precisely, the unitary in the polar decomposition of M."""
	U, s, Vh = sp.linalg.svd(M, compute_uv=True, overwrite_a=False)
	return U @ Vh


def jlog_unitary(U):
	"""Returns (1/i) log(U), a Hermitian matrix.  Assumes U is a unitary matrix."""
	n = len(U)
	assert U.shape == (n,n)
	##	Diagonalize matrix:  Ustep = Z @ T @ Z^dag
	T,Z = sp.linalg.schur(U, output='complex')
	arg_v = np.angle(np.diag(T))
#TODO can make this more efficient
	log_U = Z @ np.diag(arg_v) @ np.conj(np.transpose(Z))
	return log_U

def jlog_unitary_old(U):
	"""Returns (1/i) log(U), a Hermitian matrix."""
	n = len(U)
	assert U.shape == (n,n)
	##	Diagonalize matrix:  Ustep = w @ diag(v) @ inv(w)
	v,w = np.linalg.eig(U)		# TODO, use scipy.linalg.schur
	arg_v = np.angle(v)
	log_U = w @ np.diag(arg_v) @ np.linalg.inv(w)
	return log_U


##################################################
##	Random matrices
def Gaussian_Hermitian(n, RNG=None, sigma=1.0):
	"""Return an n*n Hermitian matrix where each element is pull from a Gaussian distribution."""
	M = RNG.normal(scale=sigma, size=(n,n))
	Mupper = np.triu(M, k=1) / np.sqrt(2)
	Mlower = np.tril(M, k=-1) / np.sqrt(2)
	return np.diag(np.diag(M)) + Mupper.transpose() + Mupper + 1j * Mlower - 1j * Mlower.transpose()


def random_Unitary(n, RNG=None):
	"""Return a Haar-random unitary matrix."""
	assert isinstance(n, (int, np.integer))
	A = RNG.normal(scale=1, size=(n,n)) + 1j * RNG.normal(scale=1, size=(n,n))
	Q, R = sp.linalg.qr(A, mode='full', pivoting=False)
	RD = np.diag(R)
	return Q * (np.abs(RD)/RD)


def random_small_Unitary(n, RNG=None, sigma=1.0):
	"""Return a unitary matrix close to the identity."""
	assert isinstance(n, (int, np.integer))
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
##	such that U[i] = V[i+1] V[i]^{-1}
##	Vfinal = Vs[N] = U[N-1] ... U[1] U[0]
##	Utarget = U_to_target Vfinal, or equivalently U_to_target = Utarget Vfinal^{-1}

	def __init__(self, Utarget):
		self.N = 1		## number of steps
		assert isinstance(Utarget, np.ndarray)
		self.d = len(Utarget)		## size of unitary matrix
		assert Utarget.shape == (self.d, self.d)
		self.Utarget = Utarget.copy()
		self.Utarget.flags.writeable = False

		self.dtype = complex		## work everything out with complex numbers
		self.Vs = [ np.eye(self.d, dtype=self.dtype), self.Utarget.astype(self.dtype, copy=True) ]
		self.coef = {}
		self.reset_cache()
	##	specify UnitaryChain class because this __init__ is called by subclass' initializers, which at this point have not finished yet and so this object may not be subclass-consistent.
		#UnitaryChain.check_consistency(self)		#optional


	def copy(self):
		"""Return a deep copy of the current object.  Overloading not recommend (overload _deepcopy_to_c instead)."""
		c = type(self)(self.Utarget)
		self._deepcopy_to_c(c)
		c.check_consistency()		#optional
		return c

	def _deepcopy_to_c(self, c):
		"""Helper for copy().  Can be overloaded for derived classes.

Note for derived class: if self.coef contains mutable data, then its wise to overload this function making deepcopy of such objects."""
		c.N = N = self.N
		c.dtype = self.dtype
		c.Vs = [ self.Vs[i].copy() for i in range(N+1) ]
		c.coef = self.coef.copy()
		c.cache['U_decomp'] = self.cache['U_decomp'].copy()
		##	somehow, recompuoting the weights is faster than copying the 'weights2' dictionary
		#c.cache['weights2'] = self.cache['weights2'].copy()
		return c


	def check_consistency(self, tol=1e-13):
		d = self.d
		N = self.N
		dtype = self.dtype
		Utarget = self.Utarget
		Vs = self.Vs
		coef = self.coef
		cache = self.cache
		output = { 'Utarget unitarity': None, 'Vs unitarity': np.zeros(N+1), 'U_decomp err': np.zeros(N+1), }
		output['U_decomp err'].fill(-np.inf)
	##	basic checks
		assert type(d) == int and d > 0
		assert type(N) == int and N > 0
		assert isinstance(Vs, list)
		assert len(Vs) == N + 1
		assert isinstance(coef, dict)
	##	check cache structure
		assert type(cache) == dict
		assert type(cache['U_decomp']) == dict
		assert type(cache['weights2']) == dict
		assert type(cache['fragile']) == dict
		#TODO, check cache keys
	##	check matrix values
		def compareMx(M1, M2, cmptxt=None):
			maxdiff = np.max(np.abs( M1 - M2 ))
			if maxdiff > tol: print('UnitaryChain.check_consistency(tol = {}) failed{}'.format( tol, " for '"+str(cmptxt)+"'" if cmptxt is not None else '' ))
			return maxdiff
		IdMx = np.eye(d)
		assert isinstance(Utarget, np.ndarray) and Utarget.shape == (d, d)
		##	Utarget.dtype does not have to match self.dtype
		output['Utarget unitarity'] = compareMx( Utarget.conj().T @ Utarget , IdMx , "Utarget unitarity" )
		for i in range(N+1):
			V = Vs[i]
			assert id(V) != id(Utarget)		## make sures that references aren't duplicated
			assert isinstance(V, np.ndarray) and V.shape == (d,d)
			if np.isnan(V).any(): raise AssertionError
			if V.dtype != dtype:
				raise AssertionError("UnitaryChain.check_consistency():  mismatched dtype:  {} (Vs[{}]) != {}".format( V.dtype, i, dtype ))
			if i == 0:
				output['Vs unitarity'][0] = compareMx( V , IdMx , "Vs[0] unitarity ")
				continue
			##	i > 0 from now on
			output['Vs unitarity'][i] = compareMx( V.conj().T @ V , IdMx , "Vs["+str(i)+"] unitarity" )		## determine how close each matrix is to unitary
			Ustep = V @ Vs[i-1].conj().T
			try:		## test that Ustep = Z @ e^{i v) @ Z^dag
				Uv, UZ = cache['U_decomp'][i-1]
				assert isinstance(Uv, np.ndarray) and Uv.shape == (d,)
				assert Uv.dtype == float
				assert isinstance(UZ, np.ndarray) and UZ.shape == (d,d)
				output['U_decomp err'][i-1] = compareMx( UZ.conj().T @ UZ , IdMx , "U_decomp["+str(i-1)+"] unitarity" )
				output['U_decomp err'][i-1] += compareMx( UZ @ np.diag(np.exp(1j * Uv)) @ UZ.conj().T , Ustep , "U_decomp["+str(i-1)+"] err" )
			except KeyError: pass
	##
		output['err'] = max( output['Utarget unitarity'], np.max(output['Vs unitarity']), np.max(output['U_decomp err']), 0 )
		if type(tol) == float and output['err'] > tol:
			raise ArithmeticError("UnitaryChain.check_consistency:  {} > tol ({})".format( output['err'], tol ))
		return output


	##################################################
	##	Cache handling

	def reset_cache(self):
		self.cache = { 'U_decomp':{}, 'weights2':{}, 'fragile':{} }

#TODO: use array instead of dict for weights2 cache
	def reset_weight2_cache(self):
		self.cache['weights2'] = {}

	def invalidate_cache_at_step(self, s):
		#TODO documentatation
		self.cache['weights2'].pop(s, None)
		self.cache['U_decomp'].pop(s, None)
		self.cache['fragile'] = {}

	def invalidate_cache_at_point(self, p):
		"""Invalidate any cache data that depends on Vs[p].
This includes calculations that depends on steps (p-1) or (p).  Assumes 1 <= p <= N."""
		self.cache['weights2'].pop(p - 1, None)
		self.cache['weights2'].pop(p, None)
		self.cache['U_decomp'].pop(p - 1, None)
		self.cache['U_decomp'].pop(p, None)
		self.cache['fragile'] = {}


	##################################################
	##	Retrieval

	def Vfinal(self):
		return self.Vs[self.N]


	def U(self, s):
		"""Returns the unitary matrix U at step s, where 0 <= s < N."""
		return self.Vs[s+1] @ self.Vs[s].conj().T


	def jlogU(self, s):
		"""Returns -i times the log of the unitary matrix U at step s, where 0 <= s < N."""
		v, Z = self.U_decomp(s)
		return (Z * v[np.newaxis, :]) @ Z.conj().T


	def U_to_target(self, V=None):
		"""Return the unitary needed to reach Utarget from V.  If V is None, then use Vfinal."""
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
	def str(self, verbose=2):
		s = ""
		if verbose >= 1:
			for i in range(self.N):
				s += self.step_str(i, verbose=verbose) + "\n"
			s += "Final U:\n" + stringtools.joinstr([ "  ", zero_real_if_close(self.Vfinal()) ]) + "\n"
			s += "U to target:\t(weight2 = {})\n".format( self.weight2_to_target() )
			if verbose >= 2: s += stringtools.joinstr([ "  ", zero_real_if_close(self.U_to_target()) ]) + "\n"
			s += "Total weight1 = {:12.8f}\n".format( self.weight1_total() )
			s += "Total weight2 = {:12.8f}\n".format( self.weight_total() )
			s += "Total weight2 * {} = {:12.8f}\n".format( self.N, self.weight_total() * self.N )
		return s

	def step_str(self, s, verbose=2):
		"""Output a string detailing the s'th step.  (No linebreak at the end.)
Can be overloaded."""
		if verbose >= 1: outstr = "Step {:2d}:  (weight2 = {})".format( s, self.weight2_at_step(s) )
		if verbose >= 2: outstr += "\n" + stringtools.joinstr([ "  ", zero_real_if_close(self.jlogU(s)) ])
		return outstr


	##################################################
	##

	def load_U_to_V(self, U):
#TODO, use load_from_Ulist instead
		print("UnitaryChain.load_U_to_V() is deprecated, use load_from_Vlist instead.")
		self.subdivide_at_step(0, len(U))
		for i in range(1, len(U)+1):
			self.update_V_at_point(i, U[i-1] @ self.Vs[i-1])

	def load_from_Vlist(self, Vlist):
		"""Vlist is a list of d*d unitary matrices.  Their dimensions must match that of Utarget, Vlist[0] must be identity."""
		#TODO, test this code
		N = len(Vlist) - 1
		assert N >= 1
		d = self.d
		dtype = self.dtype
		for s in range(N):
			assert isinstance(Vlist[s], np.ndarray) and Vlist[s].shape == (d,d)
			dtype = np.promote_types(Vlist[s].dtype, dtype)
	##	load data
		self.N = N
		self.dtype = dtype
		self.Vs = [ Vlist[i].astype(dtype, copy=True) for i in range(N+1) ]
		self.reset_cache()
		self.check_consistency()

	def load_from_Ulist(self, Ulist, unitarize_input=False):
		"""Ulist is a list of d*d unitary matrices.  The dimensions must match that of Utarget."""
		N = len(Ulist)
		assert N >= 1
		d = self.d
		for s in range(N):
			assert isinstance(Ulist[s], np.ndarray) and Ulist[s].shape == (d,d)
	##	load data
		self.N = N
		self.Vs = [ np.eye(d) ] + [ None ] * N
		for s in range(N):
			if unitarize_input: raise NotImplementedError #TODO fix
			self.Vs[s+1] = Ulist[s] @ self.Vs[s]
		self.dtype = np.promote_types(self.Vs[N].dtype, self.dtype)
		for s in range(N):
			self.Vs[s] = self.Vs[s].astype(self.dtype, copy=False)
		self.reset_cache()
		self.check_consistency()


	def update_V_at_point(self, p, newV):
		"""Update Vs[p] to newV.
p is an integer between 1 <= p <= N.  This will alter steps p-1 and p."""
		assert isinstance(newV, np.ndarray) and newV.shape == (self.d, self.d)
		self.Vs[p] = newV.astype(self.dtype, copy=True)
		self.invalidate_cache_at_point(p)

	def apply_U_to_V_at_point(self, p, U):
		"""Update Vs[p] -> U Vs[p], where U is a d*d unitary matrix (assuming U is unitary).
p is an integer between 1 <= p <= N.  This will alter steps p-1 and p."""
		self.Vs[p] = U @ self.Vs[p]
		self.invalidate_cache_at_point(p)

	def apply_expiH_to_V_at_point(self, p, H):
		"""Update Vs[p] -> exp[iH] Vs[p], where H is a d*d Hermitian matrix.
p is an integer between 1 <= p <= N.  This will alter steps p-1 and p."""
		A = 0.5j * ( H + np.conj(np.transpose(H)) )
		self.Vs[p] = sp.linalg.expm(A) @ self.Vs[p]
		self.invalidate_cache_at_point(p)


	def unitarize_point(self, p):
		"""Make Vs[p] unitary (via unitarize)."""
		if p == 'all':
			for i in range(1, self.N+1): self.unitarize_point(i)
			return
		assert isinstance(p, (int, np.integer))
		assert p >= 1
		#old_unitarity = np.max(np.abs( self.Vs[p] @ self.Vs[p].conj().T - np.eye(self.d) ))
		self.Vs[p] = unitarize(self.Vs[p])
		#new_unitarity = np.max(np.abs( self.Vs[p] @ self.Vs[p].conj().T - np.eye(self.d) ))
		self.invalidate_cache_at_point(p)


	##################################################
	##	Tools to add/remove steps

	def subdivide_at_step(self, step, num_div):
		"""Evenly subdivide the unitary at step (step) into num_div pieces.
The resulting UnitaryChain has (num_div-1) extra steps."""
		assert isinstance(step, (int, np.integer))
		assert isinstance(num_div, (int, np.integer))
		N = self.N
		num_div = int(num_div)
		assert num_div > 0
		assert 0 <= step and step < N
		if num_div == 1: return		# nothing to do
		Vs = self.Vs
		Vstart = Vs[step]
	##	Diagonalize matrix:  Ustep = UZ . diag(exp(i v)) . UZdag
		Uv, UZ = self.U_decomp(step)
		UZdag_Vstart = UZ.conj().T @ Vstart
	##	reset cache
		old_cache_U_decomp = self.cache['U_decomp']
		self.reset_cache()
	##	compute subdivisions: step -> [step: step+num_div]
		Vs_insert = []
		for i in range(1, num_div):
		##	The i^th term is  Ustep^(i/n) @ Vstart  =  UZ @ diag(1j * Uv^(i/n)) @ UZdag @ Vstart
			D = np.diag(np.exp(1j * Uv * i / num_div))
			Vs_insert.append(UZ @ D @ UZdag_Vstart)
	##	Add extra matrices
		self.Vs = Vs[:step+1] + Vs_insert + Vs[step+1:]
		self.N += num_div - 1
	##	Rebuild (partially) cache, only keep 'U_decomp' (throw away 'weights2')
		for i in range(num_div):
			self.cache['U_decomp'][step + i] = (Uv / num_div, UZ)
		for s in range(N):		# N is the old N
			if s in old_cache_U_decomp:
				if s < step: self.cache['U_decomp'][s] = old_cache_U_decomp[s]
				if s > step: self.cache['U_decomp'][s + num_div - 1] = old_cache_U_decomp[s]
		self.check_consistency()


	def subdivide_every_step(self, num_div):
		"""Evenly subdivide every unitary step into num_div pieces.  If num_div is a len-N array, then subdivide step s into num_div[s] pieces."""
		N = self.N
		try:
			iter(num_div)
			num_divs = num_div
		except TypeError:
			assert isinstance(num_div, (int, np.integer))
			assert num_div >= 1
			num_divs = np.ones(N, dtype=int) * num_div
		assert len(num_divs) == N
		for s in range(N):
			assert num_divs[s] >= 1
		for s in range(N - 1, -1, -1):
			self.subdivide_at_step(s, num_divs[s])


	def del_Vs(self, p):
		"""Delete Vs[p] from the list, where 0 < p <= N.
If p < N, then this combines steps (p-1), p into one step.  If p == N, then this removes the final step and makes Vs[N-1] the new Vfinal."""
		N = self.N
		if N == 1: raise RuntimeError("Can't have less than one step.")
		assert isinstance(p, (int, np.integer))
		assert 0 < p and p <= N
		del self.Vs[p]
		self.N = N - 1
		self.reset_cache()		#TODO, update cache more nuanced way
		self.check_consistency()


	def backup_Vs(self):
		self.backupVs = [ self.Vs[i].copy() for i in range(self.N+1) ]
		print("backup_Vs deprecated")
		raise RuntimeError

	def restore_from_backup_Vs(self):
		self.Vs = [ self.backupVs[i].copy() for i in range(self.N+1) ]
		self.reset_cache()
		print("restore_from_backup_Vs deprecated")
		raise RuntimeError


	##################################################
	##	Tools to manipulate unitaries, compute weights and derivatives

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
		#j_log_U = Z @ np.diag(v) @ Z.conj().T
		#print(np.max(np.abs( Z @ np.diag(np.exp(1j * v)) @ np.conj(np.transpose(Z)) - U )))
		self.cache['U_decomp'][s] = (v, Z)
		return v, Z


	def d_jlogU_before(self, s):
		"""Determines the 1st order change of (-i)logU (at step s) from altering Vs[s]."""
		#TODO
		raise NotImplementedError


	def d_jlogU_after(self, s):
		"""Determines the 1st order change of (-i)logU (at step s) from altering Vs[s+1]."""
		#TODO
		raise NotImplementedError


	## TODO:
	##	apply_expiHlist_to_Vs
	##	make something to interface with scipy.optimize.minimize


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
Optional attributes:
	U2t_nullweight_proj:

Also the class will have to overload _deepcopy_to_c() to make sure these attributes are copied.

Formula:
	MxComps_i = 2 / pi * Tr[ ConjMxComp_list[i] . (-i)logU ]
	weight2(Ustep) = sum_i MxComp_weights2_i MxComps_i^2
	weight2(U2target) = ODpenalty + ...
"""

	def check_consistency(self, tol=1e-13):
		output = super().check_consistency(tol=tol)
		d = self.d
		N = self.N
		MxComp_list = self.MxComp_list
		ConjMxComp_list = self.ConjMxComp_list
		MxComp_weights2 = self.MxComp_weights2
		U2t_DiagTests = self.U2t_DiagTests
		nMx = d**2
		def compareMx(M1, M2, cmptxt=None):
			maxdiff = np.max(np.abs( M1 - M2 ))
			if maxdiff > tol: print('UnitaryChain_MxCompWeight.check_consistency(tol = {}) failed{}'.format( tol, " for '"+str(cmptxt)+"'" if cmptxt is not None else '' ))
			return maxdiff
	##	check MxComp_list and ConjMxComp_list
		assert isinstance(MxComp_list, np.ndarray) and MxComp_list.shape == (nMx, d, d)
		assert isinstance(ConjMxComp_list, np.ndarray) and MxComp_list.shape == (nMx, d, d)
		output['ConjMxCom Herm'] = compareMx( ConjMxComp_list.transpose(0,2,1).conj() , ConjMxComp_list, 'ConjMxCom Herm' )
		output['MxComp compat'] = compareMx( np.dot( MxComp_list.reshape(nMx, d**2) , ConjMxComp_list.reshape(nMx, d**2).conj().T ) , np.eye(nMx), 'MxComp compat' )
	##	check MxComp_weights2
		assert isinstance(MxComp_weights2, np.ndarray) and MxComp_weights2.shape == (nMx,)
		assert MxComp_weights2.dtype == float
		assert np.all(MxComp_weights2 >= 0)
	##	check U2t_DiagTests
		assert isinstance(U2t_DiagTests, list)
		for chk in U2t_DiagTests:
			assert isinstance(chk, tuple) and len(chk) == 4
			for ii in range(4): assert 0 <= chk[ii] and chk[ii] < d
	##
		output['U2t_0w_proj'] = -1
		if hasattr(self, 'U2t_nullweight_proj'):
			U2t_proj = self.U2t_nullweight_proj
			assert isinstance(U2t_proj, np.ndarray) and U2t_proj.shape == (d, d)
			output['U2t_0w_proj'] = compareMx( U2t_proj @ U2t_proj , U2t_proj, 'U2t_nullweight_proj err' )
			#TODO check compat with U2t_DiagTests
	##	check cache
	#TODO, loop over keys
		if 'grad_w2 U2t0' in self.cache['fragile']:
			gradHv = self.cache['fragile']['grad_w2 U2t0']
			assert isinstance(gradHv, np.ndarray) and gradHv.shape == (N * d **2,)
			print('chk: grad_w2 U2t0')
		if 'grad_w2' in self.cache['fragile']:
			gradHv = self.cache['fragile']['grad_w2']
			assert isinstance(gradHv, np.ndarray) and gradHv.shape == (N * d **2,)
			print('chk: grad_w2')
	##
		output['err'] = max( output['ConjMxCom Herm'], output['MxComp compat'], output['U2t_0w_proj'], 0, output['err'] )
		if type(tol) == float and output['err'] > tol:
			raise ArithmeticError("UnitaryChain_MxCompWeight.check_consistency:  {} > tol ({})".format( output['err'], tol ))
		return output


	def compute_weight2_at_step(self, s):
#TODO document
		jlogU = self.jlogU(s)		# expects a Hermitian matrix
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
		##	we penalize off-diagonal terms and how far the diagonal are pure phases, computing:
		##		dist2 += Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1)) + np.sum(1 - np.abs(D)**2)
		## since frobnorm(U) = d, the two terms are actually equal
		##	however, the latter term may be negative due to numerical errors, so we use the only former term
		dist2 += 2 * Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1))
		#print("OD weight =", dist2); return dist2
		for chk in self.U2t_DiagTests:
			dist2 += np.abs( D[chk[0]] * D[chk[3]] - D[chk[1]] * D[chk[2]] )**2
		return self.coef['penalty']**2 * dist2


	def _Vfinal_remove_OD(self):
		"""Alter Vfinal such that it is diagonal (and unitary).
This does not necessarily make weight_to_target vanish, since it doesn't enforce the U2t_DiagTests."""
		N = self.N
		U2t = self.U_to_target()
		D = np.diag(U2t)
		if np.any(np.abs(D) < 0.1): print("UnitaryChain_MxCompWeight.project_Vfinal():  Unreliable projection!")
		self.Vs[N] = (D.conj() / np.abs(D))[:, np.newaxis] * self.Utarget
		self.invalidate_cache_at_point(N)


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
		#jlogU = Z @ np.diag(v) @ Z.conj().T		# is this needed?
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
		return grH.conj() * self.coef['penalty']**2


	def compute_grad_weight2(self, enforce_U2t_0weight=False, output_form='MxList'):
		"""Compute the gradient of total weight2 with respect to H[s]* (applied to Vs[s])
Specifically:  d weight2_total( exp[i H[1]) . Vs[1] , ..., exp[i H[N]) . Vs[N] ) / d H[s]_{i,j}*

Returns gradH, a list (length N+1), such that gradH[s] is a d*d Hermitian matrix for 1 <= s <= N.
"""
#TODO, explain enforce_U2t_0weight
		assert output_form == 'MxList' or output_form == 'vec'
		d = self.d
		N = self.N
		gradH = [ None ] * (N + 1)
		gradHv = None		# to be set later
	##	Check cache
		if enforce_U2t_0weight:
			if 'grad_w2 U2t0' in self.cache['fragile']: gradHv = self.cache['fragile']['grad_w2 U2t0']
		else:
			if 'grad_w2' in self.cache['fragile']: gradHv = self.cache['fragile']['grad_w2']
		if gradHv is not None:		# the cache found something!
			if output_form == 'MxList': return [ None ] + [ gradHv.reshape(N,d,d)[i] for i in range(N) ]
			if output_form == 'vec': return gradHv
	##	Collate gradient data
		for s in range(N):
			grHL, grHR = self.compute_grad_weight2_at_step(s)
			if s > 0: gradH[s] += grHR
			gradH[s + 1] = grHL
		if enforce_U2t_0weight:
		##	assume weight2_to_target is zero, and force gradH[n] to maintain the zero weight
			HNdiag = np.diag(gradH[N])
			try:
				U2t_proj = self.U2t_nullweight_proj
				HNdiag = np.dot(U2t_proj, HNdiag)
			except AttributeError: pass
			gradH[N] = np.diag(HNdiag)
		else:
			U2t = self.U_to_target()
			gradH[N] += self.compute_grad_weight2_to_target(U2t)
	##	Vectorize
		gradHv = np.array(gradH[1:]).reshape(-1)
		gradHv.flags.writeable = False		# small safeguard against accidentally corrupting cache
	## Cache and return
		if enforce_U2t_0weight: self.cache['fragile']['grad_w2 U2t0'] = gradHv
		else: self.cache['fragile']['grad_w2'] = gradHv
		if output_form == 'vec': return gradHv
		else: return gradH


	def apply_random_small_phases_to_Vfinal(self, RNG=None, sigma=0):
		"""Modify Vfinal (and hence the last U step and U_to_target) such that weight_to_final remains unchanged."""
		N = self.N
		small_ph = RNG.normal(scale=sigma, size=(self.d,))
		if hasattr(self, 'U2t_nullweight_proj'): small_ph = np.dot(self.U2t_nullweight_proj, small_ph)
		self.Vs[N] = np.exp(1j * small_ph)[:, np.newaxis] * self.Vs[N]		# left multiply by diagonal matrix
		self.invalidate_cache_at_point(N)
		self.check_consistency()		# optional




################################################################################
class qubit_UChain(UnitaryChain_MxCompWeight):
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

	def __init__(self, Utarget):
		super().__init__(Utarget)
		assert self.d == 2
		self.coef = {'Rabi':1., 'penalty':np.sqrt(15)}
		self.set_coef()
	##	Set up weights
		qubit_UChain._set_up_MxComp_lists()
	##	Done!
		#self.check_consistency()		# optional


	def set_coef(self, Rabi=None, penalty=None):
		if Rabi is not None:
			assert type(Rabi) == float and 0 <= Rabi
			self.coef['Rabi'] = float(Rabi)
		if penalty is not None:
			assert type(penalty) == float and 0 <= penalty
			self.coef['penalty'] = float(penalty)
		R1 = self.coef['Rabi']**2; pe = self.coef['penalty']**2
		self.MxComp_weights2 = np.array([ pe, R1, R1, pe ])
		self.reset_weight2_cache()
		self.cache['fragile'] = {}		# reset cached gradients


	def _deepcopy_to_c(self, c):
		##	overloads (grand)parent class, this is a helper called by copy()
		super()._deepcopy_to_c(c)
		c.set_coef()


	@classmethod
	def _set_up_MxComp_lists(cls):
		if hasattr(qubit_UChain, 'ConjMxComp_list'): return
		qubit_UChain.PauliList = np.array([cls.I2, cls.PX, cls.PY, cls.PZ])
		qubit_UChain.PauliList.flags.writeable = False
		assert qubit_UChain.PauliList.shape == (4,2,2)
		qubit_UChain.MxComp_list = qubit_UChain.PauliList
		qubit_UChain.ConjMxComp_list = qubit_UChain.PauliList / 2
		qubit_UChain.ConjMxComp_list.flags.writeable = False
		qubit_UChain.U2t_DiagTests = []		# no constraints on the phases of U_to_target


	def force_weight2t_to_zero(self):
		"""Fixes Vs[N] such that weight2_to_target vanishes."""
		self._Vfinal_remove_OD()
		self.check_consistency()


## old stuff
	def _old_compute_weight2_at_step(self, s):
		logU = self.jlogU(s)
		weight = ( np.abs(logU[0,0]**2) + np.abs(logU[1,1]**2) ) * self.coef['penalty']**2
		weight += np.abs(logU[0,1]**2) * self.coef['Rabi']**2
		return weight


	def _old_compute_weight2_to_target(self, U2t):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate."""
		weight = Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1))	# off diagonal term
		weight += Frob_norm(np.abs(np.diag(U2t)) - 1)		# measures how far the diagonal terms are to pure phases
		return self.coef['penalty']**2 * weight



################################################################################
# TODO, rename to two_qubits_UChain
class two_qubits_unitary(UnitaryChain_MxCompWeight):
	"""Specialize to 2 qubits.

Coefficients:
	Rabi1: the weight given to an single qubit X/Y drives (assigns Rabi1 to half Rabi period)
	Rabi2: the weight given to pair drives (assigns2 Rabi2 to half Rabi period of conversion or gain)
	penalty: the weight given to other drives

From (-i)log(U),
	Rabi1 applies to IX, IY, XI, YI components;
	Rabi2 applies to X/Y between |00>,|11>, and also X/Y between |01>,|10>;
	penalty applies to others (Paulis II, ?Z, Z?).
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
		self.coef = {'Rabi1':0.1, 'Rabi2':1, 'penalty':5.}
		self.set_coef()
	##	Set up weights
		two_qubits_unitary._set_up_2Q_static_data()
	##	Done!
		#self.check_consistency()		# optional


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
		self.reset_weight2_cache()
		self.cache['fragile'] = {}		# reset cached gradients


	def _deepcopy_to_c(self, c):
		##	overloads (grand)parent class, this is a helper called by copy()
		super()._deepcopy_to_c(c)
		c.set_coef()		# call set_coef again after self.coef has been updated


	@classmethod
	def _set_up_2Q_static_data(cls):
		"""This function sets up a bunch static data associated with the two_qubits_unitary class."""
		if hasattr(two_qubits_unitary, 'P2list') and hasattr(two_qubits_unitary, 'U2t_DiagTests'): return
		##	Pauli list:  11, 1X, 1Y, 1Z, X1, XX, XY, ..., ZY, ZZ
		two_qubits_unitary.P2list = np.array([ np.kron(P1,P2) for P1 in two_qubits_unitary.PauliList for P2 in two_qubits_unitary.PauliList ])
		two_qubits_unitary.P2list.flags.writeable = False
		assert two_qubits_unitary.P2list.shape == (16,4,4)
		two_qubits_unitary.MxComp_list = two_qubits_unitary.P2list
		two_qubits_unitary.ConjMxComp_list = two_qubits_unitary.P2list / 4
		two_qubits_unitary.ConjMxComp_list.flags.writeable = False
		##	for reference: MxComp_weights2 = [ pe, R1, R1, pe, R1, 2*R2, 2*R2, pe, R1, 2*R2, 2*R2, pe, pe, pe, pe, pe ]
		two_qubits_unitary.R1_comps = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
		two_qubits_unitary.R2_comps = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
		two_qubits_unitary.pe_comps = np.ones(16) - two_qubits_unitary.R1_comps - two_qubits_unitary.R2_comps
		two_qubits_unitary.U2t_nullweight_proj = np.array([[3,1,1,-1],[1,3,-1,1],[1,-1,3,1],[-1,1,1,3]], dtype=float) / 4
		two_qubits_unitary.U2t_nullweight_proj.flags.writeable = False
		two_qubits_unitary.U2t_DiagTests = [ (0,1,2,3), ]		# used to check how close the diagonal is to a Kronecker product


#	def check_consistency(self, tol=1e-13):
#		output = super().check_consistency(tol=tol)
#		assert self.d == 4
#		return output
#	#TODO check_consistency to check for coef


	def decomp_jlogU(self, jlogU):
		"""Decompose jlogU (a matrix) into its Rabi1, Rabi2, and penalty components.

Returns a tuple (weight2_decomp, jlogU_decomp).

weight2_decomp is a 3-array, jlogU_decomp is a list (len 3) of matrices.
The elements corresponds to, respectively, 1-body Rabi terms, 2-body Rabi terms, and penalty terms.
Their sum, i.e. np.sum(weight2_decomp) and jlogU_decomp[0] + jlogU_decomp[1] + jlogU_decomp[2], gives weight2(jlogU) and jlogU, respectively.
"""
		jlogUT = jlogU.conj()
		comps = [ two_qubits_unitary.R1_comps, two_qubits_unitary.R2_comps, two_qubits_unitary.pe_comps ]
		jlogU_decomp = [ None ] * 3
		weight2_decomp = np.zeros(3, dtype=float)
		for c in range(3):
			MxComps = (2/np.pi) * np.array([ comps[c][i] * np.sum(self.ConjMxComp_list[i] * jlogUT) for i in range(16) ]).real
			jlogU_decomp[c] = np.tensordot(MxComps, self.MxComp_list, axes=[[0],[0]]) / 2
			weight2_decomp[c] = np.sum(MxComps**2 * self.MxComp_weights2)
		return weight2_decomp, jlogU_decomp


	def step_str(self, s, verbose=3):
		outstr = ""
		if verbose < 1: return ""
		jlogU = self.jlogU(s)
		weight2_decomp, jlogU_decomp = self.decomp_jlogU(jlogU)
		outstr += "Step {:2d}:  \t(weights2 = {} = {})".format(s, self.weight2_at_step(s), " + ".join(map(str, weight2_decomp)))
		if verbose >= 2: outstr += "\n" + stringtools.joinstr([ "  ", zero_real_if_close(jlogU) ])
		if verbose >= 3:
			outstr += "\n" + "\n".join([ stringtools.joinstr([ "   = pi * " if c == 0 else "   + pi * ", zero_real_if_close(jlogU_decomp[c]) ]) for c in range(3) ]) + "\n"
#		jlogUT = jlogU.conj()
#		##	MxComp_weights2 = [ pe, R1, R1, pe, R1, 2*R2, 2*R2, pe, R1, 2*R2, 2*R2, pe, pe, pe, pe, pe ]
#		R1_comps = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
#		R2_comps = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
#		pe_comps = np.ones(16) - R1_comps - R2_comps
#		comps = [ R1_comps, R2_comps, pe_comps ]
#		wsum_str = "weight2 = {} = ".format( self.weight2_at_step(s) )
#		mxsum_str = stringtools.joinstr([ "  ", zero_real_if_close(jlogU) ])
#		for c in range(3):
#			MxComps = (2/np.pi) * np.array([ comps[c][i] * np.sum(self.ConjMxComp_list[i] * jlogUT) for i in range(16) ]).real
#			M = np.tensordot(MxComps, self.MxComp_list, axes=[[0],[0]]) / 2
#			if c > 0: wsum_str += " + "
#			wsum_str += str(np.sum(MxComps**2 * self.MxComp_weights2))
#			if verbose >= 3: mxsum_str += "\n" + stringtools.joinstr([ "   + pi * " if c > 0 else "   = pi * ", zero_real_if_close(M) ])
#		if verbose >= 3: mxsum_str += "\n"
#		outstr += "\n[old] (" + wsum_str + ")"
#		if verbose >= 2: outstr += "\n" + mxsum_str
		return outstr


	def force_weight2t_to_zero(self):
		"""Fixes Vs[N] such that weight2_to_target vanishes."""
		U2t = self.U_to_target()
		newD = rank1tensor_approx(np.diag(U2t).reshape(2,2)).reshape(4)
		self.Vs[self.N] = (newD.conj() / np.abs(newD))[:, np.newaxis] * self.Utarget
		self.invalidate_cache_at_point(self.N)
		self.check_consistency()


	##	compute_weight2_at_step() and compute_weight2_to_target() derived from UnitaryChain_MxCompWeight


	#TODO remove, old code
	def _old_compute_weight2_to_target(self, U2t):
		"""Provides the weight of U_to_target.  This function measures how far U_to_target is to a phase gate on either qubit."""
		weight = Frob_norm(np.triu(U2t, k=1) + np.tril(U2t, k=-1))	# off diagonal term
		Dg = np.diag(U2t)		# diagonal part of U2t
		weight += Frob_norm(np.abs(Dg) - 1)		# measures how far the diagonal terms are to pure phases
		weight += np.abs( Dg[0] * Dg[3] - Dg[1] * Dg[2] )**2		# check how close the diagonal is to a Kronecker product
		return self.coef['penalty'] * weight

	def _old_compute_weight2_at_step(self, s):
	## superseded by UnitaryChain_MxCompWeight.compute_weight2_at_step()
		logU = self.jlogU(s)		# returns a Hermitian matrix
		logUT = logU.transpose()
		##	Pauli components: Pcomp[i] = tr(P2[i] . logU) / 2pi, or logU = (pi/2) sum_i Pcomp[i] P2[i]
		MxComps = np.array([ np.sum(P * logUT) for P in two_qubits_unitary.P2list ]).real / (2 * np.pi)
		return np.sum(MxComps**2 * self.MxComp_weights2)



################################################################################
class three_qubits_UChain(UnitaryChain_MxCompWeight):
	pass
	## TODO

#	three_qubits_UChain.U2t_DiagTests = [ (0,1,2,3), (0,1,4,5), (0,2,4,6), (1,3,5,7), (2,3,6,7), (4,5,6,7), ]
##	there are 12 possible quadruplets:
##		from (0,3)=(1,2) , (0,5)=(1,4) , (0,6)=(2,4) , (0,7)=(1,6)=(2,5)=(3,4) , (1,7)=(3,5) , (2,7)=(3,6) , (4,7)=(5,6)
##		(0,1,2,3) (0,1,4,5) (0,1,6,7) (0,2,4,6) (0,2,5,7) (0,3,4,7) (1,2,5,6) (1,3,4,6) (1,3,5,7) (2,3,4,5) (2,3,6,7) (4,5,6,7)
##	there are, for example, not quadruplets: (0,3,5,6) (1,2,4,7)




################################################################################
