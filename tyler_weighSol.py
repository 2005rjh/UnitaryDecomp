import numpy as np
import scipy as sp

def testSol(U):
    weights = []
    for s in range(len(U)):
        weights.append(compute_weight2_at_step(U, s))
    return weights, np.sum(weights)


def compute_weight2_at_step(U, s):
    R1 = 0.1
    R2 = 1
    pe = 10
    
    I2 = np.eye(2, dtype=float)
    PX = np.array([[0,1.],[1.,0]])
    PY = np.array([[0,-1j],[1j,0]])
    PZ = np.array([[1.,0],[0,-1.]])
    PauliList = [I2,PX,PY,PZ]
    P2list = np.array([ np.kron(P1,P2) for P1 in PauliList for P2 in PauliList ])
    T, Z = sp.linalg.schur(U[s], output='complex')
    v = np.angle(np.diag(T))
    logU = Z @ np.diag(v) @ Z.conj().T	# returns a Hermitian matrix
    logUT = logU.transpose()
    ##	Pauli components: Pcomp[i] = tr(P2[i] . logU) / 2pi, or logU = (pi/2) sum_i Pcomp[i] P2[i]
    MxComps = np.array([ np.sum(P * logUT) for P in P2list ]).real / (2 * np.pi)
    return np.sum(MxComps**2 * np.array([ pe, R1, R1, pe, R1, 2*R2, 2*R2, pe, R1, 2*R2, 2*R2, pe, pe, pe, pe, pe ]))        