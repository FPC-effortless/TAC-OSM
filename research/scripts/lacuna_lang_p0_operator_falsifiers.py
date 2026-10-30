"""Independent constructed no-fit mathematical falsifiers for Lacuna-inspired TAC-OSM memory operators.

Scope: exact-real numerical examples only. Does NOT load trained TAC, publisher FIT/DEV,
optimizer/RNG, real language results or protected TEST. Does NOT certify neural source
IEEE/AD, no inference time or fair comparator claims. Research issue #633.

Run: python research/scripts/lacuna_lang_p0_operator_falsifiers.py
"""
import numpy as np

rng = np.random.default_rng(1030)

def delta(M, k, v, rho, beta):
    prior = rho * M
    return prior + beta * np.outer(v - prior @ k, k)

def erase_then_delta(M, e, k, v, rho, gamma, beta):
    prior = rho * M @ (np.eye(M.shape[1]) - gamma * np.outer(e, e))
    return prior + beta * np.outer(v - prior @ k, k)

def kalman(M, P, k, v, rho, sigma2, Q):
    prior = rho * M
    denom = sigma2 + k @ P @ k
    gain = (P @ k) / denom
    post = prior + np.outer(v - prior @ k, gain)
    Ppost = P - np.outer(P @ k, P @ k) / denom + Q
    return post, Ppost

def mobius(H, x):
    a, b, c, d = H.ravel()
    return (a*x + b) / (c*x + d)

def affine(A, b, x):
    return A @ x + b

def main():
    d = 8
    k = rng.normal(size=d); k /= np.linalg.norm(k)
    e = rng.normal(size=d); e /= np.linalg.norm(e)
    v = rng.normal(size=d); M = rng.normal(size=(d, d))
    rho, beta = 0.985, 0.6
    # 1: normalized-key delta interpolation identity.
    D = delta(M, k, v, rho, beta)
    np.testing.assert_allclose(D@k, (1-beta)*rho*(M@k)+beta*v, atol=1e-12)

    # 2: erasure reduces to delta at gamma=0; erase/write order is noncommutative.
    np.testing.assert_allclose(erase_then_delta(M,e,k,v,rho,0,beta),D,atol=1e-12)
    er=erase_then_delta(M,e,k,v,rho,0.9,beta)
    wrong=delta(M,k,v,rho,beta)@(np.eye(d)-0.9*np.outer(e,e))
    assert np.linalg.norm(er-wrong)>1e-6

    # 3: full positive-covariance update retains PSD posterior.
    B=rng.normal(size=(d,d)); P=B.T@B+0.1*np.eye(d)
    Q=0.001*np.eye(d)
    MK, PK=kalman(M,P,k,v,rho,0.2,Q)
    assert np.linalg.eigvalsh(PK).min()>0
    assert np.all(np.isfinite(MK)) and np.all(np.isfinite(PK))

    # 4: a diagonal covariance becomes dense after one full Kalman update.
    Pdiag=np.diag(np.diag(P))
    _, dense=kalman(M,Pdiag,k,v,rho,0.2,Q)
    assert np.linalg.norm(dense-np.diag(np.diag(dense)))>1e-6

    # 5: Möbius transition composition matches its sequential expression.
    H1=np.array([[0.9,0.1],[0.02,1.0]])
    H2=np.array([[0.95,0.12],[0.01,1.0]])
    x0=0.37
    np.testing.assert_allclose(mobius(H2,mobius(H1,x0)),mobius(H2@H1,x0),atol=1e-14)

    # 6: lower-triangular affine fast->slow recurrence admits pair scan.
    A=np.array([[.83,0.],[.2,.995]])
    b1=np.array([0.04,-0.01]); b2=np.array([-0.02,0.03])
    state=np.array([1.0,.5])
    np.testing.assert_allclose(affine(A,b2,affine(A,b1,state)),affine(A@A,A@b1+b2,state),atol=1e-14)

    # 7: a bounded exact-token ring must never exceed its declared slot budget.
    budget=12; slots=[]
    for i in range(10000):
        slots.append(i)
        if len(slots)>budget: slots.pop(0)
        assert len(slots)<=budget

    print("PASS: seven constructed operator/property checks; NO_FIT; NO_REAL_DATA; NO_REPLAY")

if __name__=="__main__":
    main()
