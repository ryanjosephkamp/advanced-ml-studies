"""Learning rules, recall dynamics, and pattern generators for the capacity study.

All patterns are +/-1 vectors of length N = 144 (12 x 12). Each rule returns a weight matrix W
and a per-neuron bias b; a neuron is set to +1 when (W s)_i + b_i >= 0 and to -1 otherwise.

Rules
-----
hebb      Hebbian outer-product rule, standard +/-1 update.
hebb01    Hebbian weights with the assignment's update, which takes the dot product of a weight
          column with the 0/1 state. In +/-1 terms that is the standard field plus the column sum
          of W, so it is the standard update with bias b = W.sum(axis=0).
storkey   Storkey (1997), with the local field h_ij that excludes neurons i and j.
pinv      Pseudoinverse (projection) rule, W = Q (Q^T Q)^+ Q^T with a zero diagonal.
cov       Covariance rule for biased patterns (Amit, Gutfreund, and Sompolinsky, 1987):
          W = sum_mu (xi - a)(xi - a)^T with a the mean of all stored bits, the field computed
          on the centered state, sum_j W_ij (s_j - a), and a threshold midway between the
          expected fields of +1 and -1 bits. Identical to hebb when a = 0.
modern    Modern Hopfield network (Ramsauer et al., 2021) on unit vectors: x <- X^T softmax(beta X x),
          renormalized, output sign(x).
nn        Nearest stored pattern by overlap: the beta -> infinity limit of the modern network.
"""
import numpy as np

N = 144
MAX_SWEEPS = 50


# ----------------------------------------------------------------------------------------------
# Pattern families
# ----------------------------------------------------------------------------------------------

def prototype_family(n_patterns, rho, rng):
    """Patterns that share a random balanced prototype.

    Each pattern flips each prototype bit independently with probability p = (1 - sqrt(rho)) / 2,
    so the expected overlap between two patterns, (1/N) xi^mu . xi^nu, is (1 - 2p)^2 = rho, and the
    expected density of +1 bits stays at 0.5.
    """
    p = (1.0 - np.sqrt(rho)) / 2.0
    proto = rng.choice(np.array([-1, 1], dtype=np.int8), N)
    flips = rng.random((n_patterns, N)) < p
    return np.where(flips, -proto, proto).astype(np.int8)


def biased_family(n_patterns, density, rng):
    """Independent bits that are +1 with probability `density`; expected overlap (2d - 1)^2."""
    return np.where(rng.random((n_patterns, N)) < density, 1, -1).astype(np.int8)


def pattern_stats(S):
    """Density of +1 bits, mean pairwise overlap, and minimum pairwise Hamming distance."""
    S = S.astype(np.float64)
    G = S @ S.T / N
    iu = np.triu_indices(len(S), 1)
    ham = (N - G[iu] * N) / 2
    return float((S > 0).mean()), float(G[iu].mean()), int(round(ham.min()))


# ----------------------------------------------------------------------------------------------
# Learning rules
# ----------------------------------------------------------------------------------------------

def hebb(S):
    S = S.astype(np.float64)
    W = S.T @ S                      # integer-valued, so the incremental field below stays exact
    np.fill_diagonal(W, 0.0)
    return W, np.zeros(N)


def hebb01(S):
    W, _ = hebb(S)
    return W, W.sum(axis=0)


def storkey(S):
    S = S.astype(np.float64)
    W = np.zeros((N, N))
    for xi in S:
        h = W @ xi                                  # h_i = sum_k w_ik xi_k (w_ii = 0)
        H = h[:, None] - W * xi[None, :]            # H[i, j] = h_ij = h_i - w_ij xi_j
        W += (np.outer(xi, xi) - xi[:, None] * H.T - H * xi[None, :]) / N
        np.fill_diagonal(W, 0.0)
    return W, np.zeros(N)


def storkey_snapshots(S, loads):
    """Storkey weights for the first P patterns, for each P in `loads` (the rule is incremental)."""
    S = S.astype(np.float64)
    W = np.zeros((N, N))
    out = {}
    for count, xi in enumerate(S, start=1):
        h = W @ xi
        H = h[:, None] - W * xi[None, :]
        W += (np.outer(xi, xi) - xi[:, None] * H.T - H * xi[None, :]) / N
        np.fill_diagonal(W, 0.0)
        if count in loads:
            out[count] = W.copy()
    return out


def pinv(S):
    Q = S.astype(np.float64).T
    W = Q @ np.linalg.pinv(Q.T @ Q) @ Q.T
    # A unit with projector diagonal 1 has zero weights in exact arithmetic; set them exactly to 0.
    unit = np.abs(1.0 - np.diag(W)) < 1e-8
    W[unit, :] = 0.0
    W[:, unit] = 0.0
    np.fill_diagonal(W, 0.0)
    return W, np.zeros(N)


def cov(S):
    S = S.astype(np.float64)
    a = S.mean()
    C = S - a
    W = C.T @ C
    np.fill_diagonal(W, 0.0)
    # Field on the centered state, sum_j W_ij (s_j - a), compared with the midpoint threshold
    # -a * c, where c = mean_mu sum_j (xi_j - a)^2: at a stored pattern the field is about
    # (xi_i - a) * c, so the midpoint between the +1 and -1 cases is -a * c.
    c = float((C ** 2).sum(axis=1).mean())
    return W, -a * W.sum(axis=1) + a * c


ITERATIVE_RULES = {"hebb": hebb, "hebb01": hebb01, "storkey": storkey, "pinv": pinv, "cov": cov}


# ----------------------------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------------------------

def is_fixed_point(W, b, xi):
    """True if one sweep from xi changes nothing, in any update order."""
    return bool(np.array_equal(np.where(W @ xi + b >= 0, 1, -1), xi))


def recall(W, b, s0, rng, max_sweeps=MAX_SWEEPS):
    """Asynchronous updates in a fresh random order each sweep, until a sweep changes nothing.

    The field is updated incrementally after each change and recomputed at the start of each
    sweep. Returns (final state, sweeps used, converged).
    """
    s = s0.astype(np.float64).copy()
    for sweep in range(1, max_sweeps + 1):
        h = W @ s + b
        changed = 0
        for j in rng.permutation(N).tolist():
            new = 1.0 if h[j] >= 0 else -1.0
            if new != s[j]:
                h += W[:, j] * (new - s[j])
                s[j] = new
                changed += 1
        if changed == 0:
            return s, sweep, True
    return s, max_sweeps, False


def modern_update(X, x, beta, max_iter=10):
    """X: stored patterns as unit rows; x: unit query. Returns the sign of the fixed point."""
    for _ in range(max_iter):
        a = beta * (X @ x)
        a = np.exp(a - a.max())
        y = X.T @ (a / a.sum())
        y /= np.linalg.norm(y) + 1e-12
        if np.allclose(y, x, atol=1e-10):
            x = y
            break
        x = y
    return np.where(x >= 0, 1.0, -1.0)


def nearest(S, q):
    return S[int(np.argmax(S.astype(np.float64) @ q))].astype(np.float64)
