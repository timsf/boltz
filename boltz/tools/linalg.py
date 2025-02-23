import numpy as np
import numpy.typing as npt
import scipy.linalg


FloatArr = npt.NDArray[np.float64]


def lorank_decompose(a: FloatArr, n_nil_eig: int = 0) -> FloatArr:

    u, s, _ = np.linalg.svd(a, hermitian=True)
    rt_a = (u[:, :-n_nil_eig] * np.sqrt(s[:-n_nil_eig]))
    return rt_a


def eval_invdet(a: FloatArr, nugget: float = 1e-12) -> tuple[FloatArr, float]:

    reg = np.identity(a.shape[0])
    a = a + nugget * reg
    cf_a = np.linalg.cholesky(a)
    icf_a = scipy.linalg.solve_triangular(cf_a, np.identity(a.shape[0]), lower=True, check_finite=False)
    return icf_a.T @ icf_a, 2 * np.sum(np.log(np.diag(icf_a)))


def eval_pinvdet(a: FloatArr, rank: int, nugget: float = 1e-12) -> tuple[FloatArr, float]:

    reg = np.ones((len(a), len(a))) / len(a)
    a = a + nugget * reg
    u, s, _ = np.linalg.svd(a, hermitian=True)
    pinv_s = np.append(1 / s[:rank], np.zeros(len(s) - rank))
    pinv_a = (u * pinv_s) @ u.T
    plogdet_a = np.sum(np.log(s[:rank]))
    return pinv_a, -plogdet_a


def eval_grpinv(p: FloatArr) -> tuple[FloatArr, FloatArr]:

    a = np.identity(p.shape[0]) - p
    q, r = np.linalg.qr(a)
    u = r[:-1, :-1]
    pi = q[:, -1] / np.sum(q[:, -1])

    u2 = np.zeros(np.array(u.shape) + 1)
    u2[:-1, :-1] = np.linalg.inv(u)
    a2 = np.identity(len(pi)) - np.outer(np.ones(len(pi)), pi)

    return pi, a2 @ u2 @ q.T @ a2


def power_iterate_grpinv(f_inv0: FloatArr, f0: FloatArr, pi0: FloatArr, n_iter: int) -> FloatArr:

    if n_iter == 0:
        return f_inv0
    step = np.identity(len(pi0)) - np.outer(np.ones(len(pi0)), pi0)
    return power_iterate_grpinv(step @ f0 @ f_inv0 + step, f0, pi0, n_iter - 1)


def power_iterate_stat(f: FloatArr, pi: FloatArr, n_iter: int) -> FloatArr:

    if n_iter == 0:
        return pi / sum(pi)
    return power_iterate_stat(f, f.T @ pi, n_iter - 1)


def eval_quad(x: FloatArr, s: FloatArr) -> FloatArr:
    """Evaluate the quadratic form x[i].T @ inv(s) @ x[i] for each row i in x.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :returns: evaluated quadratic forms

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_quad(x, s)
    array([1876.35129871, 3804.08373042,  902.76990678])
    """

    l, _ = scipy.linalg.cho_factor(s, lower=True)
    root = scipy.linalg.solve_triangular(l, x.T, lower=True).T

    return np.sum(root ** 2, 1)


def eval_matquad(x: FloatArr, s: FloatArr) -> FloatArr:
    """Evaluate the quadratic form x @ inv(s) @ x.T.

    :param x:
    :param s: inverse scaling matrix. if 1-dimensional, assume diagonal matrix
    :returns: evaluated quadratic form
    :returns: evaluated quadratic forms

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_matquad(x, s)
    array([[ 1876.35129871,  2671.64559208, -1301.46391505],
           [ 2671.64559208,  3804.08373042, -1853.03472648],
           [-1301.46391505, -1853.03472648,   902.76990678]])
    """

    l, _ = scipy.linalg.cho_factor(s, lower=True)
    root = scipy.linalg.solve_triangular(l, x.T, lower=True).T

    return root @ root.T


def logdet_pd(s: FloatArr) -> float:
    """Evaluate log determinant of the PD matrix s

    :param s: PD matrix
    :returns: log determinant of s

    >>> np.random.seed(666)
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> logdet_pd(s)
    -1.854793046254502
    """

    r, _ = scipy.linalg.cho_factor(s)

    return float(np.sum(np.log(np.diag(r))) * 2)


def precond_solve_pd(a: FloatArr, b: FloatArr) -> FloatArr:
    """Solve the linear system a @ x = b in a more stable way by preconditioning a.

    :param a:
    :param b:
    :returns: solution of a @ x = b

    >>> np.random.seed(666)
    >>> a = np.diag(np.random.standard_normal(2) ** 2)
    >>> b = np.random.standard_normal(2) ** 2
    >>> x = precond_solve_pd(a, b)
    >>> a @ x, b
    (array([1.37702718, 0.82636839]), array([1.37702718, 0.82636839]))
    """

    precond = 1 / np.sqrt(np.diag(a))

    return (scipy.linalg.solve(((a * precond).T * precond).T, (b.T * precond).T, sym_pos=True).T * precond).T


def eval_detquad(x: FloatArr, s: FloatArr) -> tuple[FloatArr, float]:
    """

    :param x:
    :param s:
    :returns:

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_detquad(x, s)
    (array([[ 43.31388528,   0.50856729],
           [ 61.66973278,   0.96321845],
           [-30.04590564,  -0.11602223]]), -8.039424065916027)
    """

    if len(s.shape) == 2:
        l, _ = scipy.linalg.cho_factor(s, lower=True)
        root = scipy.linalg.solve_triangular(l, x.T, lower=True).T
        logdet_s = np.sum(np.log(np.diag(l))) * 2
    else:
        root = x / np.sqrt(s)
        logdet_s = np.sum(np.log(s))

    return root, float(logdet_s)


def eval_double_detquad(x: FloatArr, s: FloatArr, t: FloatArr) -> tuple[FloatArr, float, float]:
    """

    :param x:
    :param s:
    :param t:
    :returns:

    >>> np.random.seed(666)
    >>> x = np.random.standard_normal((3, 2))
    >>> s = np.diag(np.random.standard_normal(3) ** 2)
    >>> t = np.diag(np.random.standard_normal(2) ** 2)
    >>> eval_double_detquad(x, s, t)
    (array([[55.07567087, 41.42730829],
           [ 1.58103634,  1.5819772 ],
           [-1.13487599, -0.28074367]]), -8.930207969151354, -1.4727706473071034)
    """

    root, logdet_s = eval_detquad(x.T, s)
    root, logdet_t = eval_detquad(root.T, t)

    return root, logdet_s, logdet_t
