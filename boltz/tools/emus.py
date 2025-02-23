import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

import boltz.tools.linalg


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def eval_vardi_estimator(
    log_psi: list[FloatArr],
    max_iter: int = 1,
    n_polish: int = 100,
) -> tuple[FloatArr, FloatArr]:

    def iter_vardi(old_z: FloatArr, n_iter: int) -> tuple[FloatArr, FloatArr]:
        f_est = est_overlap(log_psi, old_z / n_samples)
        new_z, new_z_rel, f_inv = solve_emus_system(f_est, old_z, n_polish)
        if n_iter == 1 or np.allclose(new_z_rel, n_samples / np.sum(n_samples)):
            return new_z, f_inv
        return iter_vardi(new_z, n_iter - 1)

    n_samples = np.array([log_psi_.shape[0] for log_psi_ in log_psi])
    return iter_vardi(n_samples / np.sum(n_samples), max_iter)


def power_iterate(f: FloatArr, z: FloatArr, n_iter: int) -> FloatArr:
    
    y = z
    for _ in range(n_iter):
        y = f.T @ y
    return y


def solve_emus_system(
    f_est: FloatArr,
    old_z: FloatArr,
    n_polish: int,
) -> tuple[FloatArr, FloatArr, FloatArr]:

    new_z_rel, f_inv = boltz.tools.linalg.eval_grpinv(f_est)
    new_z_rel = boltz.tools.linalg.power_iterate_stat(f_est, np.where(new_z_rel > 0, new_z_rel, 0) / np.sum(new_z_rel[new_z_rel > 0]), n_polish)
    f_inv = boltz.tools.linalg.power_iterate_grpinv(f_inv, f_est, new_z_rel, n_polish)
    new_z = new_z_rel * old_z / np.sum(new_z_rel * old_z)
    return new_z, new_z_rel, f_inv


def eval_vardi_estimator_alt(
    log_psi: list[FloatArr],
    max_iter: int = 100,
    eps: float = 1,
) -> tuple[FloatArr, FloatArr]:

    def iter_vardi(old_z: FloatArr, n_iter: int) -> tuple[FloatArr, FloatArr]:
        f_est = est_overlap(log_psi, old_z / n_samples)
        new_z_rel = eps * (np.sum(f_est, 0) - 1) + 1
        new_z_rel = np.where(new_z_rel > 0, new_z_rel, 0) / np.sum(new_z_rel[new_z_rel > 0])
        new_z = new_z_rel * old_z / np.sum(new_z_rel * old_z)
        if n_iter == 1 or np.allclose(new_z_rel, n_samples / np.sum(n_samples)):
            return new_z
        return iter_vardi(new_z, n_iter - 1)

    n_samples = np.array([log_psi_.shape[0] for log_psi_ in log_psi])
    return iter_vardi(n_samples / np.sum(n_samples), max_iter)


def est_overlap(log_psi: list[FloatArr], z_guess: FloatArr) -> FloatArr:

    log_r = [log_psi_ - np.log(z_guess[np.newaxis]) for log_psi_ in log_psi]
    log_f_est = np.array([logsumexp(log_r_ - logsumexp(log_r_, 1)[:, np.newaxis], 0) - np.log(log_r_.shape[0])
                          for log_r_ in log_r])
    return np.exp(log_f_est)


def est_overlap_var(log_psi: list[FloatArr]) -> list[FloatArr]:

    norm_log_psi = [log_psi_ - logsumexp(log_psi_, 1)[:, np.newaxis] for log_psi_ in log_psi]
    target = [np.ones(log_psi_.shape[1]) for log_psi_ in norm_log_psi]
    f_cov = [np.cov(np.exp(log_psi_.T)) if log_psi_.shape[0] > 1 else target for log_psi_ in norm_log_psi]
    return f_cov


def eval_importance_coefs(
    log_psi: list[FloatArr],
    z_est: FloatArr,
    inv_est_if: FloatArr,
) -> list[FloatArr]:

    f_cov = est_overlap_var(log_psi)
    z_jac = [z_est[i] * inv_est_if for i in range(len(log_psi))]
    coefs = [z_jac[i].T @ f_cov[i] @ z_jac[i] for i in range(len(log_psi))]
    return coefs


def eval_emus_cov(
    log_psi: list[FloatArr],
    z_est: FloatArr,
    inv_est_if: FloatArr,
) -> FloatArr:

    return sum([chi_sq_ / log_psi_.shape[0] for chi_sq_, log_psi_ in zip(eval_importance_coefs(log_psi, z_est, inv_est_if), log_psi)])


def eval_analytic_weights(chi_sq: list[FloatArr]) -> FloatArr:

    diag_chi_sq = np.array([np.diag(chi_sq[i]) for i in range(len(chi_sq))])
    score = np.sqrt(np.sum(diag_chi_sq, 1))
    return score / np.sum(score)


def extrapolate(
    log_psi: list[FloatArr], 
    log_tpsi: list[FloatArr], 
    zt_est: FloatArr
) -> FloatArr:

    log_g_est = np.array([logsumexp(log_psi_ - logsumexp(log_tpsi_, 1)[:, np.newaxis], 0) - np.log(log_psi_.shape[0])
                          for log_psi_, log_tpsi_ in zip(log_psi, log_tpsi)])
    g_est = np.exp(log_g_est)
    z_est = g_est.T @ zt_est
    return z_est / np.sum(z_est)


def extrapolate_signed(
    log_psi: list[FloatArr],
    sgn_psi:  list[FloatArr],
    log_tpsi: list[FloatArr],
    zt_est: FloatArr
) -> FloatArr:
    
    def sumexp(log_x, axis, sgn_x):
        log_y, sgn_y = logsumexp(log_x, axis, sgn_x, return_sign=True)
        return sgn_y * np.exp(log_y)
    g_est = np.array([sumexp(log_psi_ - logsumexp(log_tpsi_, 1)[:, np.newaxis], 0, sgn_psi_) / log_psi_.shape[0]
                      for log_psi_, log_tpsi_, sgn_psi_ in zip(log_psi, log_tpsi, sgn_psi)])
    z_est = g_est.T @ zt_est
    return z_est / np.sum(z_est)
