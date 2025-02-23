from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import expit, log_expit

import boltz.tools.walkers


BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_ensemble(
    x0: list[BoolArr],
    u: list[bool],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
    rng: np.random.Generator = np.random.default_rng(),
) -> Iterator[list[BoolArr]]:

    eval_log_psi = lambda x_nil: eval_target(x_nil, o, b, w)
    eval_log_prop = lambda x_nil, x_prime: eval_kernel(x_nil, x_prime, u, o, b, w)
    sample_prop = lambda x_nil: sample_kernel(x_nil, u, o, b, w, rng)
    log_psi = np.array([eval_log_psi(s) for s in x0])
    log_q = np.array([[eval_log_prop(h1, h2) for h2 in x0] for h1 in x0])
    x = x0

    while True:
        x, log_psi, log_q = boltz.tools.walkers.update_telewalk(eval_log_psi, eval_log_prop, sample_prop, x, log_psi, log_q, rng)
        yield x


def sample_kernel(
    x_nil: list[BoolArr],
    u: list[bool],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
    rng: np.random.Generator,
) -> list[BoolArr]:

    x_prime = [x_.copy() for x_ in x_nil]
    for i in list(range(0, len(x_prime), 2)) + list(range(1, len(x_prime), 2)):
        if u[i]:
            if i == 0:
                linpred = b[i] + (x_prime[i+1] - o[i+1]) @ w[i].T
            elif i == len(x_prime) - 1:
                linpred = b[i] + (x_prime[i-1] - o[i-1]) @ w[i-1]
            else:
                linpred = b[i] + (x_prime[i+1] - o[i+1]) @ w[i].T + (x_prime[i-1] - o[i-1]) @ w[i-1]
            log_pon = log_expit(linpred)
            x_prime[i] = np.log(rng.uniform(size=x_prime[i].shape)) < log_pon
    return x_prime


def eval_kernel(
    x_nil: list[BoolArr],
    x_prime: list[BoolArr],
    u: list[bool],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
) -> float:

    x_mid = [x_.copy() for x_ in x_nil]
    log_prob = np.empty(len(x_mid))
    for i in list(range(0, len(x_mid), 2)) + list(range(1, len(x_mid), 2)):
        if u[i]:
            if i == 0:
                linpred = b[i] + (x_mid[i+1] - o[i+1]) @ w[i].T
            elif i == len(x_prime) - 1:
                linpred = b[i] + (x_mid[i-1] - o[i-1]) @ w[i-1]
            else:
                linpred = b[i] + (x_mid[i+1] - o[i+1]) @ w[i].T + (x_mid[i-1] - o[i-1]) @ w[i-1]
            log_pon = log_expit(linpred)
            log_prob[i] = np.sum(np.where(x_prime[i], log_pon, log_pon - linpred))
            x_mid[i] = x_prime[i]
    return np.sum(log_prob)


def contract_kernel(
    x_nil: list[FloatArr],
    u: list[bool],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
) -> list[FloatArr]:

    x_prime = [x_.copy() for x_ in x_nil]
    for i in list(range(0, len(x_prime), 2)) + list(range(1, len(x_prime), 2)):
        if u[i]:
            if i == 0:
                linpred = b[i] + (x_prime[i+1] - o[i+1]) @ w[i].T
            elif i == len(x_prime) - 1:
                linpred = b[i] + (x_prime[i-1] - o[i-1]) @ w[i-1]
            else:
                linpred = b[i] + (x_prime[i+1] - o[i+1]) @ w[i].T + (x_prime[i-1] - o[i-1]) @ w[i-1]
            x_prime[i] = expit(linpred)
    return x_prime


def initialize(
    n_iter: int,
    x_nil: list[BoolArr],
    u: list[bool],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
    rng: np.random.Generator,
) -> BoolArr:

    x_prime = x_nil
    for _ in range(n_iter):
        x_prime = sample_kernel(x_prime, u, o, b, w, rng)
    return x_prime


def eval_target(
    x: list[BoolArr],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
) -> float:

    log_prob = sum([np.sum((x[i] - o[i]) @ b[i]) for i in range(len(b))]) + sum([np.sum(((x[i] - o[i]) @ w[i]) * (x[i+1] - o[i+1])) for i in range(len(w))])
    return log_prob


def eval_gradient(
    x: list[BoolArr],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
) -> tuple[list[FloatArr], list[FloatArr]]:
    
    db = [np.sum(x[i] - o[i], 0) for i in range(len(b))]
    dw = [(x[i] - o[i]).T @ (x[i+1] - o[i+1]) for i in range(len(w))]
    return db, dw


def eval_elbo(
    v: BoolArr,
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
    n_iters: int
) -> float:
    
    h = [np.ones((v.shape[0], b_.shape[0])) / 2 for b_ in b[1:]]
    for _ in range(n_iters):
        h = contract_kernel([v] + h, [False] + len(h) * [True], o, b, w)[1:]
    return eval_target([v] + h, o, b, w) + np.sum([h_ * np.log(h_) + (1 - h_) * np.log(1 - h_) for h_ in h])
