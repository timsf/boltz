from typing import Iterator

import numpy as np
import numpy.typing as npt

from boltz import samplers


BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def optimizer(
    v: BoolArr,
    widths: tuple[int],
    n_gibbs: int,
    n_walkers: int,
    n_subsamples: int,
    step: float = 1,
    mom: float = 0,
    pen: float = 0,
    rng: np.random.Generator = np.random.default_rng(),
) -> Iterator[tuple[list[FloatArr], list[FloatArr]]]:

    w = [rng.standard_normal((i0, i1)) for i0, i1 in zip((v.shape[1],) + widths, widths)]
    b = [rng.standard_normal(i) for i in (v.shape[1],) + widths]
    h = [np.ones((v.shape[0], i)) / 2 for i in widths]
    x = [[rng.uniform(size=(1, i)) < .5 for i in (v.shape[1],) + widths] for _ in range(n_walkers)]
    vb = [np.zeros_like(b_) for b_ in b]
    vw = [np.zeros_like(w_) for w_ in w]
    while True:
        o0 = eval_offset(v, h, x)
        db, dw, h, x = est_grad(n_gibbs, n_subsamples, v, h, x, o0, b, w, rng)
        vb = [mom * vb_ + step * (db_ - b_) for vb_, db_, b_ in zip(vb, db, b)]
        vw = [mom * vw_ + step * (dw_ - pen * w_) for vw_, dw_, w_ in zip(vw, dw, w)]
        b = [b_ + vb_ for b_, vb_ in zip(b, vb)]
        w = [w_ + vw_ for w_, vw_ in zip(w, vw)]
        o1 = eval_offset(v, h, x)
        b = transform(b, w, o0, o1)
        bstd = transform(b, w, o1, [np.zeros_like(o_) for o_ in o1])
        yield bstd, w


def est_grad(
    n_iters: int,
    n_subsamples: int,
    v: BoolArr,
    h_nil: list[FloatArr],
    x_nil: list[list[BoolArr]],
    o: list[FloatArr],
    b_nil: list[FloatArr],
    w_nil: list[FloatArr],
    rng: np.random.Generator,
) -> tuple[list[FloatArr], list[FloatArr], list[list[BoolArr]]]:
    
    h_prime = [h_.copy() for h_ in h_nil]
    sel = np.sort(rng.choice(v.shape[0], size=n_subsamples, replace=False))
    db_pos, dw_pos, h_new = wake(n_iters, v[sel], [h_[sel] for h_ in h_prime], o, b_nil, w_nil)
    db_neg, dw_neg, x_prime = dream(n_iters, x_nil, o, b_nil, w_nil, rng)
    db = [db_pos_ / n_subsamples - db_nex_ for db_pos_, db_nex_ in zip(db_pos, db_neg)]
    dw = [dw_pos_ / n_subsamples - dw_nex_ for dw_pos_, dw_nex_ in zip(dw_pos, dw_neg)]
    for i, h_ in enumerate(h_new):
        h_prime[i][sel] = h_
    return db, dw, h_prime, x_prime


def wake(
    n_iters: int,
    v: BoolArr | None,
    h_nil: list[FloatArr],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
) -> tuple[list[FloatArr], list[FloatArr], list[FloatArr]]:

    h_prime = h_nil
    for _ in range(n_iters):
        h_prime = samplers.contract_kernel([v] + h_prime, [False] + len(h_nil) * [True], o, b, w)[1:]
    db, dw = samplers.eval_gradient([v] + h_prime, o, b, w)
    return db, dw, h_prime


def dream(
    n_iters: int,
    x_nil: list[list[BoolArr]],
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
    rng: np.random.Generator,
) -> tuple[list[FloatArr], list[FloatArr], list[BoolArr]]:
    
    x_prime = [samplers.initialize(n_iters, x_, len(b) * [True], o, b, w, rng) for x_ in x_nil]
    x_sampler = samplers.sample_ensemble(x_prime, len(b) * [True], o, b, w, rng)
    x = [next(x_sampler) for _ in range(n_iters)]
    grads = [samplers.eval_gradient(x_, o, b, w) for x_ in sum(x, [])]
    db = [np.mean(db_, 0) for db_ in zip(*(x[0] for x in grads))]
    dw = [np.mean(dw_, 0) for dw_ in zip(*(x[1] for x in grads))]
    return db, dw, x[-1]


def eval_offset(
    v: BoolArr,
    h: list[FloatArr],
    x: list[list[BoolArr]],
) -> list[FloatArr]:
    
    o = [(np.mean(h_, 0) + np.mean(x_, 0)) / 2 for h_, x_ in zip([v] + h, [np.vstack(x_) for x_ in zip(*x)])]
    return o


def transform(
    b_nil: list[FloatArr],
    w_nil: list[FloatArr],
    o_nil: list[FloatArr],
    o_prime: list[FloatArr],
) -> list[FloatArr]:
    
    b_prime = [b_.copy() for b_ in b_nil]
    for i in range(len(b_nil)):
        if i == 0:
            b_prime[i] = b_nil[i] + (o_prime[i+1] - o_nil[i+1]) @ w_nil[i].T
        elif i == len(b_nil) - 1:
            b_prime[i] = b_nil[i] + (o_prime[i-1] - o_nil[i-1]) @ w_nil[i-1]
        else:
            b_prime[i] = b_nil[i] + (o_prime[i+1] - o_nil[i+1]) @ w_nil[i].T + (o_prime[i-1] - o_nil[i-1]) @ w_nil[i-1]
    return b_prime
