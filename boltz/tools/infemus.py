from typing import Callable, Iterator, TypeVar

import numpy as np
import numpy.typing as npt

import boltz.tools.emus


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
Hyper = TypeVar('H')
Params = TypeVar('P')


def est_mlik(
    lams: list[Hyper], 
    lame: list[Hyper], 
    instantiate_sampler: Callable[[Hyper], Iterator[Params]],
    eval_logprior: Callable[[Params, Hyper], float],
    rng: np.random.Generator, 
    n_samples: int, 
    n_burnin: int,
) -> FloatArr:

    log_psis, log_psie = resample(lams, lame, instantiate_sampler, eval_logprior, rng, n_samples, n_burnin)
    us = boltz.tools.emus.eval_vardi_estimator(log_psis)[0]
    ue = boltz.tools.emus.extrapolate(log_psie, log_psis, us)
    return ue / np.sum(ue)


def resample(
    lams: list[Hyper], 
    lame: list[Hyper], 
    instantiate_sampler: Callable[[Hyper, np.random.Generator], Iterator[Params]],
    eval_logprior: Callable[[Params, Hyper], float],
    rng: np.random.Generator, 
    n_samples: int, 
    n_burnin: int,
) -> FloatArr:

    samplers = [integrand_sampler(lams_, lams, lame, instantiate_sampler, eval_logprior, rng, n_burnin) for lams_ in lams]
    log_psi = [list(zip(range(n_samples), sampler)) for sampler in samplers]
    log_psis = [np.array([log_psis__ for _, (log_psis__, _) in log_psi_]) for log_psi_ in log_psi]
    log_psie = [np.array([log_psie__ for _, (_, log_psie__) in log_psi_]) for log_psi_ in log_psi]
    return log_psis, log_psie


def integrand_sampler(
    lams_: Hyper,
    lams: list[Hyper], 
    lame: list[Hyper],
    instantiate_sampler: Callable[[Hyper, np.random.Generator], Iterator[Params]],
    eval_logprior: Callable[[Params, Hyper], float],
    rng: np.random.Generator,
    n_burnin: int,
) -> Iterator[tuple[FloatArr, FloatArr]]:

    sampler = instantiate_sampler(lams_, rng)
    for _, the in zip(range(n_burnin), sampler):
        pass
    for the in sampler:
        log_psis = np.array([eval_logprior(the, lams_) for lams_ in lams])
        log_psie = np.array([eval_logprior(the, lame_) for lame_ in lame])
        yield log_psis, log_psie
