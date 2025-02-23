from itertools import product
from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

import boltz.tools.infemus
from boltz import samplers


BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def est_likelihood(
    v: BoolArr,
    b: list[FloatArr],
    w: list[FloatArr],
    n_temps: int,
    n_gibbs: int,
    n_walkers: int,
    rng: np.random.Generator,
) -> float:
    
    def eval_logprior(params, hyper) -> float:
        return samplers.eval_target(params, o, *hyper)
    
    def instantiate_sampler(hyper, rng) -> Iterator[list[BoolArr]]:
        x = [samplers.initialize(n_gibbs, [rng.uniform(size=(1, len(b_))) < .5 for b_ in b], len(b) * [True], o, hyper[0], hyper[1], rng) for _ in range(n_walkers)]
        return (sample[rng.choice(len(sample))] for sample in samplers.sample_ensemble(x, len(b) * [True], o, hyper[0], hyper[1], rng))
        # while True:
        #     x = [samplers.sample_kernel(x_, len(b) * [True], o, hyper[0], hyper[1], rng) for x_ in x]
        #     yield x[rng.choice(len(sample))]
    
    o = [np.zeros_like(b_) for b_ in b]
    temps = np.linspace(0, 1, n_temps)
    path = [([b_ * t for b_ in b], [w_ * t for w_ in w]) for t in temps]
    z = boltz.tools.infemus.est_mlik(path, path, instantiate_sampler, eval_logprior, rng, n_gibbs, n_gibbs)
    return samplers.eval_elbo(v, o, b, w, n_gibbs) + v.shape[0] * (np.log(z[0]) - np.log(z[-1]))


def eval_likelihood(
    v: BoolArr,
    o: list[FloatArr],
    b: list[FloatArr],
    w: list[FloatArr],
) -> float:
    
    shapes = [len(o_) for o_ in o]
    h = np.array(list(product([False, True], repeat=sum(shapes[1:]))))
    x = np.array(list(product([False, True], repeat=sum(shapes))))
    num = [logsumexp([samplers.eval_target([v_] + np.split(h_, np.cumsum(shapes[1:])[:-1]), o, b, w) for h_ in h]) for v_ in v]
    denom = logsumexp([samplers.eval_target(np.split(x_, np.cumsum(shapes)[:-1]), o, b, w) for x_ in x])
    return sum(num) - v.shape[0] * denom
