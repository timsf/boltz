from typing import Callable, TypeVar, Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
StateSpace = TypeVar('StateSpace')


def sample_ensemble(
    eval_log_psi: Callable[[StateSpace], float],
    eval_log_prop: Callable[[StateSpace], float],
    sample_prop: Callable[[StateSpace], StateSpace],
    state0: list[StateSpace],
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[list[StateSpace]]:

    state = state0
    log_psi = np.array([eval_log_psi(s) for s in state])

    while True:
        for _ in range(len(state)):
            state, log_psi, log_q = update_telewalk(eval_log_psi, eval_log_prop, sample_prop, state, log_psi, log_q, ome)
        yield state


def update_telewalk(
    eval_log_psi: Callable[[StateSpace], FloatArr],
    eval_log_prop: Callable[[StateSpace], FloatArr],
    sample_prop: Callable[[StateSpace], FloatArr],
    state_nil: StateSpace,
    log_psi_nil: FloatArr,
    log_q_nil: FloatArr,
    ome: np.random.Generator,
) -> tuple[float, float, FloatArr, FloatArr, FloatArr]:

    origin = ome.choice(len(state_nil))
    parent = state_nil[origin]
    child = sample_prop(parent)
    log_psi_child = eval_log_psi(child)
    log_q_child = np.array([eval_log_prop(child, s) for s in state_nil])
    log_weights_forw = eval_log_weights(log_psi_nil, log_psi_child, log_q_nil, log_q_child)
    destination = ome.choice(len(state_nil), p=np.exp(log_weights_forw - logsumexp(log_weights_forw)))

    state_prime = state_nil.copy()
    log_psi_prime = log_psi_nil.copy()
    log_q_prime = log_q_nil.copy()
    state_prime[destination] = child
    log_psi_prime[destination] = log_psi_child
    log_q_prime[destination] = log_q_child
    log_psi_parent = eval_log_psi(parent)
    log_q_parent = np.array([eval_log_prop(parent, s) for s in state_prime])
    log_weights_back = eval_log_weights(log_psi_prime, log_psi_parent, log_q_prime, log_q_parent)

    log_odds = log_psi_parent - log_psi_child + logsumexp(log_weights_forw) - logsumexp(log_weights_back)
    log_acc_prob = min(0, log_odds)
    if np.log(ome.uniform()) < log_acc_prob:
        # if origin != destination:
        #     a = 0
        return state_prime, log_psi_prime, log_q_prime
    return state_nil, log_psi_nil, log_q_nil


def eval_log_weights(
    log_psi: FloatArr,
    log_psi_child: FloatArr,
    log_q: FloatArr,
    log_q_child: FloatArr,
) -> FloatArr:

    log_q_jump = log_q.copy()
    log_q_jump[np.diag_indices_from(log_q_jump)] = log_q_child
    log_weights = logsumexp(log_q_jump, 0) + log_psi_child - log_psi
    return log_weights
