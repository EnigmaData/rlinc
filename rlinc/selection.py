"""
This modules has all the algorithms related to the multiarm bandit problems
ranging from absolute greedy to state of the art UCB algorithm.
"""
from bisect import bisect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt


@dataclass
class ArmSelection():

    """A class that is used to select the multiarm bandit arm."""

    _n_arms: int = 10
    count: npt.NDArray[np.int32] = np.empty(
        dtype=np.int32,
        shape=_n_arms)
    values: npt.NDArray[np.float16] = np.empty(
        dtype=np.float16,
        shape=_n_arms)

    @classmethod
    def from_array(cls, values: list[float] or npt.NDArray[np.float_]):
        """
        Alternate constructor for construction from value array.
        """
        values_np: npt.NDArray[np.float16] = np.array(values, dtype=np.float16)
        _n_arms = len(values_np)
        count_np: npt.NDArray[np.int32] = np.zeros(
            shape=_n_arms, dtype=np.int32)
        return cls(_n_arms, count_np, values_np)

    def preffered_arm(self) -> int:
        """Returning the index of the maximum value in the array."""
        return int(np.argmax(self.values))

    def choose_arm(self) -> int:
        """Returning the index of the maximum value in the array."""
        return self.preffered_arm()

    def initialize(self) -> None:
        """Initializing the count and values array to zero."""
        self.count = np.zeros(shape=self._n_arms, dtype=np.int32)
        self.values = np.zeros(shape=self._n_arms, dtype=np.float16)

    def optimistic_initialization(self, high_value: float = 1.96) -> None:
        """Initializing the count array to zero and values array to an
        optimistic high value. It does in helping initial exploration"""
        self.count = np.zeros(shape=self._n_arms, dtype=np.int32)
        self.values = np.full(
            shape=self._n_arms,
            fill_value=high_value,
            dtype=np.float16)

    def update(self, arm: int, reward: float, alpha: float | object = None) -> None:
        """Updating the count and values array.
        You can also provide alpha parameter for non-stationary bandits

        Note -
        ----

        Provide value of alpha between 0 and 1
        """
        self.count[arm] = self.count[arm] + 1
        count = self.count[arm]
        old_val = self.values[arm]
        if not alpha:
            new_val = ((count - 1) / float(count)) * \
                old_val + (1 / float(count)) * reward
            self.values[arm] = new_val
        else:
            self.values[arm] = old_val + alpha*(reward - old_val)

    def change_count(self, count: list[int]
                     or npt.NDArray[np.int_], new_arr: bool = False) -> None:
        """It intends to change the count array"""
        if (length := len(count)) != self._n_arms:
            if not new_arr:
                self._n_arms = length
        self.count = np.array(count, dtype=np.int32)

    @staticmethod
    def proportional_selection(prob: list[float]) -> int:
        '''
        proportional_selection
        =====================

        This function takes probability array as input and will output the sample
        proportional to the probability

        Param
        -----

        Prob: list[float], e.g. [0.3, 0.4, 0.2, 0.1] here probability of arm
        0 is 0.3, arm 1 is 0.4 and so on.
        convert the np array to prob using ndarray.tolist() option

        Return
        ------
        The arm selected randomly from that discrete probability distribution.
        '''
        ran: float = np.random.random()
        processed_prob: list[float] = [prob[i] + prob[i - 1]
                                       for i in range(1, len(prob))]
        processed_prob.insert(0, prob[0])
        processed_prob[-1] = float('inf')
        return bisect(processed_prob, ran)


@dataclass
class EpsilonGreedy(ArmSelection):

    """
    If a random number is greater than epsilon, then return the preffered arm, otherwise return a
    random arm. The Select arm function is different from the Arm Selection here
    """

    epsilon: float = 0.1

    def __post_init__(self) -> None:
        """adding for the sake of pylint"""
        self.initialize()

    def select_arm(self) -> int:
        """The first line is calculation 1-epsilon and exploiting randomly.
        the else condition will pick and arm randomly for epsilon times whever it want
        """
        if np.random.random() > self.epsilon:  # prob is 1-epsilon
            return self.preffered_arm()
        return np.random.randint(low=0, high=len(self.values))


@dataclass
class AnnealingEpsilonGreedy(EpsilonGreedy):

    """
    Adding annuling epsilon greedy, we are expecting that it will accept
    functions as input and reduces the epsilon so as to reduce overexploration
    after a long time.
    """

    func: Callable[[int], float] = lambda x: 1 - np.sin(x + 3)

    def __post_init__(self) -> None:
        self.epsilon = self.func(0)

    def annulator(update_func: Callable):  # type: ignore ## for VS Code py-lance
        """Decorator to update epsilon after every itteration"""

        def inner_func(self, *args, **kwargs):
            update_func(args, kwargs)
            self.epsilon -= self.func(int(np.sum(self.count)))
        return inner_func

    update = annulator(super().update)
    # We would need to accept an annuling function and
    # change the epsilon accordingly


@dataclass
class Softmax(ArmSelection):

    """Softmax algorithm implementation"""

    tau: float = 0.1

    def __post_init__(self):
        """this function will run after __init__ autommatically"""
        self.initialize()

    def select_arm(self) -> int:
        """Selecting the arm with highest exponential proportion"""
        exp_sum: float = np.sum(np.exp((self.values / self.tau),
                                       dtype=np.float64), dtype=np.float64)
        prob_dist = [np.exp(val / self.tau, dtype=np.float64) /
                     exp_sum for val in self.values]
        del exp_sum
        return Softmax.proportional_selection(prob=prob_dist)


@dataclass
class UCB1(ArmSelection):

    """It is the implementation of the state of the art UCB algorithm."""

    beta: float = 0.99

    def select_arm(self) -> int:
        """Selecting the arm with highest upper-bound."""
        if (arms := np.sum(self.count)) < self._n_arms:
            return int(arms)

        confidence_half: npt.NDArray[np.float32] = self.beta * \
            np.sqrt(arms * np.reciprocal(self.values, dtype=np.float32))
        ucb_t: npt.NDArray[np.float32] = np.add(
            self.values, confidence_half, dtype=np.float32)
        return int(np.argmax(ucb_t))
