from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from bisect import bisect


@dataclass
class ArmSelection():

    """A class that is used to select the multiarm bandit arm."""

    n_arms: int = 10
    count: npt.NDArray[np.int32] = np.empty(
        dtype=np.int32,
        shape=n_arms)
    values: npt.NDArray[np.float16] = np.empty(
        dtype=np.float16,
        shape=n_arms)

    def preffered_arm(self) -> int:
        """Returning the index of the maximum value in the array."""
        return int(np.argmax(self.values))

    def choose_arm(self) -> int:
        """Returning the index of the maximum value in the array."""
        return self.preffered_arm()

    def initialize(self) -> None:
        """Initializing the count and values array to zero."""
        self.count = np.zeros(shape=self.n_arms, dtype=np.int32)
        self.values = np.zeros(shape=self.n_arms, dtype=np.float16)

    def optimistic_initialization(self, high_value: float = 1.96) -> None:
        """Initializing the count array to zero and values array to an optimistic high value.
        It does in helping initial exploration"""
        self.count = np.zeros(shape=self.n_arms, dtype=np.int32)
        self.values = np.full(
            shape=self.n_arms,
            fill_value=high_value,
            dtype=np.float16)

    def update(self, arm: int, reward: float) -> None:
        """Updating the count and values array."""
        self.count[arm] = self.count[arm] + 1
        count = self.count[arm]
        old_val = self.values[arm]
        new_val = ((count - 1) / float(count)) * \
            old_val + (1 / float(count)) * reward
        self.values[arm] = new_val

    @staticmethod
    def proportional_selection(prob: list[float]):
        '''
        proportional_selection 
        =====================

        This function takes probability array as input and will output the sample 
        proportional to the probability

        Param
        ++++++

        Prob: list[float], e.g. [0.3, 0.4, 0.2, 0.1] here probability of arm 0 is 0.3, arm 1 is 0.4 and so on.
        convert the np array to prob using ndarray.tolist() option

        Return
        ------
        The arm selected randomly from that discrete probability distribution.
        '''
        ran: float = np.random.random()
        processed_prob: list[float] = [prob[i] + prob[i-1]
                                       for i in range(1, len(prob))]
        processed_prob.insert(0, prob[0])
        processed_prob[len(prob)-1] = float('inf'),
        return bisect(processed_prob, ran)


@dataclass
class EpsilonGreedy(ArmSelection):

    """
    If a random number is greater than epsilon, then return the preffered arm, otherwise return a
    random arm. The Select arm function is different from the Arm Selection here
    """
    epsilon: float = 0.1

    def __post_init__(self):
        self.initialize()

    def select_arm(self) -> int:
        """The first line is calculation 1-epsilon and exploiting randomly.
        the else condition will pick and arm randomly for epsilon times whever it want
        """
        if np.random.random() > self.epsilon:  # prob is 1-epsilon
            return self.preffered_arm()
        else:
            return np.random.randint(low=0, high=len(self.values))


@dataclass
class Softmax(ArmSelection):
    tau: float = 0.1

    def __post_init__(self):
        """this function will run after __init__ autommatically"""
        self.initialize()

    def select_arm(self) -> int:
        exp_sum = np.sum(np.exp((self.values/self.tau), dtype=np.float64))
        return -1
