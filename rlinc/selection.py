from dataclasses import dataclass
import numpy as np


@dataclass
class ArmSelection():

    """A class that is used to select the multiarm bandit arm."""

    n_arms: int = 10
    count: np.ndarray(
        dtype=np.int32,
        shape=n_arms) = np.empty(
        dtype=np.int32,
        shape=n_arms)
    values: np.ndarray(
        dtype=np.float16,
        shape=n_arms) = np.empty(
        dtype=np.int32,
        shape=n_arms)

    def preffered_arm(self) -> int:
        """Returning the index of the maximum value in the array."""
        return np.argmax(self.values)

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

    def update(self, arm: int, reward: float) -> float:
        """Updating the count and values array."""
        self.count[arm] = self.count[arm] + 1
        count = self.count[arm]
        old_val = self.values[arm]
        new_val = ((count - 1) / float(count)) * \
            old_val + (1 / float(count)) * reward
        self.values[arm] = new_val


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

    @staticmethod
    def proportional_selection():
        '''
        proportional_selection 
        ----------------------

        __ summary __

        This function takes probability array as input and will output the sample 
        proportional to the probability
        '''
        ran: float = np.random.random()
        cum_prob: float = 0.0

        pass

    def select_arm(self) -> int:
        pass
