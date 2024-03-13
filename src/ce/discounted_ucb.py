import math
from typing import List

class DiscountedUCB:
    """Implementation of the Discounted Upper Confidence Bound (UCB) algorithm.
    Attributes:
        count: A list containing the counts of times each arm has been played.
        value: A list containing the estimated value of each arm.
        gamma: The discount factor, a value between 0 and 1, where closer to 1 means less discounting.
        total_plays: Total number of plays across all arms.
    """
    def __init__(self, n_arms: int, gamma: float):
        """Initializes the DiscountedUCB with a specified number of arms and discount factor.

        Args:
            n_arms: The number of arms in the bandit problem.
            gamma: The discount factor used in value updates.
        """
        self.n_arms = n_arms
        self.gamma = gamma
        self.counts = [1.0] * n_arms  # Initialize counts of arm pulls to zero
        self.values = [0.0] * n_arms  # Initialize estimated values of arms to zero
        self.total_plays = 1

    def select_arm(self) -> int:
        """Selects an arm based on the highest Upper Confidence Bound.
        Returns:
            The index of the selected arm.
        """
        # # If any arm hasn't been played yet, select it
        # for arm in range(self.n_arms):
        #     if self.counts[arm] == 0:
        #         return arm

        ucb_values = [0.0 for _ in range(self.n_arms)]
        for arm in range(self.n_arms):
            # Calculate UCB value for each arm
            bonus = math.sqrt((2 * math.log(self.total_plays)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        # Select the arm with the highest UCB value
        chosen_arm =  ucb_values.index(max(ucb_values))
        self.total_plays += 1
        self.counts[chosen_arm] += 1
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """Updates the estimated values of the chosen arm based on the received reward.
        Args:
            chosen_arm: The index of the arm that was played.
            reward: The reward received from playing the chosen arm.
        """

        # Update the estimated value of the chosen arm with discounting
        # n = self.counts[chosen_arm]
        # self.values[chosen_arm] = (self.gamma * self.values[chosen_arm]) + ((1 - self.gamma) * reward)
        self.values[chosen_arm] = (self.gamma * self.values[chosen_arm]) +  reward

