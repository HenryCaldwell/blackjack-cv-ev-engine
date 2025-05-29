from abc import ABC, abstractmethod

from typing import Any, Dict


class IHandEvaluator(ABC):
    """
    Interface for evaluating grouped blackjack hands.

    This interface defines a contract for computing expected values for each possible player action and
    choosing the best action.
    """

    @abstractmethod
    def evaluate_hands(
        self, hands: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate each player's hand and select the optimal action.

        Parameters:
            hands (Dict[str, Dict[str, Any]]): A mapping of hand IDs to their hand information.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of hand IDs to their evaluation information.
        """
        pass
