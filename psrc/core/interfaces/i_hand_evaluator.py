from abc import ABC, abstractmethod
from typing import Any, Dict

class IHandEvaluator(ABC):
  """
  Interface for evaluating grouped blackjack hands and determining optimal actions.

  This interface defines a method to compute the expected values (EV) for each possible player action given
  pre-grouped hand information, and to identify the best action based on those EVs.
  """

  @abstractmethod
  def evaluate_hands(self, hands_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate each player's hand and select the optimal action.

    Parameters:
      hands_info (Dict[str, Any]): A dictionary mapping hand identifiers to their details (e.g., cards, score,
      boxes).

    Returns:
      Dict[str, Any]: A dictionary of evaluation results, mapping hand identifiers to a structure containing EVs
      for each action and the best action.
    """
    pass
