from abc import ABC, abstractmethod
from typing import Dict, List

class IEVCalculator(ABC):
  """
  Interface for calculating expected values (EV) for blackjack actions.

  This interface defines a method for computing the EV of various actions (e.g., hit, stand, double, split) based
  on the current deck composition and the hands of the player and dealer.
  """

  @abstractmethod
  def calculate_ev(self, action: str, deck: Dict[int, int],
                    player_hand: List[int], dealer_hand: List[int]) -> float:
    """
    Calculate the expected value (EV) for a given blackjack action.

    Parameters:
      action (str): The blackjack action to evaluate (e.g., "stand", "hit", "double", "split").
      deck (Dict[int, int]): The current deck state as a mapping from card labels to counts.
      player_hand (List[int]): The list of card labels in the player's hand.
      dealer_hand (List[int]): The list of card labels in the dealer's hand.

    Returns:
      float: The computed expected value for the specified action.
    """
    pass
