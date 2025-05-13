from abc import ABC, abstractmethod
from typing import Dict, List


class IEVCalculator(ABC):
    """
    Interface for calculating expected values (EV) of blackjack actions.

    This interface defines methods to compute the EV for standing, hitting, doubling, splitting, and surrendering
    based on the current deck and hands.
    """

    @abstractmethod
    def calculate_stand_ev(
        self, deck: Dict[int, int], player_hand: List[int], dealer_hand: List[int]
    ) -> float:
        """
        Compute the expected value when the player stands.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the stand decision.
        """
        pass

    @abstractmethod
    def calculate_hit_ev(
        self, deck: Dict[int, int], player_hand: List[int], dealer_hand: List[int]
    ) -> float:
        """
        Compute the expected value when the player hits.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the hit decision.
        """
        pass

    @abstractmethod
    def calculate_double_ev(
        self, deck: Dict[int, int], player_hand: List[int], dealer_hand: List[int]
    ) -> float:
        """
        Compute the expected value when the player doubles.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the double decision.
        """
        pass

    @abstractmethod
    def calculate_split_ev(
        self, deck: Dict[int, int], player_hand: List[int], dealer_hand: List[int]
    ) -> float:
        """
        Compute the expected value when the player splits.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the split decision.
        """
        pass

    @abstractmethod
    def calculate_surrender_ev(
        self, deck: Dict[int, int], player_hand: List[int], dealer_hand: List[int]
    ) -> float:
        """
        Compute the expected value when the player surrenders.

        Parameters:
          deck (Dict[int, int]): Mapping of card labels to remaining counts.
          player_hand (List[int]): List of card labels in the player's hand.
          dealer_hand (List[int]): List of card labels in the dealer's hand.

        Returns:
          float: The expected value (EV) for the surrender decision.
        """
        pass
