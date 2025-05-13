from abc import ABC, abstractmethod


class ICardDeck(ABC):
    """
    Interface for managing a blackjack card deck.

    This interface defines methods for updating the deck by adding or removing cards, which is useful for tracking
    the current state of play.
    """

    @abstractmethod
    def add_card(self, card_label: int) -> bool:
        """
        Add a card to the deck.

        Parameters:
          card_label (int): The numeric label of the card to be added.

        Returns:
          bool: True if the card was successfully added, False otherwise.
        """
        pass

    @abstractmethod
    def remove_card(self, card_label: int) -> bool:
        """
        Remove a card from the deck.

        Parameters:
          card_label (int): The numeric label of the card to be removed.

        Returns:
          bool: True if the card was successfully removed, False otherwise.
        """
        pass
